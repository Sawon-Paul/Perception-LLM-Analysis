"""Find the true video/sensor time offset automatically.

The idea: when the vehicle moves, the video scene changes. When it stops, the
scene freezes. So frame-to-frame pixel difference is a proxy for speed. Sensors
record actual speed. Cross-correlate the two signals and the lag that maximises
correlation is your offset.

This beats scrubbing to a speed bump by hand, and it works the same way on all
nine traces.

    python -m src.detect.sync_finder --trace "PVS 2"
    python -m src.detect.sync_finder --all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

PROBE_FPS = 2.0          # sampling rate for both signals
MAX_LAG_S = 120.0        # widest offset we will consider, either direction


def video_motion(video: Path, probe_fps: float = PROBE_FPS) -> np.ndarray:
    """Mean absolute frame-to-frame difference, sampled at probe_fps.

    Decoded straight from ffmpeg as tiny greyscale frames — no temp files,
    no OpenCV dependency, a few seconds for a 20 minute video.
    """
    w, h = 64, 36
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"fps={probe_fps},scale={w}:{h},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[:300].decode(errors='ignore')}")

    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(buf) // (w * h)
    frames = buf[:n * w * h].reshape(n, h * w).astype(np.float32)
    if n < 2:
        raise RuntimeError("video too short to measure motion")
    return np.abs(np.diff(frames, axis=0)).mean(axis=1)


def sensor_speed(sensor_csv: Path, probe_fps: float = PROBE_FPS) -> tuple[np.ndarray, float]:
    """Speed resampled onto a regular probe_fps grid. Returns (signal, t0)."""
    df = pd.read_csv(sensor_csv, usecols=["timestamp", "speed"]).sort_values("timestamp")
    t = df["timestamp"].to_numpy(dtype=float)
    v = df["speed"].to_numpy(dtype=float)
    t0, t1 = t[0], t[-1]
    grid = np.arange(t0, t1, 1.0 / probe_fps)
    return np.interp(grid, t, v), float(t0)


def _z(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    s = x.std()
    return x / s if s > 1e-9 else x


def best_lag(motion: np.ndarray, speed: np.ndarray,
             probe_fps: float = PROBE_FPS) -> tuple[float, float, np.ndarray]:
    """Lag in seconds that best aligns motion to speed, plus peak correlation."""
    m, s = _z(motion), _z(speed)
    max_lag = int(MAX_LAG_S * probe_fps)

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.empty(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        # lag > 0: video frame 0 sits lag samples INTO the sensor log
        if lag >= 0:
            a, b = m[:len(m) - 0], s[lag:]
        else:
            a, b = m[-lag:], s
        n = min(len(a), len(b))
        corrs[i] = np.dot(a[:n], b[:n]) / n if n > probe_fps * 30 else -np.inf

    k = int(np.argmax(corrs))
    return float(lags[k] / probe_fps), float(corrs[k]), corrs


def analyse(trace: str) -> dict:
    pvs = cfg.pvs_dir(trace)
    video = pvs / "video_environment.mp4"
    sensor_csv = pvs / "dataset_gps_mpu_left.csv"

    motion = video_motion(video)
    speed, t0 = sensor_speed(sensor_csv)

    lag_s, peak, corrs = best_lag(motion, speed)
    runner_up = float(np.sort(corrs[np.isfinite(corrs)])[-int(10 * PROBE_FPS)]) \
        if np.isfinite(corrs).sum() > 10 * PROBE_FPS else 0.0
    margin = peak - runner_up

    vid_s = len(motion) / PROBE_FPS
    sen_s = len(speed) / PROBE_FPS

    return {
        "trace": trace,
        "video_duration_s": round(vid_s, 2),
        "sensor_span_s": round(sen_s, 2),
        "duration_gap_s": round(sen_s - vid_s, 2),
        "t0_offset_s": round(lag_s, 2),
        "peak_correlation": round(peak, 4),
        "margin_over_nearby": round(margin, 4),
        "sensor_t0": t0,
    }


def verdict(r: dict) -> str:
    if r["peak_correlation"] < 0.25:
        return ("WEAK - correlation too low to trust. The vehicle may never stop "
                "in this trace, leaving nothing to align on. Fall back to the "
                "manual speed-bump method.")
    if r["margin_over_nearby"] < 0.05:
        return ("AMBIGUOUS - several offsets score nearly the same. Treat the "
                "number as a starting guess and confirm by eye.")
    return "STRONG - use this offset."


def report(r: dict) -> None:
    print(f"\n=== {r['trace']} ===")
    print(f"  video          : {r['video_duration_s']} s")
    print(f"  sensors        : {r['sensor_span_s']} s")
    print(f"  gap            : {r['duration_gap_s']} s")
    print(f"  best offset    : {r['t0_offset_s']:+.2f} s")
    print(f"  correlation    : {r['peak_correlation']:.3f} "
          f"(margin {r['margin_over_nearby']:.3f})")
    print(f"  verdict        : {verdict(r)}")
    if r["peak_correlation"] >= 0.25:
        print(f"\n  python -m src.detect.extract_frames --trace \"{r['trace']}\" "
              f"--t0-offset {r['t0_offset_s']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--all", action="store_true", help="every trace in DATASET_DIR")
    args = ap.parse_args()

    if args.all:
        traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                        if d.is_dir() and d.name.upper().startswith("PVS"))
    else:
        traces = [args.trace]

    results = []
    for tr in traces:
        try:
            r = analyse(tr)
            report(r)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {tr} ===\n  FAILED: {str(e)[:200]}")

    out = cfg.DATA / "sync_offsets.json"
    out.write_text(json.dumps({r["trace"]: r for r in results}, indent=2))
    print(f"\nsaved -> {out}")

    if len(results) > 1:
        print("\nsummary")
        print(f"{'trace':<10}{'offset_s':>10}{'corr':>8}  verdict")
        for r in results:
            print(f"{r['trace']:<10}{r['t0_offset_s']:>+10.2f}"
                  f"{r['peak_correlation']:>8.3f}  {verdict(r).split(' -')[0]}")


if __name__ == "__main__":
    main()
