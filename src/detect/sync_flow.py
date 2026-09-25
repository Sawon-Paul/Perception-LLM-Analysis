"""Sync attempt 3: optical flow, plus a visual check that does not rely on any model.

Why the earlier attempts failed:

  attempt 1  mean frame-to-frame pixel difference vs speed. A forward dashcam
             changes almost as much at 40 km/h as at 80, so the signal saturates
             and carries little speed information. All nine traces came out weak.

  attempt 2  assume the video spans the whole sensor log with wrong fps metadata.
             Only 2 of 9 fitted. The rest are variable-frame-rate recordings whose
             container duration is probably honest.

Attempt 3 uses optical flow magnitude, which really is proportional to speed for a
forward-facing camera. If that also fails, use --verify, which checks alignment
against something no inference can fake: at the moment of a large measured jolt,
the road ahead should visibly contain a defect.

    python -m src.detect.sync_flow --trace "PVS 2"
    python -m src.detect.sync_flow --all
    python -m src.detect.sync_flow --trace "PVS 2" --verify --offset 35
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

PROBE_FPS = 2.0
MAX_LAG_S = 150.0
FW, FH = 160, 90


def _decode_gray(video: Path, fps: float, w: int, h: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"fps={fps},scale={w}:{h},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[:300].decode(errors='ignore')}")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(buf) // (w * h)
    return buf[:n * w * h].reshape(n, h, w)


def flow_speed_proxy(video: Path, fps: float = PROBE_FPS) -> np.ndarray:
    """Median optical-flow magnitude per frame pair. Tracks real speed."""
    import cv2

    frames = _decode_gray(video, fps, FW, FH)
    if len(frames) < 2:
        raise RuntimeError("video too short")

    out = np.empty(len(frames) - 1, dtype=np.float32)
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # lower half of the frame is road surface; sky and trees add noise
        out[i] = np.median(mag[FH // 2:, :])
    return out


def sensor_speed(sensor_csv: Path, fps: float = PROBE_FPS) -> tuple[np.ndarray, float]:
    df = pd.read_csv(sensor_csv, usecols=["timestamp", "speed"]).sort_values("timestamp")
    t = df["timestamp"].to_numpy(dtype=float)
    v = df["speed"].to_numpy(dtype=float)
    grid = np.arange(t[0], t[-1], 1.0 / fps)
    return np.interp(grid, t, v), float(t[0])


def _z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean()
    s = x.std()
    return x / s if s > 1e-9 else x


def best_lag(a: np.ndarray, b: np.ndarray, fps: float = PROBE_FPS):
    """Lag in seconds aligning signal a (video) into signal b (sensors)."""
    a, b = _z(a), _z(b)
    max_lag = int(MAX_LAG_S * fps)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.full(len(lags), -np.inf)
    min_overlap = int(fps * 120)     # need 2 minutes of overlap to mean anything
    for i, lag in enumerate(lags):
        x, y = (a, b[lag:]) if lag >= 0 else (a[-lag:], b)
        n = min(len(x), len(y))
        if n >= min_overlap:
            corr[i] = float(np.dot(x[:n], y[:n]) / n)
    k = int(np.argmax(corr))
    finite = corr[np.isfinite(corr)]
    # margin against the best value at least 15 s away from the peak
    far = np.concatenate([corr[:max(0, k - int(15 * fps))], corr[k + int(15 * fps):]])
    far = far[np.isfinite(far)]
    margin = float(corr[k] - far.max()) if len(far) else 0.0
    return float(lags[k] / fps), float(corr[k]), margin, len(finite)


def analyse(trace: str) -> dict:
    pvs = cfg.pvs_dir(trace)
    flow = flow_speed_proxy(pvs / "video_environment.mp4")
    speed, t0 = sensor_speed(pvs / "dataset_gps_mpu_left.csv")
    lag, peak, margin, _ = best_lag(flow, speed)

    speed_cv = float(np.std(speed) / (np.mean(speed) + 1e-9))
    return {
        "trace": trace,
        "t0_offset_s": round(lag, 2),
        "correlation": round(peak, 4),
        "margin": round(margin, 4),
        "speed_variation": round(speed_cv, 3),
        "verdict": ("STRONG" if peak >= 0.40 and margin >= 0.08 else
                    "MODERATE" if peak >= 0.25 else "WEAK"),
    }


# ---------------------------------------------------------------- verify mode

SHEET = """<!doctype html><meta charset="utf-8"><title>sync verification</title>
<style>body{font-family:system-ui;margin:24px;background:#111;color:#eee}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}
.c{background:#1c1c1c;border-radius:8px;padding:10px}
.c img{width:100%;border-radius:4px}
.m{font-size:12px;color:#aaa;margin-top:6px;line-height:1.5}
h1{font-size:19px}.n{color:#f2b544;font-size:13px;margin-bottom:18px}</style>
<h1>Sync verification &mdash; offset {offset} s</h1>
<p class="n">These are the frames the pipeline would pair with the largest measured
jolts. If the offset is right, most should show a pothole, bump, or rough patch in
the road ahead. If they show smooth road, the offset is wrong.</p>
<div class="g">
"""


def verify(trace: str, offset: float, top_n: int = 12) -> None:
    pvs = cfg.pvs_dir(trace)
    sensor_csv = pvs / "dataset_gps_mpu_left.csv"

    df = pd.read_csv(sensor_csv, usecols=["timestamp", "acc_z_below_suspension", "speed"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    z = df["acc_z_below_suspension"]
    df["jolt"] = (z - z.rolling(100, center=True, min_periods=1).mean()).abs()
    t0 = float(df["timestamp"].min())

    df = df[df["speed"] > 2.0]                      # ignore jolts while parked
    picks = df.nlargest(top_n * 8, "jolt")
    chosen, used = [], []
    for _, r in picks.iterrows():
        ts = float(r["timestamp"])
        if all(abs(ts - u) > 8.0 for u in used):    # spread them out
            used.append(ts)
            chosen.append(r)
        if len(chosen) >= top_n:
            break

    out_dir = cfg.DATA / "sync_check" / cfg.slug(trace)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    cards = []
    for i, r in enumerate(chosen):
        video_t = float(r["timestamp"]) - t0 - offset
        if video_t < 0:
            continue
        img = out_dir / f"jolt_{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{video_t:.3f}",
                        "-i", str(pvs / "video_environment.mp4"),
                        "-frames:v", "1", "-q:v", "2", str(img)], check=False)
        if img.exists():
            cards.append(
                f'<div class="c"><img src="file://{img.resolve()}">'
                f'<div class="m">video t = {video_t:.1f} s<br>'
                f'jolt = {r["jolt"]:.2f} m/s²<br>'
                f'speed = {r["speed"] * 3.6:.0f} km/h</div></div>')

    sheet = out_dir / "verify.html"
    sheet.write_text(SHEET.replace("{offset}", f"{offset:g}") + "\n".join(cards) + "</div>",
                     encoding="utf-8")
    print(f"{len(cards)} frames -> {sheet}")
    print("\nOpen it. Do most frames show a defect in the road ahead?")
    print("  yes -> this offset is right")
    print("  no  -> try another offset, or the video and sensors may not be alignable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--offset", type=float, default=0.0)
    args = ap.parse_args()

    if args.verify:
        verify(args.trace, args.offset)
        return

    traces = (sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                     if d.is_dir() and d.name.upper().startswith("PVS"))
              if args.all else [args.trace])

    results = []
    for tr in traces:
        try:
            print(f"analysing {tr} ...", flush=True)
            r = analyse(tr)
            results.append(r)
            print(f"  offset {r['t0_offset_s']:+.2f} s, corr {r['correlation']:.3f}, "
                  f"margin {r['margin']:.3f}  -> {r['verdict']}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {str(e)[:200]}")

    if results:
        out = cfg.DATA / "sync_flow.json"
        out.write_text(json.dumps({r["trace"]: r for r in results}, indent=2))
        print(f"\n{'trace':<9}{'offset':>9}{'corr':>8}{'margin':>8}  verdict")
        for r in results:
            print(f"{r['trace']:<9}{r['t0_offset_s']:>+9.2f}{r['correlation']:>8.3f}"
                  f"{r['margin']:>8.3f}  {r['verdict']}")
        print(f"\nsaved -> {out}")
        print("\nConfirm any STRONG result before trusting it:")
        print(f'  python -m src.detect.sync_flow --trace "{results[0]["trace"]}" '
              f'--verify --offset {results[0]["t0_offset_s"]}')


if __name__ == "__main__":
    main()
