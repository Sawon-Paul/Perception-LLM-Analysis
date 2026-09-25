"""Work out how video time maps to sensor time, for every trace.

Cross-correlating motion against speed failed on all nine traces. The numbers
point somewhere else: for PVS 2, nb_frames / sensor_span = 23.99, a suspiciously
clean 24 fps, while the container claims 30.

So the likely story is not an offset at all. The video probably spans the whole
trace and the duration metadata is simply wrong.

This script tests three hypotheses per trace and reports which one fits:

  A  container fps is right, video is a subset of the trace  -> needs an offset
  B  video spans the full sensor log, fps metadata is wrong  -> stretch to fit
  C  neither                                                 -> investigate by hand

    python -m src.detect.sync_diagnose --all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

COMMON_FPS = [15.0, 20.0, 23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]


def probe(video: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
           "-show_entries", "format=duration",
           "-of", "json", str(video)]
    j = json.loads(subprocess.check_output(cmd))
    s = j["streams"][0]

    def _rate(v):
        if not v or v == "0/0":
            return None
        n, d = v.split("/")
        return float(n) / float(d) if float(d) else None

    nb = s.get("nb_frames")
    nb = int(nb) if nb not in (None, "N/A") else None
    dur = s.get("duration") or j.get("format", {}).get("duration")
    return {
        "r_frame_rate": _rate(s.get("r_frame_rate")),
        "avg_frame_rate": _rate(s.get("avg_frame_rate")),
        "nb_frames": nb,
        "container_duration_s": float(dur) if dur else None,
    }


def count_frames_exact(video: Path) -> int:
    """Decode-accurate frame count. Slower, but nb_frames is often absent or wrong."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)])
    return int(out.decode().strip().split(",")[0])


def nearest_standard_fps(fps: float, tol: float = 0.02) -> float | None:
    for c in COMMON_FPS:
        if abs(fps - c) / c <= tol:
            return c
    return None


def diagnose(trace: str, exact: bool = False) -> dict:
    pvs = cfg.pvs_dir(trace)
    video = pvs / "video_environment.mp4"
    sensor_csv = pvs / "dataset_gps_mpu_left.csv"

    meta = probe(video)
    ts = pd.read_csv(sensor_csv, usecols=["timestamp"])["timestamp"]
    span = float(ts.max() - ts.min())

    n = count_frames_exact(video) if exact else meta["nb_frames"]
    if n is None:
        n = count_frames_exact(video)

    fps_from_duration = n / meta["container_duration_s"] if meta["container_duration_s"] else None
    fps_if_full_span = n / span

    std = nearest_standard_fps(fps_if_full_span)
    duration_at_container_fps = n / meta["r_frame_rate"] if meta["r_frame_rate"] else None

    # Hypothesis B: video spans the whole sensor log
    span_error_s = abs(n / std - span) if std else None
    b_fits = std is not None and span_error_s is not None and span_error_s < 3.0

    # Hypothesis A: container fps right, video is a shorter window of the trace
    a_fits = (duration_at_container_fps is not None
              and duration_at_container_fps < span - 3.0)

    return {
        "trace": trace,
        "n_frames": n,
        "container_fps": round(meta["r_frame_rate"], 4) if meta["r_frame_rate"] else None,
        "container_duration_s": round(meta["container_duration_s"], 2),
        "sensor_span_s": round(span, 2),
        "fps_from_container_duration": round(fps_from_duration, 4) if fps_from_duration else None,
        "fps_if_video_spans_trace": round(fps_if_full_span, 4),
        "nearest_standard_fps": std,
        "span_error_if_B_s": round(span_error_s, 3) if span_error_s is not None else None,
        "hypothesis": "B_stretch" if b_fits else ("A_offset" if a_fits else "C_unknown"),
    }


def report(r: dict) -> None:
    print(f"\n=== {r['trace']} ===")
    print(f"  frames                     : {r['n_frames']}")
    print(f"  container says             : {r['container_fps']} fps, "
          f"{r['container_duration_s']} s")
    print(f"  frames / container duration: {r['fps_from_container_duration']} fps")
    print(f"  sensor span                : {r['sensor_span_s']} s")
    print(f"  frames / sensor span       : {r['fps_if_video_spans_trace']} fps"
          + (f"   -> matches standard {r['nearest_standard_fps']} fps"
             if r["nearest_standard_fps"] else "   -> no standard rate nearby"))
    if r["span_error_if_B_s"] is not None:
        print(f"  error if video spans trace : {r['span_error_if_B_s']} s")
    print(f"  VERDICT                    : {r['hypothesis']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--exact", action="store_true",
                    help="decode-count frames instead of trusting nb_frames (slow)")
    args = ap.parse_args()

    traces = (sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                     if d.is_dir() and d.name.upper().startswith("PVS"))
              if args.all else [args.trace])

    results = []
    for tr in traces:
        try:
            r = diagnose(tr, exact=args.exact)
            report(r)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {tr} ===\n  FAILED: {str(e)[:200]}")

    out = cfg.DATA / "sync_diagnosis.json"
    out.write_text(json.dumps({r["trace"]: r for r in results}, indent=2))

    print("\n" + "=" * 74)
    print(f"{'trace':<9}{'frames':>8}{'cont_fps':>10}{'fps_if_span':>13}"
          f"{'std':>7}{'err_s':>8}  verdict")
    for r in results:
        print(f"{r['trace']:<9}{r['n_frames']:>8}{r['container_fps'] or 0:>10.3f}"
              f"{r['fps_if_video_spans_trace']:>13.3f}"
              f"{(r['nearest_standard_fps'] or 0):>7.2f}"
              f"{(r['span_error_if_B_s'] if r['span_error_if_B_s'] is not None else -1):>8.2f}"
              f"  {r['hypothesis']}")

    votes = {}
    for r in results:
        votes[r["hypothesis"]] = votes.get(r["hypothesis"], 0) + 1
    print(f"\ntally: {votes}")
    if votes.get("B_stretch", 0) >= len(results) - 1:
        print("\nEvery trace fits B: the video spans the whole sensor log and the")
        print("fps metadata is wrong. Re-extract with --stretch and no offset.")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
