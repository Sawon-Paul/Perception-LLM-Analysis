"""Step 1: video_environment.mp4 -> timestamped frames joined to PVS sensor rows.

Only video_environment.mp4 is road footage. video_dataset_left.mp4 and
video_dataset_right.mp4 are animated sensor plots; the video_environment_dataset_*
files are composites of the two. Running YOLO on those produces nothing useful.

Sync assumption: video frame 0 == first timestamp in dataset_gps_mpu_<side>.csv.
This script checks it and warns. Verify manually before trusting results.

    python -m src.detect.extract_frames --side left --fps 4
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

SENSOR_COLS = [
    "timestamp", "latitude", "longitude", "speed",
    "acc_x_below_suspension", "acc_y_below_suspension", "acc_z_below_suspension",
    "gyro_x_below_suspension", "gyro_y_below_suspension", "gyro_z_below_suspension",
    "acc_z_above_suspension", "acc_z_dashboard",
]


def probe(video: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,nb_frames,duration,width,height",
           "-of", "json", str(video)]
    s = json.loads(subprocess.check_output(cmd))["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nb = int(s["nb_frames"]) if s.get("nb_frames", "N/A") != "N/A" else None
    dur = float(s["duration"]) if s.get("duration") else (nb / fps if nb else None)
    return {"fps": fps, "n_frames": nb, "duration_s": dur,
            "width": int(s["width"]), "height": int(s["height"])}


def verify_sync(meta: dict, sensor_csv: Path) -> float:
    ts = pd.read_csv(sensor_csv, usecols=["timestamp"])["timestamp"]
    span = float(ts.max() - ts.min())
    delta = abs(meta["duration_s"] - span)
    print(f"  video duration : {meta['duration_s']:.2f} s")
    print(f"  sensor span    : {span:.2f} s")
    print(f"  difference     : {delta:.2f} s")
    if delta > 2.0:
        print("  WARNING: >2 s mismatch — frame 0 is probably NOT the first sensor row.")
        print("  Anchor manually: pick a row where speed_bump_asphalt == 1, note its")
        print("  offset from t0, scrub the video there, and pass --t0-offset.")
    else:
        print("  OK: durations agree within 2 s.")
    return delta


def extract(video: Path, out_dir: Path, target_fps: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vf", f"fps={target_fps}", "-q:v", "2",
        str(out_dir / "frame_%06d.jpg"),
    ], check=True)


def build_index(out_dir: Path, t0: float, target_fps: float,
                span_s: float | None = None) -> pd.DataFrame:
    """Timestamp every extracted frame.

    Default: frame i sits i/target_fps seconds after t0. Correct when the
    container fps is trustworthy.

    --stretch (span_s given): frames are spread evenly across the whole sensor
    log instead. Use this when the fps metadata is wrong, which sync_diagnose
    will tell you.
    """
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"no frames in {out_dir} — did ffmpeg run?")
    n = len(frames)
    if span_s is not None and n > 1:
        step = span_s / (n - 1)
        stamps = [t0 + i * step for i in range(n)]
        print(f"  stretch mode: {n} frames spread over {span_s:.2f} s "
              f"({1 / step:.3f} effective fps)")
    else:
        stamps = [t0 + i / target_fps for i in range(n)]
    return pd.DataFrame({
        "frame_path": [str(p) for p in frames],
        "frame_idx": range(n),
        "timestamp": stamps,
    })


def load_sensors(sensor_csv: Path) -> pd.DataFrame:
    sens = pd.read_csv(sensor_csv, usecols=lambda c: c in SENSOR_COLS)
    sens = sens.sort_values("timestamp").reset_index(drop=True)
    w = cfg.SENSOR_HZ  # 1 s window
    z = sens["acc_z_below_suspension"]
    # subtract the local mean: removes gravity AND road slope, leaves the jolt
    sens["jolt_z"] = (z - z.rolling(w, center=True, min_periods=1).mean()).abs()
    sens["jolt_z_std_1s"] = z.rolling(w, center=True, min_periods=1).std()
    sens["jolt_z_max_1s"] = sens["jolt_z"].rolling(w, center=True, min_periods=1).max()

    # The camera sees a defect before the wheels reach it. At 20 km/h a pothole
    # 15 m ahead is struck about 3 s later, so the jolt at the frame's own
    # timestamp describes road the car has ALREADY crossed, not what is in view.
    # Look forward instead.
    for secs in (2, 3, 5):
        fwd = sens["jolt_z"][::-1].rolling(w * secs, min_periods=1).max()[::-1]
        sens[f"jolt_z_ahead_{secs}s"] = fwd
    sens["speed_kmh"] = sens["speed"] * 3.6
    return sens


def _surface_of(r) -> str:
    for c in ("cobblestone_road", "dirt_road", "asphalt_road"):
        if r.get(c) == 1:
            return c.replace("_road", "")
    if r.get("unpaved_road") == 1:
        return "unpaved"
    if r.get("paved_road") == 1:
        return "paved"
    return "unknown"


def add_surface_relative_jolt(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each jolt against other readings on the SAME surface type.

    Absolute jolt is not evidence of damage: cobblestone shakes the car
    constantly by design, so a raw figure like 8 m/s^2 means "damaged" on
    asphalt and "normal" on setts. Feeding the raw number to the model made it
    accept more, not less. A percentile within the surface is comparable across
    surfaces.
    """
    df["surface"] = df.apply(_surface_of, axis=1)
    for col in ("jolt_z", "jolt_z_ahead_3s"):
        if col not in df:
            continue
        pct = df.groupby("surface")[col].rank(pct=True) * 100
        df[f"{col}_pct_for_surface"] = pct.round(1)
    return df


def attach_labels(sens: pd.DataFrame, sensor_csv: Path, labels_csv: Path) -> pd.DataFrame:
    """dataset_labels.csv carries no timestamp — it is row-index aligned."""
    raw_ts = pd.read_csv(sensor_csv, usecols=["timestamp"])
    labels = pd.read_csv(labels_csv)
    if len(raw_ts) != len(labels):
        raise ValueError(
            f"row count mismatch: sensors={len(raw_ts)} labels={len(labels)}. "
            "Row-index alignment is invalid — every label would be shifted."
        )
    lab = pd.concat([raw_ts, labels], axis=1).sort_values("timestamp")
    return pd.merge_asof(sens, lab, on="timestamp", direction="nearest",
                         tolerance=cfg.JOIN_TOLERANCE_S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE,
                    help='which PVS folder, e.g. "PVS 2"')
    ap.add_argument("--pvs", default=None, help="override the full folder path")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--fps", type=float, default=cfg.EXTRACT_FPS)
    ap.add_argument("--t0-offset", type=float, default=0.0)
    ap.add_argument("--stretch", action="store_true",
                    help="spread frames evenly across the sensor log instead of "
                         "trusting the container fps (see src.detect.sync_diagnose)")
    ap.add_argument("--skip-extract", action="store_true",
                    help="reuse frames already on disk, just redo the join")
    args = ap.parse_args()

    pvs = Path(args.pvs) if args.pvs else cfg.pvs_dir(args.trace)
    tag = cfg.slug(args.trace)
    frames_dir = cfg.FRAMES / tag
    print(f"trace: {args.trace}  ->  {pvs}")
    video = pvs / "video_environment.mp4"
    sensor_csv = pvs / f"dataset_gps_mpu_{args.side}.csv"
    labels_csv = pvs / "dataset_labels.csv"
    for p in (video, sensor_csv, labels_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    meta = probe(video)
    print(f"video: {meta['width']}x{meta['height']} @ {meta['fps']:.3f} fps, "
          f"{meta['n_frames']} frames, {meta['duration_s']:.2f} s")
    if args.stretch:
        ts = pd.read_csv(sensor_csv, usecols=["timestamp"])["timestamp"]
        print(f"sync: stretch mode — video ({meta['duration_s']:.1f} s) is mapped "
              f"linearly onto the sensor log ({float(ts.max() - ts.min()):.1f} s).")
        print("      The duration difference is expected and handled.")
    else:
        print("sync check:")
        verify_sync(meta, sensor_csv)

    sens = load_sensors(sensor_csv)
    sens = attach_labels(sens, sensor_csv, labels_csv)
    sens = add_surface_relative_jolt(sens)
    t0 = float(sens["timestamp"].min()) + args.t0_offset

    if not args.skip_extract:
        print(f"extracting at {args.fps} fps ...")
        extract(video, frames_dir, args.fps)

    span = float(sens["timestamp"].max() - sens["timestamp"].min()) if args.stretch else None
    idx = build_index(frames_dir, t0, args.fps, span_s=span)
    joined = pd.merge_asof(idx.sort_values("timestamp"), sens, on="timestamp",
                           direction="nearest", tolerance=cfg.JOIN_TOLERANCE_S)

    # Save the full surface-normalised jolt series. Time-to-impact needs to look
    # it up at t + tau, which is not a frame timestamp, so a per-frame column
    # cannot serve.
    jz = sens["jolt_z"].to_numpy(dtype="float32")
    jpct = sens.get("jolt_z_pct_for_surface")
    jpct = (jpct.to_numpy(dtype="float32") if jpct is not None
            else np.zeros(len(sens), dtype="float32"))
    np.savez_compressed(
        cfg.RAW / f"jolt_{tag}_{args.side}.npz",
        t=sens["timestamp"].to_numpy(dtype="float64"),
        raw=jz, pct=jpct,
        surface=sens["surface"].to_numpy().astype("U16"),
    )
    print(f"jolt series -> {cfg.RAW / f'jolt_{tag}_{args.side}.npz'}")

    out = cfg.RAW / f"frames_{tag}_{args.side}.csv"
    joined.to_csv(out, index=False)

    matched = int(joined["latitude"].notna().sum())
    print(f"{len(joined)} frames, {matched} matched -> {out}")
    if matched < len(joined) * 0.95:
        print("WARNING: <95% matched. t0 or JOIN_TOLERANCE_S is wrong.")


if __name__ == "__main__":
    main()
