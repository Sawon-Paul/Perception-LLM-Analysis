"""Decide the offset by eye. No correlation, no inference, no model.

For each candidate offset, pull the video frames that offset would pair with the
biggest measured jolts. If an offset is correct, those frames should show a
pothole, bump or rough patch in the road ahead. Wrong offsets show random smooth
road.

One row per offset, one column per jolt event, all in a single page.

    python -m src.detect.sync_sweep --trace "PVS 2"
    python -m src.detect.sync_sweep --trace "PVS 2" --offsets -60,-30,0,30,60
    python -m src.detect.sync_sweep --trace "PVS 2" --offsets 0 --fine
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

HEAD = """<!doctype html><meta charset="utf-8"><title>sync sweep - {trace}</title>
<style>
body{{font-family:system-ui;margin:20px;background:#101010;color:#eee}}
h1{{font-size:20px;margin-bottom:4px}}
.note{{color:#f2b544;font-size:13px;margin:10px 0 22px;max-width:900px;line-height:1.6}}
.row{{margin-bottom:26px}}
.lbl{{font-size:15px;font-weight:600;margin-bottom:8px;color:#8ab4f8}}
.strip{{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}}
.cell{{flex:0 0 200px}}
.cell img{{width:200px;height:113px;object-fit:cover;border-radius:4px;background:#000}}
.cap{{font-size:10px;color:#888;margin-top:3px}}
</style>
<h1>Sync sweep &mdash; {trace}</h1>
<p class="note">Each row is one candidate offset. The frames in that row are what the
pipeline would pair with the twelve largest measured jolts.<br><br>
<b>Look for the row where most frames show a defect in the road ahead</b> &mdash; a
pothole, a patch, a bump, a bad joint. That row's offset is the right one. If no row
looks better than the others, the video and sensors are not alignable this way.</p>
"""


def jolt_events(sensor_csv: Path, n: int, min_gap_s: float = 8.0) -> pd.DataFrame:
    df = pd.read_csv(sensor_csv,
                     usecols=["timestamp", "acc_z_below_suspension", "speed"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    z = df["acc_z_below_suspension"]
    df["jolt"] = (z - z.rolling(100, center=True, min_periods=1).mean()).abs()
    df = df[df["speed"] > 2.0]                       # ignore jolts while stopped

    chosen, used = [], []
    for _, r in df.nlargest(n * 20, "jolt").iterrows():
        ts = float(r["timestamp"])
        if all(abs(ts - u) > min_gap_s for u in used):
            used.append(ts)
            chosen.append(r)
        if len(chosen) >= n:
            break
    return pd.DataFrame(chosen)


def grab(video: Path, t: float, dst: Path) -> bool:
    if t < 0:
        return False
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                    "-frames:v", "1", "-q:v", "3", "-vf", "scale=400:-1", str(dst)],
                   check=False)
    return dst.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--offsets", default="-60,-30,-15,0,15,30,60",
                    help="comma separated seconds")
    ap.add_argument("--fine", action="store_true",
                    help="also sweep +/-10 s in 2.5 s steps around each offset")
    ap.add_argument("--n", type=int, default=12, help="jolt events per row")
    args = ap.parse_args()

    offsets = [float(x) for x in args.offsets.split(",")]
    if args.fine:
        fine = []
        for o in offsets:
            fine += [o + d for d in (-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10)]
        offsets = sorted(set(fine))

    pvs = cfg.pvs_dir(args.trace)
    video = pvs / "video_environment.mp4"
    sensor_csv = pvs / "dataset_gps_mpu_left.csv"

    ev = jolt_events(sensor_csv, args.n)
    t0 = float(pd.read_csv(sensor_csv, usecols=["timestamp"])["timestamp"].min())
    print(f"{len(ev)} jolt events, {len(offsets)} offsets "
          f"= {len(ev) * len(offsets)} frames to pull")

    out_dir = cfg.DATA / "sync_sweep" / cfg.slug(args.trace)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    rows = []
    for oi, off in enumerate(offsets):
        cells = []
        for ei, (_, r) in enumerate(ev.iterrows()):
            vt = float(r["timestamp"]) - t0 - off
            img = out_dir / f"o{oi:02d}_e{ei:02d}.jpg"
            if grab(video, vt, img):
                cells.append(
                    f'<div class="cell"><img src="file://{img.resolve()}">'
                    f'<div class="cap">t={vt:.0f}s &middot; jolt {r["jolt"]:.1f} '
                    f'&middot; {r["speed"] * 3.6:.0f} km/h</div></div>')
        rows.append(f'<div class="row"><div class="lbl">offset {off:+g} s</div>'
                    f'<div class="strip">{"".join(cells)}</div></div>')
        print(f"  offset {off:+g}s: {len(cells)} frames", flush=True)

    sheet = out_dir / "sweep.html"
    sheet.write_text(HEAD.replace("{trace}", args.trace) + "\n".join(rows),
                     encoding="utf-8")
    print(f"\n-> {sheet}")
    print("Open it and look for the row where the road actually shows damage.")


if __name__ == "__main__":
    main()
