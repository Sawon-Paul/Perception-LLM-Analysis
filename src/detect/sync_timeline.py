"""The decisive sync test: a timeline strip.

PVS 2 is 48% asphalt, 36% dirt, 17% cobblestone — long stretches, not scattered
patches. So the surface changes a handful of times across the drive, and each
change is obvious on camera.

This lays out frames sampled evenly across the whole video in one strip, then
draws a colour bar under them for each candidate offset showing what the labels
claim the surface was at that moment.

The correct offset is the one whose colour changes line up with the changes you
can see in the pictures.

    python -m src.detect.sync_timeline --trace "PVS 2"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

COLOR = {"asphalt": "#3b6fd4", "paved": "#5a86dd",
         "dirt": "#8a5a2b", "cobblestone": "#d4823b",
         "unpaved": "#a06a35", "unknown": "#444"}

HEAD = """<!doctype html><meta charset="utf-8"><title>sync timeline</title>
<style>
body{font-family:system-ui,sans-serif;margin:18px;background:#0e0e0e;color:#eee}
h1{font-size:20px;margin-bottom:6px}
.note{color:#f2b544;font-size:13px;margin:8px 0 20px;max-width:980px;line-height:1.6}
.wrap{overflow-x:auto;padding-bottom:12px}
table{border-collapse:collapse}
td,th{padding:0}
.f img{width:150px;height:84px;object-fit:cover;display:block;border-right:1px solid #0e0e0e}
.t{font-size:10px;color:#777;text-align:center;padding:3px 0 6px}
.bar{height:26px;border-right:1px solid #0e0e0e}
.rl{font-size:12px;color:#8ab4f8;white-space:nowrap;padding-right:10px;text-align:right;
    font-weight:600}
.key{margin:16px 0;font-size:12px}
.key span{display:inline-block;padding:3px 10px;border-radius:3px;margin-right:8px}
</style>
"""

INTRO = """<h1>Sync timeline &mdash; {trace}</h1>
<p class="note">The top row is the video, sampled evenly from start to finish. Each
coloured bar below is one candidate offset, showing what the sensor labels say the
surface was at that moment.<br><br>
<b>Look at the pictures and find where the road changes</b> &mdash; asphalt to dirt, dirt
to cobblestone. Then find the bar whose colour changes at the same columns. That bar's
offset is the correct one.</p>
<div class="key">
<span style="background:#3b6fd4">asphalt</span>
<span style="background:#8a5a2b">dirt</span>
<span style="background:#d4823b">cobblestone</span>
<span style="background:#444">unknown</span>
</div>
"""


def surface_series(trace: str) -> tuple[pd.DataFrame, float]:
    pvs = cfg.pvs_dir(trace)
    sens = pd.read_csv(pvs / "dataset_gps_mpu_left.csv", usecols=["timestamp"])
    labels = pd.read_csv(pvs / "dataset_labels.csv")
    if len(sens) != len(labels):
        raise ValueError(f"row mismatch: sensors={len(sens)} labels={len(labels)}")
    df = pd.concat([sens, labels], axis=1).sort_values("timestamp").reset_index(drop=True)

    def surface(r) -> str:
        for c in ("cobblestone_road", "dirt_road", "asphalt_road"):
            if r.get(c) == 1:
                return c.replace("_road", "")
        if r.get("unpaved_road") == 1:
            return "unpaved"
        if r.get("paved_road") == 1:
            return "paved"
        return "unknown"

    df["surface"] = df.apply(surface, axis=1)
    return df[["timestamp", "surface"]], float(df["timestamp"].min())


def video_duration(video: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)])
    return float(out.decode().strip())


def grab(video: Path, t: float, dst: Path) -> bool:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(t, 0):.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", "scale=300:-1",
         "-pix_fmt", "yuvj420p",          # avoids the non-full-range mjpeg error
         "-q:v", "4", str(dst)], check=False)
    return dst.exists() and dst.stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--offsets", default="-90,-60,-45,-30,-15,0,15,30,45,60,90")
    ap.add_argument("--n", type=int, default=40, help="frames sampled across the video")
    args = ap.parse_args()

    offsets = [float(x) for x in args.offsets.split(",")]
    pvs = cfg.pvs_dir(args.trace)
    video = pvs / "video_environment.mp4"

    lab, t0 = surface_series(args.trace)
    span = float(lab["timestamp"].max() - lab["timestamp"].min())
    vdur = video_duration(video)
    print(f"video {vdur:.1f}s, sensors {span:.1f}s")
    print(f"surface counts: {lab['surface'].value_counts().to_dict()}")

    out_dir = cfg.DATA / "sync_timeline" / cfg.slug(args.trace)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    ts_lab = lab["timestamp"].to_numpy()
    surf = lab["surface"].to_numpy()

    cols = []
    for i in range(args.n):
        vt = vdur * i / (args.n - 1) * 0.995
        img = out_dir / f"t{i:03d}.jpg"
        if grab(video, vt, img):
            cols.append((vt, img))
        print(f"  {i + 1}/{args.n}", end="\r", flush=True)
    print(f"\n{len(cols)} frames captured")

    img_row = "".join(
        f'<td class="f"><img src="{p.resolve().as_uri()}">'
        f'<div class="t">{vt:.0f}s</div></td>' for vt, p in cols)

    bars = []
    for off in offsets:
        cells = []
        for vt, _ in cols:
            want = t0 + off + vt
            idx = int(ts_lab.searchsorted(want))
            s = surf[min(idx, len(surf) - 1)] if 0 <= idx else "unknown"
            if want < ts_lab[0] or want > ts_lab[-1]:
                s = "unknown"
            cells.append(f'<td class="bar" style="background:{COLOR.get(s, "#444")}"></td>')
        bars.append(f'<tr><td class="rl">{off:+g} s</td>{"".join(cells)}</tr>')

    html = (HEAD + INTRO.replace("{trace}", args.trace)
            + '<div class="wrap"><table>'
            + f'<tr><td class="rl">video</td>{img_row}</tr>'
            + "".join(bars) + "</table></div>")

    sheet = out_dir / "timeline.html"
    sheet.write_text(html, encoding="utf-8")
    print(f"\n-> {sheet}")
    print("\nOpen it. Find where the road visibly changes surface in the photos,")
    print("then find the bar whose colour changes at the same place.")


if __name__ == "__main__":
    main()
