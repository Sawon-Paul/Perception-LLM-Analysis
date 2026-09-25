"""Work out which parts of the frame are not road.

Phase 2 reported crops showing "a part of a car's roof with a small scratch",
so the detector is firing on the vehicle's own bonnet. The bonnet occupies the
bottom of every dashcam frame and never contains a road defect, so detections
there are guaranteed false positives.

This renders a sample frame with candidate cut lines drawn on it, plus a
histogram of where detections actually sit vertically, as one JPG.

    python -m src.detect.roi_check --trace "PVS 2"
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402

CANDIDATES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
LINE_RGB = [(255, 90, 90), (255, 150, 60), (255, 220, 60),
            (140, 230, 120), (100, 190, 255), (200, 130, 255)]


def frame_at(video: Path, t: float, width: int) -> Image.Image | None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:-1", "-pix_fmt", "yuvj420p",
         "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    args = ap.parse_args()

    tag = cfg.slug(args.trace)
    dets = load_jsonl(cfg.DETECTIONS / f"dets_{tag}_{args.side}.jsonl")
    print(f"{len(dets)} detections")

    # vertical position of each box, as a fraction of frame height
    ys, areas = [], []
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        ys.append(((y1 + y2) / 2) / d["img_h"])
        areas.append(d.get("bbox_area_px", 0) / (d["img_w"] * d["img_h"]))
    ys = np.array(ys)
    areas = np.array(areas)

    print("\ndetections by vertical band:")
    for lo in np.arange(0.0, 1.0, 0.1):
        n = int(((ys >= lo) & (ys < lo + 0.1)).sum())
        bar = "#" * int(60 * n / max(len(ys), 1))
        print(f"  {lo:.1f}-{lo + 0.1:.1f}  {n:4d}  {bar}")

    print("\nif detections below a line were dropped:")
    for c in CANDIDATES:
        kept = int((ys < c).sum())
        print(f"  cut at {c:.2f}  keeps {kept}/{len(ys)} "
              f"({100 * kept / len(ys):.0f}%), drops {len(ys) - kept}")

    print(f"\nbox area as fraction of frame: median {np.median(areas):.5f}, "
          f"90th pct {np.percentile(areas, 90):.5f}")
    tiny = int((areas < 0.005).sum())
    print(f"  boxes under 0.5% of the frame: {tiny}/{len(areas)} "
          f"({100 * tiny / len(areas):.0f}%) — these crop to blurry thumbnails")

    # render a few frames with the candidate lines drawn on
    video = cfg.pvs_dir(args.trace) / "video_environment.mp4"
    W = 640
    picks = [0.15, 0.4, 0.65, 0.9]
    from src.detect.sync_image import duration
    vdur = duration(video)
    frames = [f for f in (frame_at(video, vdur * p, W) for p in picks) if f]
    if not frames:
        print("could not read the video")
        return

    h = frames[0].size[1]
    out = Image.new("RGB", (W * len(frames), h + 30), (14, 14, 14))
    d = ImageDraw.Draw(out)
    for i, fr in enumerate(frames):
        out.paste(fr, (i * W, 30))
        for c, col in zip(CANDIDATES, LINE_RGB):
            y = 30 + int(h * c)
            d.line([(i * W, y), ((i + 1) * W - 1, y)], fill=col, width=2)
            d.text((i * W + 4, y - 12), f"{c:.2f}", fill=col)
    d.text((8, 8), "Which line sits just above the car bonnet? "
                   "Everything below it is never road.", fill=(240, 240, 240))

    dst = cfg.DATA / f"roi_{tag}.jpg"
    out.save(dst, quality=88)
    print(f"\n-> {dst}")
    print("Open it, pick the line that sits just above the bonnet, then rerun "
          "detection with e.g.  --roi-bottom 0.75")


if __name__ == "__main__":
    main()
