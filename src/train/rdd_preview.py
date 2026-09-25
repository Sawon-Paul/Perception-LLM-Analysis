"""Show what each class id actually looks like, so the map is decided by eye.

The dataset ships without class names. Counting boxes suggests an order but
cannot separate three classes of similar size, so this draws real examples of
each id onto one image. Upload it and the classes can be named from what is
visible: a hole, a crack running along the road, a crack across it, a web of
cracks, a patched repair.

    python -m src.train.rdd_preview --root "C:/Sawon/Data/RDD_SPLIT"
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png")
DASHCAM = ("Japan", "India", "United_States", "Czech")


def prefix_of(stem: str) -> str:
    m = re.match(r"([A-Za-z]+(?:_[A-Za-z]+)*?)_\d", stem)
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    idir, ldir = root / "train" / "images", root / "train" / "labels"
    by_class: dict[int, list[tuple[Path, list[float]]]] = defaultdict(list)

    for lp in sorted(ldir.glob("*.txt")):
        if not prefix_of(lp.stem).startswith(DASHCAM):
            continue
        img = next((idir / f"{lp.stem}{e}" for e in IMG_EXT
                    if (idir / f"{lp.stem}{e}").exists()), None)
        if img is None:
            continue
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            try:
                cid = int(float(p[0]))
                box = [float(x) for x in p[1:]]
            except ValueError:
                continue
            # skip slivers: a tiny box shows nothing to judge by
            if box[2] * box[3] > 0.004:
                by_class[cid].append((img, box))

    rng = random.Random(args.seed)
    cell, ctx = 260, 1.8
    ids = sorted(by_class)
    sheet = Image.new("RGB", (120 + cell * args.per_class, 40 + cell * len(ids)),
                      (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), "Each row is one class id. Red box = the labelled region.",
           fill=(230, 230, 230))

    for row, cid in enumerate(ids):
        samples = by_class[cid]
        rng.shuffle(samples)
        y0 = 40 + row * cell
        d.text((12, y0 + cell // 2 - 16), f"id {cid}", fill=(255, 210, 90))
        d.text((12, y0 + cell // 2 + 2), f"n={len(by_class[cid])}",
               fill=(160, 160, 160))
        for col, (img, (cx, cy, bw, bh)) in enumerate(samples[:args.per_class]):
            with Image.open(img) as im:
                im = im.convert("RGB")
                W, H = im.size
                x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                half = max(x2 - x1, y2 - y1) * ctx / 2
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                box = (int(max(0, mx - half)), int(max(0, my - half)),
                       int(min(W, mx + half)), int(min(H, my + half)))
                crop = im.crop(box)
                cd = ImageDraw.Draw(crop)
                cd.rectangle([x1 - box[0], y1 - box[1], x2 - box[0], y2 - box[1]],
                             outline=(255, 40, 40), width=3)
                crop.thumbnail((cell - 8, cell - 8))
            sheet.paste(crop, (120 + col * cell + 4, y0 + 4))

    out = cfg.DATA / "rdd_class_preview.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    print(f"{len(ids)} classes, {args.per_class} examples each -> {out}")
    print("Upload that image.")


if __name__ == "__main__":
    main()
