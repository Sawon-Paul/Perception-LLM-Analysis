"""Turn the labelled crops into a YOLO training set dominated by hard negatives.

The detector was trained on asphalt distress and fires constantly on cobblestone,
whose joints look like alligator cracking. Roughly 97% of its detections here are
wrong. The labelled sets already say which are which, so those judgements can be
fed back as training signal.

How a frame becomes a training sample:

- a detection judged REAL keeps its box, with its original class id
- a detection judged FALSE has its box dropped
- a frame whose detections were all false becomes a background image: an empty
  label file, which is how YOLO is taught not to fire

Train and validate on different traces with different label sources, so the
result is not measured on what it learned from. Default: train on PVS 1
(auto-labelled), validate on PVS 2 (human-labelled).

    python -m src.train.build_negatives --train "PVS 1" --val "PVS 2"
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402

EVAL = cfg.DATA / "eval"
OUT = cfg.DATA / "yolo_ft"


def load_split_labels(trace: str) -> dict[str, int]:
    path = EVAL / f"labels_{cfg.slug(trace)}.csv"
    if not path.exists():
        raise SystemExit(f"{path.name} not found — label {trace} first")
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        v = r["is_real_fault"].strip()
        if v in ("0", "1"):
            out[r["detection_id"]] = int(v)
    return out


def to_yolo_box(x1, y1, x2, y2, w, h) -> tuple[float, float, float, float]:
    """xyxy in pixels -> normalised cx, cy, bw, bh, clamped to the image."""
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    return (((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
            abs(x2 - x1) / w, abs(y2 - y1) / h)


def build_split(traces: list[str], side: str, split: str,
                max_bg_ratio: float | None = None, seed: int = 0) -> dict:
    """Several traces can feed one split. More traces means more surface
    variety, which is the point: cobblestone under one trace's lighting is not
    the same problem as cobblestone under another's."""
    by_frame: dict[str, list[dict]] = {}
    labels: dict[str, int] = {}
    for trace in traces:
        tl = load_split_labels(trace)
        labels.update(tl)
        path = cfg.EVENTS / f"events_{cfg.slug(trace)}_{side}.jsonl"
        if not path.exists():
            print(f"    skipping {trace}: {path.name} not found")
            continue
        evs = [e for e in load_jsonl(path)
               if e["stream5_model"].get("stream") == "pothole"]
        for e in evs:
            fp = e["stream1_image"].get("frame_path")
            if fp and e["detection_id"] in tl:
                by_frame.setdefault(fp, []).append(e)

    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()

    # Split frames into those keeping a box and those that become background,
    # so the background can be capped before anything is written.
    pos_frames, bg_frames = [], []
    for fp, evs in by_frame.items():
        (pos_frames if any(labels.get(e["detection_id"]) == 1 for e in evs)
         else bg_frames).append(fp)

    if max_bg_ratio is not None and pos_frames:
        cap = int(len(pos_frames) * max_bg_ratio)
        if len(bg_frames) > cap:
            random.Random(seed).shuffle(bg_frames)
            print(f"    capping background {len(bg_frames)} -> {cap} "
                  f"({max_bg_ratio:g} per positive)")
            bg_frames = bg_frames[:cap]

    keep = set(pos_frames) | set(bg_frames)
    n_bg, n_pos, n_boxes = 0, 0, 0
    class_counts: Counter = Counter()

    for fp, evs in by_frame.items():
        if fp not in keep:
            continue
        src = Path(fp)
        if not src.exists():
            continue
        stem = f"{src.parent.name}_{src.stem}"
        shutil.copy(src, img_dir / f"{stem}.jpg")

        lines = []
        for e in evs:
            if labels[e["detection_id"]] != 1:
                continue                       # rejected: its box is dropped
            x1, y1, x2, y2 = e["stream1_image"]["bbox_xyxy"]
            iw = e.get("img_w") or 1280
            ih = e.get("img_h") or 720
            cx, cy, bw, bh = to_yolo_box(x1, y1, x2, y2, iw, ih)
            if bw <= 0 or bh <= 0:
                continue
            cid = e["stream5_model"].get("class_id", 0)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[e["stream5_model"].get("class_name", cid)] += 1

        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if lines:
            n_pos += 1
            n_boxes += len(lines)
        else:
            n_bg += 1

    print(f"\n  {split} ({', '.join(traces)})")
    print(f"    frames with a kept box : {n_pos}  ({n_boxes} boxes)")
    print(f"    background frames      : {n_bg}")
    print(f"    ratio                  : {n_bg / max(n_pos, 1):.1f} background per positive")
    if class_counts:
        print(f"    classes kept           : {dict(class_counts)}")
    return {"pos": n_pos, "bg": n_bg, "boxes": n_boxes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="PVS 1",
                    help='comma separated, e.g. "PVS 1,PVS 3,PVS 4"')
    ap.add_argument("--val", default="PVS 2", help="held out, never trained on")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--max-bg-ratio", type=float, default=None,
                    help="cap background frames per positive frame, e.g. 12")
    args = ap.parse_args()

    train_traces = [s.strip() for s in args.train.split(",") if s.strip()]
    val_traces = [s.strip() for s in args.val.split(",") if s.strip()]

    overlap = {cfg.slug(t) for t in train_traces} & {cfg.slug(t) for t in val_traces}
    if overlap:
        raise SystemExit(
            f"{sorted(overlap)} appears in both train and val. The validation "
            f"trace is the only human-labelled data — training on it would make "
            f"the result meaningless.")

    # class names come from the existing model so ids stay compatible
    from ultralytics import YOLO
    names = YOLO(str(cfg.POTHOLE_WEIGHTS)).names
    print(f"classes from {cfg.POTHOLE_WEIGHTS.name}: {names}")

    tr = build_split(train_traces, args.side, "train", args.max_bg_ratio)
    va = build_split(val_traces, args.side, "val")

    yaml = OUT / "data.yaml"
    lines = [f"path: {OUT.as_posix()}", "train: images/train", "val: images/val",
             f"nc: {len(names)}", "names:"]
    for i in sorted(names):
        lines.append(f"  {i}: {names[i]}")
    yaml.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n-> {yaml}")
    if tr["boxes"] < 30:
        print(f"\nWARNING: only {tr['boxes']} positive boxes to train on. Fine-tuning "
              f"will mostly teach the model to say no. Expect false positives to fall "
              f"and recall to fall with them — check the validation numbers before "
              f"adopting the result.")
    print("\nNext:  python -m src.train.finetune_yolo")


if __name__ == "__main__":
    main()
