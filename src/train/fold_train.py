"""Train one detector per spatial fold.

Each fold's detector sees only training-region detections and is never shown a
frame that contains a test detection. That last rule matters: one video frame
can hold several detections, and if one sits in a test block while another sits
in a training block, keeping the frame would show the detector the test road.
Any such frame is dropped from training outright.

Settings match the earlier fine-tune that took detections from 366 to 27:
backbone frozen, low learning rate, mild augmentation, background frames capped
at 12 per positive frame.

    python -m src.train.fold_train --build-only     # check the datasets first
    python -m src.train.fold_train                  # build and train all folds
    python -m src.train.fold_train --fold 1         # one fold
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

OUT = cfg.DATA / "fold_ft"

def base_weights(path: str | None) -> Path:
    """The model every fold starts from, and the baseline it is compared with.

    This used to be cfg.POTHOLE_WEIGHTS, which had been pointed at the
    fine-tuned model for an unrelated comparison. That model trained on all
    eight other traces \u2014 including the road inside every fold's test region
    \u2014 so folds started from it had already seen their test road, and the
    "original" baseline was not the original. The base is now named explicitly
    and a fine-tuned or fold model is refused.
    """
    p = Path(path) if path else cfg.WEIGHTS / "yolo11s_pothole.pt"
    bad = ("_ft", "fold")
    if any(b in p.stem for b in bad):
        raise SystemExit(
            f"refusing base weights {p.name}: it was fine-tuned on traces that "
            f"overlap the test folds, so it has seen the held-out road.\n"
            f"Use the original detector, weights/yolo11s_pothole.pt.")
    if not p.exists():
        raise SystemExit(f"{p} not found")
    return p



def load_events() -> dict[str, dict]:
    """Every pothole detection across every trace, keyed by detection id."""
    out = {}
    traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                    if d.is_dir() and d.name.upper().startswith("PVS"))
    for t in traces:
        tag = cfg.slug(t)
        for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
            p = cfg.EVENTS / f"events_{tag}_left{suffix}.jsonl"
            if p.exists():
                for line in p.open(encoding="utf-8"):
                    if line.strip():
                        e = json.loads(line)
                        if e.get("stream5_model", {}).get("stream") == "pothole":
                            out[e["detection_id"]] = e
                break
    return out


def load_labels() -> dict[str, int]:
    out = {}
    for p in sorted((cfg.DATA / "eval").glob("labels_*.csv")):
        for r in csv.DictReader(p.open(encoding="utf-8")):
            v = r.get("is_real_fault", "").strip()
            if v in ("0", "1"):
                out[r["detection_id"]] = int(v)
    return out


def yolo_line(bbox, w, h, cid) -> str | None:
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
    if bw <= 0 or bh <= 0:
        return None
    return f"{cid} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} {bw:.6f} {bh:.6f}"


def build_fold(fold: dict, events: dict, labels: dict, max_bg: float,
               val_frac: float, seed: int) -> dict:
    fi = fold["fold"]
    root = OUT / f"fold{fi}"
    if root.exists():
        shutil.rmtree(root)

    test_ids = set(fold["test_ids"])
    train_ids = set(fold["train_ids"])

    # frames touched by any test detection are forbidden in training
    test_frames = {events[i]["stream1_image"]["frame_path"]
                   for i in test_ids if i in events
                   and events[i].get("stream1_image", {}).get("frame_path")}

    by_frame: dict[str, list[dict]] = defaultdict(list)
    unlabelled = 0
    for i in train_ids:
        e = events.get(i)
        if not e:
            continue
        fp = e.get("stream1_image", {}).get("frame_path")
        if not fp or fp in test_frames:
            continue
        if i not in labels:
            unlabelled += 1
            continue
        by_frame[fp].append(e)

    dropped_shared = sum(1 for i in train_ids
                         if i in events
                         and events[i].get("stream1_image", {}).get("frame_path")
                         in test_frames)

    pos, bg = [], []
    for fp, evs in by_frame.items():
        (pos if any(labels[e["detection_id"]] == 1 for e in evs) else bg).append(fp)

    rng = random.Random(seed + fi)
    rng.shuffle(bg)
    if pos and len(bg) > len(pos) * max_bg:
        bg = bg[:int(len(pos) * max_bg)]

    # Stratified: positives and background are split separately. A random split
    # drew an all-background validation set, so validation mAP was zero every
    # epoch and early stopping fired on noise. At a few percent positives that
    # is likely, not rare.
    rng.shuffle(pos)
    n_val_pos = max(1, round(len(pos) * val_frac)) if len(pos) >= 3 else 0
    n_val_bg = round(len(bg) * val_frac)
    val_set = set(pos[:n_val_pos]) | set(bg[:n_val_bg])
    frames = pos + bg

    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}   # frames, boxes, background
    for fp in frames:
        split = "val" if fp in val_set else "train"
        src = Path(fp)
        if not src.exists():
            continue
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        stem = f"{src.parent.name}_{src.stem}"
        shutil.copy(src, root / "images" / split / f"{stem}.jpg")

        lines = []
        for e in by_frame[fp]:
            if labels[e["detection_id"]] != 1:
                continue
            ln = yolo_line(e["stream1_image"]["bbox_xyxy"],
                           e.get("img_w") or 1280, e.get("img_h") or 720,
                           e["stream5_model"].get("class_id", 0))
            if ln:
                lines.append(ln)
        (root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        counts[split][0] += 1
        counts[split][1] += len(lines)
        counts[split][2] += 0 if lines else 1

    from ultralytics import YOLO
    names = YOLO(str(base_weights(None))).names
    yaml = root / "data.yaml"
    lines = [f"path: {root.as_posix()}", "train: images/train", "val: images/val",
             f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {names[i]}" for i in sorted(names)]
    yaml.write_text("\n".join(lines), encoding="utf-8")

    info = {"fold": fi, "yaml": str(yaml),
            "train_frames": counts["train"][0], "train_boxes": counts["train"][1],
            "train_background": counts["train"][2],
            "val_frames": counts["val"][0], "val_boxes": counts["val"][1],
            "dropped_shared_frames": dropped_shared, "unlabelled": unlabelled}
    print(f"fold {fi}")
    print(f"  train {counts['train'][0]:5d} frames, {counts['train'][1]:4d} boxes, "
          f"{counts['train'][2]:5d} background")
    print(f"  val   {counts['val'][0]:5d} frames, {counts['val'][1]:4d} boxes")
    print(f"  dropped: {dropped_shared} training detections sharing a frame with "
          f"test, {unlabelled} unlabelled")
    if counts["val"][1] == 0:
        print("  WARNING: validation holds no positive boxes. Validation mAP will be "
              "0 every epoch and early stopping will not mean anything.")
    if counts["train"][1] < 30:
        print(f"  WARNING: only {counts['train'][1]} positive boxes. This fold will "
              f"mostly learn to stay quiet; expect recall to fall.")
    return info


def add_extra(info: dict, extra: Path, max_extra: int | None, seed: int) -> int:
    """Add an external dataset to this fold's TRAINING split only.

    Validation stays PVS: early stopping has to judge the model on the road it
    will actually drive, not on the external dataset's road. Mixing external
    images into validation would let a model that improves on Japanese asphalt
    and worsens on Brazilian cobblestone look like it is improving.
    """
    root = Path(info["yaml"]).parent
    imgs = sorted(p for p in (extra / "images").iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    random.Random(seed).shuffle(imgs)
    if max_extra:
        imgs = imgs[:max_extra]
    n = 0
    for img in imgs:
        lab = extra / "labels" / f"{img.stem}.txt"
        if not lab.exists():
            continue
        shutil.copy(img, root / "images" / "train" / img.name)
        shutil.copy(lab, root / "labels" / "train" / f"{img.stem}.txt")
        n += 1
    return n


def train_fold(info: dict, epochs: int, lr: float, base: Path) -> Path:
    from ultralytics import YOLO
    fi = info["fold"]
    model = YOLO(str(base))
    res = model.train(
        data=info["yaml"], epochs=epochs, imgsz=640, batch=16, lr0=lr, lrf=0.1,
        freeze=10, optimizer="AdamW", warmup_epochs=1.0, patience=6,
        mosaic=0.0, mixup=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        fliplr=0.5, hsv_v=0.3, device=cfg.DEVICE,
        project=str(cfg.DATA / "runs"), name=f"fold{fi}", exist_ok=True,
        val=True, plots=True, verbose=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    dst = cfg.WEIGHTS / f"yolo11s_pothole_fold{fi}{info.get('tag', '')}.pt"
    shutil.copy(best, dst)
    print(f"  -> {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None, help="one fold only")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--max-bg-ratio", type=float, default=12.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default=None,
                    help="external YOLO dataset added to training only, e.g. data/rdd_yolo")
    ap.add_argument("--max-extra", type=int, default=None,
                    help="cap on external images per fold")
    ap.add_argument("--base", default=None,
                    help="starting weights; defaults to the original detector")
    ap.add_argument("--tag", default="",
                    help="suffix on saved weights, e.g. _rdd, so runs do not overwrite")
    args = ap.parse_args()

    folds_path = cfg.DATA / "spatial_folds.json"
    spec = json.loads(folds_path.read_text())
    folds = spec["folds"]
    if "train_ids" not in folds[0]:
        raise SystemExit("spatial_folds.json has no detection ids \u2014 rerun "
                         "src.analysis.spatial_folds with the updated script")

    base = base_weights(args.base)
    print(f"base model: {base.name}")
    events = load_events()
    labels = load_labels()
    print(f"{len(events)} detections, {len(labels)} labelled\n")

    chosen = [f for f in folds if args.fold is None or f["fold"] == args.fold]
    infos = [build_fold(f, events, labels, args.max_bg_ratio, args.val_frac,
                        args.seed) for f in chosen]
    if args.extra:
        extra = Path(args.extra)
        if not (extra / "images").exists():
            raise SystemExit(f"{extra} has no images/ folder \u2014 run rdd_convert first")
        for info in infos:
            n = add_extra(info, extra, args.max_extra, args.seed)
            info["extra_images"] = n
            print(f"fold {info['fold']}: +{n} external images in training "
                  f"(validation unchanged, PVS only)")
    (OUT / "folds_built.json").write_text(json.dumps(infos, indent=2))

    if args.build_only:
        print("\nbuilt only. Review the counts, then run without --build-only.")
        return

    for info in infos:
        info["tag"] = args.tag
        print(f"\ntraining fold {info['fold']} ...")
        train_fold(info, args.epochs, args.lr, base)
    print("\nall folds trained. Next: run each fold's detector on its held-out "
          "region and measure consensus there.")


if __name__ == "__main__":
    main()
