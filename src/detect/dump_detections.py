"""Step 2: run both yolo11s models over the paired frames, write detections JSONL.

Reads data/raw/frames_<side>.csv (from step 1) so every detection inherits
lat, lon, timestamp, jolt and the PVS labels. Run once, then leave YOLO alone.

    python -m src.detect.dump_detections --side left
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.io import write_jsonl  # noqa: E402

CARRY = [
    "timestamp", "latitude", "longitude", "speed_kmh",
    "jolt_z", "jolt_z_std_1s", "jolt_z_max_1s",
    "jolt_z_ahead_2s", "jolt_z_ahead_3s", "jolt_z_ahead_5s",
    "surface", "jolt_z_pct_for_surface", "jolt_z_ahead_3s_pct_for_surface",
    "speed",
    "acc_z_below_suspension",
    "paved_road", "unpaved_road", "dirt_road", "cobblestone_road", "asphalt_road",
    "no_speed_bump", "speed_bump_asphalt", "speed_bump_cobblestone",
    "good_road_left", "regular_road_left", "bad_road_left",
    "good_road_right", "regular_road_right", "bad_road_right",
]


def run_model(weights: Path, stream: str, frames: pd.DataFrame,
              records: list[dict], batch: int = 16,
              roi_top: float = 0.0, roi_bottom: float = 1.0,
              min_area_frac: float = 0.0) -> None:
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing — put your trained .pt in weights/")

    model = YOLO(str(weights))
    ctx_by_path = frames.set_index("frame_path").to_dict("index")
    paths = frames["frame_path"].tolist()

    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        results = model.predict(source=chunk, conf=cfg.CONF_THRESHOLD,
                                iou=cfg.IOU_THRESHOLD, device=cfg.DEVICE,
                                verbose=False)
        for res in results:
            fpath = str(res.path)
            ctx = ctx_by_path.get(fpath, {})
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            h, w = res.orig_shape
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]

                # The bonnet fills the bottom of every dashcam frame and never
                # contains a road defect; the sky fills the top. Detections
                # there are false positives by construction.
                y_mid = ((y1 + y2) / 2) / h
                if not (roi_top <= y_mid <= roi_bottom):
                    continue
                if min_area_frac > 0:
                    if (x2 - x1) * (y2 - y1) / (w * h) < min_area_frac:
                        continue

                cid = int(boxes.cls[i].item())
                rec = {
                    "detection_id": f"{stream}_{Path(fpath).parent.name}_{Path(fpath).stem}_{i}",
                    "stream": stream,
                    "model_weights": weights.name,
                    "frame_path": fpath,
                    "class_id": cid,
                    "class_name": res.names.get(cid, str(cid)),
                    "conf": round(float(boxes.conf[i].item()), 4),
                    "bbox_xyxy": [round(v, 2) for v in (x1, y1, x2, y2)],
                    "bbox_area_px": round((x2 - x1) * (y2 - y1), 2),
                    "y_bottom": round(y2, 2),      # for time-to-impact geometry
                    "img_w": int(w), "img_h": int(h),
                }
                for c in CARRY:
                    if c in ctx:
                        v = ctx[c]
                        rec[c] = None if pd.isna(v) else v
                records.append(rec)
        print(f"  {stream}: {min(start + batch, len(paths))}/{len(paths)} frames", end="\r")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--limit", type=int, default=None, help="first N frames only (smoke test)")
    ap.add_argument("--roi-top", type=float, default=0.0,
                    help="ignore detections whose centre is above this fraction of the frame")
    ap.add_argument("--roi-bottom", type=float, default=1.0,
                    help="ignore detections below this fraction — set just above the bonnet")
    ap.add_argument("--min-area-frac", type=float, default=0.0,
                    help="drop boxes smaller than this fraction of the frame")
    args = ap.parse_args()

    tag = cfg.slug(args.trace)
    frames_csv = cfg.RAW / f"frames_{tag}_{args.side}.csv"
    frames = pd.read_csv(frames_csv)
    frames = frames[frames["latitude"].notna()]
    if args.limit:
        frames = frames.head(args.limit)
    print(f"{len(frames)} frames with GPS")

    if args.roi_bottom < 1.0 or args.roi_top > 0.0:
        print(f"ROI: keeping detections between {args.roi_top:.2f} and "
              f"{args.roi_bottom:.2f} of frame height")
    if args.min_area_frac > 0:
        print(f"dropping boxes under {args.min_area_frac:.4f} of the frame")

    records: list[dict] = []
    kw = dict(roi_top=args.roi_top, roi_bottom=args.roi_bottom,
              min_area_frac=args.min_area_frac)
    run_model(cfg.POTHOLE_WEIGHTS, "pothole", frames, records, **kw)
    run_model(cfg.SIGN_WEIGHTS, "sign", frames, records, **kw)

    out = cfg.DETECTIONS / f"dets_{tag}_{args.side}.jsonl"
    write_jsonl(out, records)
    n_pot = sum(1 for r in records if r["stream"] == "pothole")
    print(f"wrote {len(records)} detections -> {out} (pothole={n_pot}, sign={len(records) - n_pot})")


if __name__ == "__main__":
    main()
