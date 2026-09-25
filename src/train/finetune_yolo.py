"""Fine-tune the pothole detector so it stops firing on cobblestone.

The training set is mostly background frames, which is the point: the detector
needs to learn that block-paved and gravel roads are not damage. But a set that
is ~95% background can push a detector into predicting nothing at all, which
would look like a triumph on false positives and be worthless.

Three guards against that:

- a low learning rate, so the existing weights are nudged rather than rewritten
- few epochs, with early stopping on validation mAP
- the backbone frozen by default, so only the detection head adapts

The original weights are never overwritten. The fine-tuned model is written
beside them, and both are kept so the comparison can be run.

    python -m src.train.finetune_yolo
    python -m src.train.finetune_yolo --epochs 30 --no-freeze
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

DATA_YAML = cfg.DATA / "yolo_ft" / "data.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(cfg.POTHOLE_WEIGHTS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.0005,
                    help="deliberately low — this is a nudge, not a retrain")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-freeze", action="store_true",
                    help="train the whole network instead of just the head")
    ap.add_argument("--name", default="cobblestone_ft")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} missing — run src.train.build_negatives first")

    from ultralytics import YOLO

    src = Path(args.weights)
    print(f"starting from : {src}")
    print(f"data          : {DATA_YAML}")
    print(f"epochs {args.epochs}, lr {args.lr}, "
          f"{'head only (backbone frozen)' if not args.no_freeze else 'whole network'}")
    print("\nThe original weights are not modified.\n")

    model = YOLO(str(src))
    kwargs = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        lrf=0.1,
        patience=args.patience,
        device=cfg.DEVICE,
        project=str(cfg.DATA / "runs"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        warmup_epochs=1.0,
        # keep augmentation mild: the point is to learn this road surface, not
        # to invent variety the deployment will never see
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        hsv_v=0.3,
        val=True,
        plots=True,
    )
    if not args.no_freeze:
        kwargs["freeze"] = 10        # backbone layers

    results = model.train(**kwargs)

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training produced no weights at {best}")

    dst = cfg.WEIGHTS / "yolo11s_pothole_ft.pt"
    shutil.copy(best, dst)
    print(f"\nfine-tuned weights -> {dst}")
    print(f"original untouched -> {src}")

    print("\nCompare them properly. Point config.POTHOLE_WEIGHTS at the new file, "
          "rerun detection on the held-out trace, and score it:")
    print('  python -m src.detect.dump_detections --trace "PVS 2"')
    print('  python -m src.context.builder --trace "PVS 2"')
    print('  python -m src.llm.phase2_verify --trace "PVS 2" --model 7b')
    print('  python -m src.eval.metrics --trace "PVS 2"')
    print("\nWhat to look for: far fewer detections, and the survivors more often "
          "real. If detections drop to near zero, the model has learned to say "
          "nothing — lower the epochs or keep the backbone frozen.")


if __name__ == "__main__":
    main()
