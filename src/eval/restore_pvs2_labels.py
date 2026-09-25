"""Rebuild the PVS 2 label file after it was overwritten.

The old shared labels.csv had no trace in its name, so labelling a second trace
pasted over the first. The detection_id column survived — only the verdicts were
lost — so the file can be rebuilt from the recorded positives.

PVS 2, 148 cards: everything is a false positive except cards 19, 72, 96, 107
and 113, which were judged real damage.

    python -m src.eval.restore_pvs2_labels          # check first
    python -m src.eval.restore_pvs2_labels --write  # then write
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

EVAL = cfg.DATA / "eval"
POSITIVES = {19, 72, 96, 107, 113}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="actually write the file")
    args = ap.parse_args()

    src = EVAL / "labels.csv"
    dst = EVAL / "labels_PVS2.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found — nothing to restore from")

    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    ids = [r.get("detection_id", "") for r in rows]
    n_pvs2 = sum(1 for i in ids if "PVS2" in i)

    print(f"{src.name}: {len(rows)} rows, {n_pvs2} with a PVS2 detection_id")
    if n_pvs2 < len(rows) * 0.9:
        raise SystemExit(
            "These rows are not PVS 2 detections, so this is the wrong file to "
            "restore from. Stop and check before writing anything.")
    if len(rows) != 148:
        print(f"WARNING: expected 148 rows, found {len(rows)}. The card numbering "
              f"may not match the recorded positives.")

    for r in rows:
        n = int(r["n"])
        r["is_real_fault"] = "1" if n in POSITIVES else "0"
        r["notes"] = "restored" if n in POSITIVES else ""

    pos = [int(r["n"]) for r in rows if r["is_real_fault"] == "1"]
    print(f"restored: {len(rows)} labelled, {len(pos)} positive at cards {sorted(pos)}")

    if not args.write:
        print(f"\nDry run. Rerun with --write to save to {dst.name}")
        return

    if src.exists():
        shutil.copy(src, EVAL / "labels.csv.bak")
        print(f"backed up the old shared file to labels.csv.bak")
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {dst}")
    print("\nCheck it:  python -m src.eval.metrics --trace \"PVS 2\"")
    print("Expect 148 labelled, 5 positive, and the 7B run at 2 false positives.")


if __name__ == "__main__":
    main()
