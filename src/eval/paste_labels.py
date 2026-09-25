"""Fill labels.csv from a string of digits, so labels can be entered in batches.

    python -m src.eval.paste_labels --start 1 --digits 000000000000000000100000
    python -m src.eval.paste_labels --start 25 --digits 000000000000000000000000

Each digit is one card, in sheet order. Use x to leave a row blank, for anything
you want to come back to.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

EVAL = cfg.DATA / "eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE,
                    help="which trace's label file to write")
    ap.add_argument("--start", type=int, required=True, help="card number of the first digit")
    ap.add_argument("--digits", required=True, help="e.g. 0010000 — one per card, x to skip")
    ap.add_argument("--note", default="", help="note to write on these rows")
    args = ap.parse_args()

    path = EVAL / f"labels_{cfg.slug(args.trace)}.csv"
    if not path.exists():
        legacy = EVAL / "labels.csv"
        raise SystemExit(
            f"{path.name} not found.\n"
            f"Run: python -m src.eval.make_eval_set --trace \"{args.trace}\" --n 200\n"
            + (f"(An old shared {legacy.name} exists; it belongs to whichever trace "
               f"made it and is no longer used.)" if legacy.exists() else ""))
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"{path} is empty — run make_eval_set first")

    digits = [d for d in args.digits.strip() if not d.isspace()]
    by_n = {int(r["n"]): r for r in rows}

    written, skipped, missing = 0, 0, []
    for i, d in enumerate(digits):
        n = args.start + i
        if n not in by_n:
            missing.append(n)
            continue
        if d.lower() == "x":
            skipped += 1
            continue
        if d not in ("0", "1"):
            raise ValueError(f"card {n}: '{d}' is not 0, 1 or x")
        by_n[n]["is_real_fault"] = d
        if args.note:
            by_n[n]["notes"] = args.note
        written += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {written} labels for cards {args.start}-{args.start + len(digits) - 1}")
    if skipped:
        print(f"  left {skipped} blank (x)")
    if missing:
        print(f"  no such card: {missing[:10]}")

    filled = sum(1 for r in rows if r["is_real_fault"].strip() in ("0", "1"))
    pos = sum(1 for r in rows if r["is_real_fault"].strip() == "1")
    print(f"total: {filled}/{len(rows)} labelled, {pos} positive")


if __name__ == "__main__":
    main()
