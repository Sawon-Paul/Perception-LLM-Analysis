"""Pool the per-fold federated learning results into one table.

A single fold cannot separate the arms: the held-out region holds a few dozen
positive boxes, so an F1 gap of a few hundredths is noise. Three folds give a
spread, and a difference smaller than that spread is not a difference.

Arms are compared at the same round. Taking each arm's best round would mean
choosing the round by looking at the held-out set, which is selection on test
data.

    python -m src.fl.summarise
    python -m src.fl.summarise --round 3
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

RESULTS = cfg.ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None,
                    help="which round to compare at; default is the last common one")
    args = ap.parse_args()

    files = sorted(RESULTS.glob("fl_fold*.csv"))
    if not files:
        raise SystemExit(f"no fl_fold*.csv in {RESULTS} \u2014 run src.fl.federated first")

    per_fold: dict[str, list[dict]] = {}
    for f in files:
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        for r in rows:
            r["f1"] = float(r["f1"])
            r["precision"] = float(r["precision"])
            r["recall"] = float(r["recall"])
            r["round"] = int(r["round"])
            r["n_boxes"] = int(r.get("n_boxes") or 0)
        per_fold[f.stem.replace("fl_fold", "")] = rows

    print(f"{len(per_fold)} fold(s): {', '.join(sorted(per_fold))}\n")
    for k, rows in sorted(per_fold.items()):
        pos = rows[0]["n_boxes"] if rows else 0
        print(f"  fold {k}: held-out positives {pos}")

    rounds = [max(r["round"] for r in rows) for rows in per_fold.values()]
    rnd = args.round if args.round else min(rounds)
    print(f"\ncomparing at round {rnd}"
          + ("" if args.round else " (the last round every fold completed)"))

    arms: dict[str, list[float]] = defaultdict(list)
    for k, rows in sorted(per_fold.items()):
        locals_ = [r["f1"] for r in rows if r["arm"].startswith("local_")]
        if locals_:
            arms["local (mean of clients)"].append(sum(locals_) / len(locals_))
        for name, key in (("federated", "federated"), ("central", "central")):
            v = next((r["f1"] for r in rows
                      if r["arm"] == key and r["round"] == rnd), None)
            if v is not None:
                arms[name].append(v)
        s = next((r["f1"] for r in rows if r["arm"] == "start"), None)
        if s is not None:
            arms["starting model"].append(s)

    print(f"\n  {'arm':26}{'per fold':>26}{'mean':>8}{'sd':>8}")
    order = ["starting model", "local (mean of clients)", "federated", "central"]
    summary = {}
    for name in order:
        vals = arms.get(name)
        if not vals:
            continue
        mean = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        summary[name] = (mean, sd, vals)
        per = " ".join(f"{v:.3f}" for v in vals)
        print(f"  {name:26}{per:>26}{mean:>8.3f}{sd:>8.3f}")

    if "federated" in summary and "local (mean of clients)" in summary:
        fm, fsd, fv = summary["federated"]
        lm, lsd, lv = summary["local (mean of clients)"]
        gap = fm - lm
        noise = max(fsd, lsd)
        print(f"\n  federated minus local: {gap:+.3f}   fold-to-fold spread: {noise:.3f}")
        if len(fv) < 2:
            print("  One fold only. No spread, so no conclusion \u2014 run the others.")
        elif abs(gap) < noise:
            print("  The gap is smaller than the spread between folds. On this data")
            print("  federated learning is not distinguishable from training alone.")
            print("  That is a result worth reporting, not a failure to find one.")
        elif gap > 0:
            print("  Federated beats local training by more than the fold spread.")
        else:
            print("  Local training beats federated by more than the fold spread.")
            print("  With clients on separate routes that is plausible: averaging")
            print("  models fitted to different roads can help neither.")

    if "central" in summary and "federated" in summary:
        cm = summary["central"][0]
        fm = summary["federated"][0]
        print(f"\n  central {cm:.3f} vs federated {fm:.3f}")
        if fm > cm:
            print("  Federated above centralised. With few positives this usually")
            print("  means both are inside the noise rather than federated winning;")
            print("  check the per-fold columns before claiming anything.")

    out = RESULTS / "fl_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "round", "mean_f1", "sd_f1", "n_folds", "per_fold"])
        for name, (mean, sd, vals) in summary.items():
            w.writerow([name, rnd, round(mean, 4), round(sd, 4), len(vals),
                        " ".join(f"{v:.4f}" for v in vals)])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
