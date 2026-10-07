"""Precision-recall and ROC curves for the verification comparison (thesis Fig. 5.2b).

Uses exactly the same inputs and scores as src.eval.report, so the PR-AUC in the
legend matches the PR-AUC column of the results table:

  YOLO11s alone   score = detector confidence
  + 3B / + 7B     score = verifier confidence if verified, else 0

Each method is scored only on the labelled detections it produced.

    python -m src.eval.plot_curves --trace "PVS 2"
    python -m src.eval.plot_curves --trace "PVS 2" --out results\\fig_pr_roc.png

Writes the figure (PNG + PDF) and a CSV of every curve point.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.eval.metrics import load_labels, pr_auc  # noqa: E402
from src.eval.report import collect  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

STYLE = {  # name -> (colour, line style)
    "YOLO11s alone": ("#2a6fdb", "-"),
    "+ Qwen2.5-VL-3B": ("#e8663d", "-"),
    "+ Qwen2.5-VL-7B": ("#1aa37a", "-"),
}


def verifier_score(e: dict) -> float:
    r = e.get("phase2") or {}
    return float(r.get("confidence", 0.0) or 0.0) * (1.0 if r.get("verified") else 0.0)


def series(events: list[dict], labels: dict[str, int], detector: bool):
    ev = [e for e in events if e["detection_id"] in labels]
    y = [labels[e["detection_id"]] for e in ev]
    if detector:
        s = [float(e["stream5_model"].get("conf") or 0.0) for e in ev]
    else:
        s = [verifier_score(e) for e in ev]
    return y, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--out", default=str(Path("results") / "fig_pr_roc.png"))
    ap.add_argument("--labels", default=None,
                    help="label CSV to use instead of data/eval/labels_<trace>.csv "
                         "(the thesis set is 148 cards, 5 positive)")
    ap.add_argument("--e3b", default=None, help="3B verifier events .jsonl to use instead")
    ap.add_argument("--e7b", default=None, help="7B verifier events .jsonl to use instead")
    args = ap.parse_args()

    if args.labels:
        labels = {}
        for r in csv.DictReader(Path(args.labels).open(encoding="utf-8")):
            v = r["is_real_fault"].strip()
            if v in ("0", "1"):
                labels[r["detection_id"]] = int(v)
    else:
        labels = load_labels(args.trace)
    pos_all = sum(labels.values())
    print(f"labels: {len(labels)} cards, {pos_all} positive, {len(labels) - pos_all} negative")
    if len(labels) - pos_all < 20:
        print("WARNING: very few negatives. This is not the 148-card thesis set "
              "(5 positive / 143 negative); ROC on this set is meaningless.")

    runs = collect(args.trace, args.side)
    for flag, name in ((args.e3b, "Qwen2.5-VL-3B"), (args.e7b, "Qwen2.5-VL-7B")):
        if flag:
            runs[name] = load_jsonl(Path(flag))
    if not runs:
        raise SystemExit(f"no verifier output for {args.trace}")

    # Detector row from the run with the most labelled detections (as in report.py).
    best = max(runs, key=lambda k: sum(1 for e in runs[k] if e["detection_id"] in labels))
    data = {"YOLO11s alone": series(runs[best], labels, detector=True)}
    if "Qwen2.5-VL-3B" in runs:
        data["+ Qwen2.5-VL-3B"] = series(runs["Qwen2.5-VL-3B"], labels, detector=False)
    if "Qwen2.5-VL-7B" in runs:
        data["+ Qwen2.5-VL-7B"] = series(runs["Qwen2.5-VL-7B"], labels, detector=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    prevalence = None
    print(f"\n{args.trace}: curves on labelled crops\n")
    print(f"{'method':<18}{'n':>5}{'pos':>5}{'PR-AUC':>9}{'ROC-AUC':>9}")
    for name, (y, s) in data.items():
        n, pos = len(y), sum(y)
        if pos == 0 or pos == n:
            print(f"{name:<18} skipped (only one class)")
            continue
        prevalence = pos / n
        ap_ = pr_auc(y, s)
        roc = roc_auc_score(y, s)
        col, ls = STYLE[name]

        p, r, _ = precision_recall_curve(y, s)
        ax1.step(r, p, where="post", color=col, ls=ls, lw=2.2,
                 label=f"{name} (PR-AUC = {ap_:.3f})")
        fpr, tpr, _ = roc_curve(y, s)
        ax2.plot(fpr, tpr, color=col, ls=ls, lw=2.2, drawstyle="steps-post",
                 label=f"{name} (ROC-AUC = {roc:.3f})")

        rows += [(name, "PR", float(a), float(b)) for a, b in zip(r, p)]
        rows += [(name, "ROC", float(a), float(b)) for a, b in zip(fpr, tpr)]
        print(f"{name:<18}{n:>5}{pos:>5}{ap_:>9.3f}{roc:>9.3f}")

    if prevalence is not None:
        ax1.axhline(prevalence, color="grey", ls="--", lw=1.2,
                    label=f"Random guess (precision = {prevalence:.3f})")
    ax2.plot([0, 1], [0, 1], color="grey", ls="--", lw=1.2, label="Random guess (AUC = 0.5)")

    ax1.set(xlabel="Recall", ylabel="Precision", xlim=(-0.02, 1.02), ylim=(-0.02, 1.05),
            title="(a) Precision–recall curve")
    ax2.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
            xlim=(-0.02, 1.02), ylim=(-0.02, 1.05), title="(b) ROC curve")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is ax2 else "upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))

    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "curve", "x", "y"])
        w.writerows(rows)
    print(f"\nwrote {out}, {out.with_suffix('.pdf')}, {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
