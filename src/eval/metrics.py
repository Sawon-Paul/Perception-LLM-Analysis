"""Scores every method against your manual labels. Output is your results table.

Methods compared:
  yolo_only     detector confidence alone
  rule          the sensor + confidence baseline
  phase1_only   text-only prior, no image
  phase2_only   multimodal, no prior (the ablation that justifies two phases)
  fused         the full two-phase pipeline

If fused does not beat both rule and phase2_only, say so in the thesis and drop
the phase you cannot justify. A negative ablation honestly reported is a valid
result; an unjustified architecture is not.

    python -m src.eval.metrics --side left
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402

EVAL = cfg.DATA / "eval"


def load_labels(trace: str | None = None) -> dict[str, int]:
    path = (EVAL / f"labels_{cfg.slug(trace)}.csv") if trace else (EVAL / "labels.csv")
    if trace and not path.exists() and (EVAL / "labels.csv").exists():
        path = EVAL / "labels.csv"      # fall back to the old shared file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python -m src.eval.make_eval_set` and label first.")
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        v = r["is_real_fault"].strip()
        if v in ("0", "1"):
            out[r["detection_id"]] = int(v)
    return out


def prf(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def pr_auc(y_true: list[int], scores: list[Optional[float]]) -> float:
    """Average precision. Threshold-free, so it survives a bad threshold choice."""
    pairs = [(s, t) for s, t in zip(scores, y_true) if s is not None]
    if not pairs or all(t == 0 for _, t in pairs):
        return 0.0
    pairs.sort(key=lambda x: -x[0])
    total_pos = sum(t for _, t in pairs)
    tp = 0
    ap = 0.0
    for i, (_, t) in enumerate(pairs, start=1):
        if t == 1:
            tp += 1
            ap += tp / i
    return round(ap / total_pos, 4)


def merge_baseline(events: list[dict], tag: str, side: str) -> int:
    """The rule baseline is written to its own file, so pull it back in here.

    Without this the rule column scores every event as unflagged and reports a
    flat zero, which looks like a result and is really a missing join.
    """
    path = cfg.EVENTS / f"events_{tag}_{side}_baseline.jsonl"
    if not path.exists():
        print(f"note: {path.name} not found — run src.eval.baseline to score the rule")
        return 0
    by_id = {e["detection_id"]: e.get("baseline") for e in load_jsonl(path)}
    n = 0
    for e in events:
        b = by_id.get(e["detection_id"])
        if b is not None:
            e["baseline"] = b
            n += 1
    return n


def evaluate(events: list[dict], labels: dict[str, int]) -> dict[str, dict]:
    ev = [e for e in events if e["detection_id"] in labels]
    if not ev:
        raise RuntimeError("no overlap between events and labels — wrong --side?")
    y = [labels[e["detection_id"]] for e in ev]

    def p1(e):
        return (e.get("phase1") or {}).get("prior_prob")

    def p2(e):
        r = e.get("phase2") or {}
        return r.get("confidence", 0.0) * (1.0 if r.get("verified") else 0.0)

    methods: dict[str, tuple[Callable, Callable]] = {
        "yolo_only":   (lambda e: e["stream5_model"].get("conf"),
                        lambda e: int((e["stream5_model"].get("conf") or 0) >= 0.5)),
        "rule":        (lambda e: float(bool((e.get("baseline") or {}).get("flagged"))),
                        lambda e: int(bool((e.get("baseline") or {}).get("flagged")))),
        "phase1_only": (p1, lambda e: int((p1(e) or 0) >= 0.5)),
        "phase2_only": (p2, lambda e: int(bool((e.get("phase2") or {}).get("verified")))),
        "fused":       (lambda e: e.get("final_score"),
                        lambda e: int((e.get("final_score") or 0) >= 0.5)),
    }

    results = {}
    for name, (score_fn, pred_fn) in methods.items():
        scores = [score_fn(e) for e in ev]
        if all(s is None for s in scores):
            continue
        results[name] = {**prf(y, [pred_fn(e) for e in ev]),
                         "pr_auc": pr_auc(y, scores),
                         "n_scored": sum(1 for s in scores if s is not None)}
    results["_meta"] = {"n_labelled": len(ev), "n_positive": sum(y)}
    return results


def print_table(results: dict[str, dict]) -> None:
    meta = results.pop("_meta")
    print(f"\nn={meta['n_labelled']} labelled, {meta['n_positive']} positive\n")
    hdr = f"{'method':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'PR-AUC':>9}{'TP':>6}{'FP':>6}{'FN':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<14}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}"
              f"{r['pr_auc']:>9.3f}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}")

    if "fused" in results and "rule" in results:
        d = results["fused"]["f1"] - results["rule"]["f1"]
        print(f"\nfused minus rule baseline: {d:+.3f} F1")
        if d <= 0.02:
            print("The LLM does not beat a free rule. Report this honestly and "
                  "reconsider the architecture before writing the results chapter.")
    if "fused" in results and "phase2_only" in results:
        if "phase1_only" not in results:
            print("\nPhase 1 was not run, so fused and phase2_only are the same "
                  "numbers. The two-phase ablation needs a Phase 1 pass first.")
        else:
            d = results["fused"]["f1"] - results["phase2_only"]["f1"]
            print(f"fused minus phase2-only:   {d:+.3f} F1")
            if d <= 0.01:
                print("Phase 1 adds nothing measurable. Either drop it or justify "
                      "it on cost, not accuracy — and show the cost numbers.")

    meta_pos = meta["n_positive"]
    if meta_pos < 15:
        print(f"\nOnly {meta_pos} positives in the labelled set. Precision and "
              f"recall here have very wide confidence intervals — one label "
              f"changing moves them a lot. Report the counts alongside the rates, "
              f"and treat differences between methods as indicative, not proven.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--events", default=None)
    args = ap.parse_args()

    if args.events:
        path = Path(args.events)
    else:
        tag = cfg.slug(args.trace)
        for suffix in ("_p2", "_p2_standalone", "_p2_nogate", ""):
            cand = cfg.EVENTS / f"events_{tag}_{args.side}{suffix}.jsonl"
            if cand.exists():
                path = cand
                break
        else:
            raise FileNotFoundError(f"no events file for {tag}/{args.side}")
    print(f"scoring {path.name}")
    events = load_jsonl(path)
    n_base = merge_baseline(events, cfg.slug(args.trace), args.side)
    if n_base:
        print(f"merged rule baseline for {n_base} events")
    print_table(evaluate(events, load_labels(args.trace)))


if __name__ == "__main__":
    main()
