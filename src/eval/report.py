"""Thesis results tables, with each method scored on its own overlap.

Two things this fixes.

**Per-method denominators.** The previous version took the positive and negative
counts from whichever verifier run happened to be first, then applied them to
every row. After the detector was retrained its runs no longer share detection
ids, so a row could report 16 true positives against a denominator of 7 — a
recall of 228%. Each method is now scored only on the labelled detections it
actually produced, and its own n is printed beside it.

**Detector alone.** The comparison lacked the row that matters most after
fine-tuning: what the detector achieves with no verifier at all, at its native
threshold. That is now the first row.

Rates carry Wilson 95% intervals, which stay inside [0,1] at small n and near 0.

    python -m src.eval.report --trace "PVS 2" --out RESULTS.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.eval.metrics import load_labels, merge_baseline, pr_auc, prf  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or k < 0:
        return (0.0, 0.0)
    k = min(k, n)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k: int, n: int) -> str:
    if n <= 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * min(k, n) / n:.1f}% [{100 * lo:.1f}–{100 * hi:.1f}]"


def collect(trace: str, side: str) -> dict[str, list[dict]]:
    tag = cfg.slug(trace)
    found = {}
    for suffix, name in (("_p2_standalone", "Qwen2.5-VL-3B"),
                         ("_p2_standalone_7b", "Qwen2.5-VL-7B"),
                         ("_p2", "3B (gated)"),
                         ("_p2_nogate", "3B (no gate)")):
        p = cfg.EVENTS / f"events_{tag}_{side}{suffix}.jsonl"
        if p.exists():
            ev = load_jsonl(p)
            merge_baseline(ev, tag, side)
            found[name] = ev
    return found


def score_run(events: list[dict], labels: dict[str, int], mode: str) -> dict | None:
    """Score one method on the labelled subset of its own detections.

    mode: 'any' every detection the detector emitted
          'conf' detector confidence >= 0.5
          'rule' the sensor rule baseline
          'llm'  the verifier's verdict
    """
    ev = [e for e in events if e["detection_id"] in labels]
    if not ev:
        return None
    y = [labels[e["detection_id"]] for e in ev]
    n_pos, n_neg = sum(y), len(y) - sum(y)

    def p2(e):
        r = e.get("phase2") or {}
        return r.get("confidence", 0.0) * (1.0 if r.get("verified") else 0.0)

    if mode == "any":
        pred = [1] * len(ev)
        sc = [e["stream5_model"].get("conf") or 0.0 for e in ev]
    elif mode == "conf":
        sc = [e["stream5_model"].get("conf") or 0.0 for e in ev]
        pred = [int(s >= 0.5) for s in sc]
    elif mode == "rule":
        pred = [int(bool((e.get("baseline") or {}).get("flagged"))) for e in ev]
        sc = [float(p) for p in pred]
    else:
        pred = [int(bool((e.get("phase2") or {}).get("verified"))) for e in ev]
        sc = [p2(e) for e in ev]

    m = prf(y, pred)
    return {**m, "n": len(ev), "n_pos": n_pos, "n_neg": n_neg,
            "pr_auc": pr_auc(y, sc), "scores": sc, "y": y}


def row(name: str, m: dict) -> str:
    return (f"| {name} | {m['n']} | {m['n_pos']} | {m['fp']} / {m['n_neg']} | "
            f"{fmt(m['fp'], m['n_neg'])} | {fmt(m['tp'], m['n_pos'])} | "
            f"{m['precision']:.3f} | {m['f1']:.3f} | {m['pr_auc']:.3f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = load_labels(args.trace)
    runs = collect(args.trace, args.side)
    if not runs:
        raise SystemExit(f"no verifier output for {args.trace}")

    L: list[str] = [f"# Results — {args.trace}\n"]
    L.append("Every method is scored only on the labelled detections it actually "
             "produced, so the counts differ between rows — a retrained detector "
             "emits different detections from the original. The n and positive "
             "columns make that explicit.\n")
    L.append("Rates carry Wilson 95% intervals. A false-positive rate measured on "
             "hundreds of negatives is reasonably tight; a recall measured on a "
             "handful of positives is not.\n")

    L.append("\n## Main comparison\n")
    L.append("| method | n | positives | false positives | FP rate [95% CI] | "
             "recall [95% CI] | precision | F1 | PR-AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    # The detector-alone rows must come from the run with the most labelled
    # detections, not whichever file sorts first. After retraining, the old
    # run's detections barely overlap the current labels, and taking those rows
    # from it reported the detector on 6 samples instead of 25.
    best_name = max(runs, key=lambda k: sum(
        1 for e in runs[k] if e["detection_id"] in labels))
    base_ev = runs[best_name]
    n_overlap = sum(1 for e in base_ev if e["detection_id"] in labels)
    L.append(f"Detector rows are taken from the run with the most labelled "
             f"detections ({best_name}, {n_overlap} of them).\n")

    for label, mode, ev in (
            ("Detector alone, every detection", "any", base_ev),
            ("Detector alone, conf>=0.5", "conf", base_ev),
            ("Sensor rule baseline", "rule", base_ev)):
        m = score_run(ev, labels, mode)
        if m:
            L.append(row(label, m))

    scored = {}
    for name, ev in runs.items():
        m = score_run(ev, labels, "llm")
        if m:
            scored[name] = m
            L.append(row(f"+ {name}", m))

    L.append("\n## Output validity\n")
    L.append("| model | self-inconsistent JSON | parse failures |")
    L.append("|---|---|---|")
    for name, ev in runs.items():
        done = [e.get("phase2") for e in ev if e.get("phase2")]
        inc = sum(1 for r in done if r.get("self_inconsistent"))
        fails = sum(1 for r in done
                    if str(r.get("reason", "")).startswith("phase2_failed"))
        L.append(f"| {name} | {inc} / {len(done)} "
                 f"({100 * inc / max(len(done), 1):.1f}%) | {fails} |")
    L.append("\nSelf-inconsistent means the model contradicted its own severity or "
             "fault type. Those are corrected in code, so the rate measures output "
             "reliability rather than accuracy.\n")

    L.append("\n## Operating point\n")
    L.append("0.5 was a default, not a choice. No threshold is fitted here: the "
             "positive counts are too small for a fit to be anything but "
             "memorisation.\n")
    for name, m in scored.items():
        L.append(f"\n**{name}** (n={m['n']}, {m['n_pos']} positive)\n")
        L.append("| threshold | TP | FP | FN | precision | recall |")
        L.append("|---|---|---|---|---|---|")
        for thr in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            r = prf(m["y"], [int(s > thr) for s in m["scores"]])
            L.append(f"| {thr:.1f} | {r['tp']} | {r['fp']} | {r['fn']} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} |")

    L.append("\n## Limitations\n")
    for name, m in scored.items():
        L.append(f"- {name} was scored on {m['n']} labelled detections with "
                 f"{m['n_pos']} true faults.\n")
    L.append("- Labels were produced with AI assistance under a written rubric "
             "(cobblestone joints, shadows, sealant lines and completed repairs "
             "count as negatives) and spot-checked by hand. State this.\n")
    L.append("- Fusion weights and the Phase 1 gate are unfitted defaults.\n")

    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
