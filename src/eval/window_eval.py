"""Evaluate against the dataset's own road-quality labels, not 5 hand labels.

The hand-labelled set has 148 crops and 5 positives. Every rate computed on it
moves by a third if one label changes, so no difference between methods can be
claimed from it. That is the single biggest weakness in the results so far.

But thousands of ground-truth labels already exist and were never used:
dataset_labels.csv records good / regular / bad road for both wheel tracks at
every sample, annotated by the dataset authors.

So evaluate at window level. Cut each drive into fixed windows, ask whether the
system flags damage in each, and compare against the recorded road quality.
Across nine traces that yields thousands of windows and hundreds of positives,
and the confidence intervals become small enough to defend.

The catch, and it is reported rather than hidden: if "bad road" in this dataset
mostly coincides with cobblestone, then window-level scores measure surface
recognition rather than damage detection. The surface-by-quality table printed
below is what tells you which of the two you are measuring.

    python -m src.eval.window_eval --trace "PVS 2"
    python -m src.eval.window_eval --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.eval.metrics import pr_auc, prf  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402


def load_quality(trace: str, side: str) -> pd.DataFrame:
    pvs = cfg.pvs_dir(trace)
    sens = pd.read_csv(pvs / f"dataset_gps_mpu_{side}.csv",
                       usecols=["timestamp", "speed"])
    labels = pd.read_csv(pvs / "dataset_labels.csv")
    if len(sens) != len(labels):
        raise ValueError(f"{trace}: sensors={len(sens)} labels={len(labels)}")
    df = pd.concat([sens, labels], axis=1).sort_values("timestamp").reset_index(drop=True)

    def surface(r) -> str:
        for c in ("cobblestone_road", "dirt_road", "asphalt_road"):
            if r.get(c) == 1:
                return c.replace("_road", "")
        if r.get("unpaved_road") == 1:
            return "unpaved"
        if r.get("paved_road") == 1:
            return "paved"
        return "unknown"

    df["surface"] = df.apply(surface, axis=1)
    df["bad"] = ((df.get("bad_road_left", 0) == 1)
                 | (df.get("bad_road_right", 0) == 1)).astype(int)
    df["regular"] = ((df.get("regular_road_left", 0) == 1)
                     | (df.get("regular_road_right", 0) == 1)).astype(int)
    return df


def surface_quality_table(df: pd.DataFrame) -> None:
    """Is "bad road" really damage, or just a proxy for the surface type?"""
    tab = pd.crosstab(df["surface"], df["bad"], normalize="index") * 100
    print("\n  share of samples labelled BAD, by surface:")
    for surf in tab.index:
        share = tab.loc[surf, 1] if 1 in tab.columns else 0.0
        n = int((df["surface"] == surf).sum())
        print(f"    {surf:12} {share:5.1f}%   (n={n})")

    if 1 in tab.columns:
        spread = tab[1].max() - tab[1].min()
        if spread > 60:
            print("\n  WARNING: bad-road labels are concentrated on one surface type.")
            print("  Window scores here largely measure surface recognition, not")
            print("  damage detection. Say so explicitly when reporting them.")
        else:
            print("\n  Bad-road labels are spread across surfaces, so window scores")
            print("  reflect road condition rather than surface type.")


def build_windows(trace: str, side: str, window_s: float,
                  min_speed_kmh: float) -> pd.DataFrame:
    q = load_quality(trace, side)
    surface_quality_table(q)

    t0 = float(q["timestamp"].min())
    q["win"] = ((q["timestamp"] - t0) // window_s).astype(int)
    agg = q.groupby("win").agg(
        t_start=("timestamp", "min"),
        t_end=("timestamp", "max"),
        bad=("bad", "max"),
        speed=("speed", "mean"),
        surface=("surface", lambda s: s.mode().iat[0] if len(s) else "unknown"),
    ).reset_index()

    # A parked vehicle produces no ride information, so those windows cannot be
    # scored fairly either way.
    before = len(agg)
    agg = agg[agg["speed"] * 3.6 >= min_speed_kmh]
    print(f"\n  {len(agg)} windows of {window_s:g}s "
          f"({before - len(agg)} dropped as stationary), "
          f"{int(agg['bad'].sum())} labelled bad")
    return agg


def score_windows(agg: pd.DataFrame, events: list[dict], window_s: float,
                  t0: float) -> dict:
    """Attach each detection to its window and score three methods."""
    by_win: dict[int, list[dict]] = {}
    for e in events:
        ts = e.get("timestamp")
        if ts is None:
            continue
        by_win.setdefault(int((float(ts) - t0) // window_s), []).append(e)

    rows = []
    for _, w in agg.iterrows():
        evs = by_win.get(int(w["win"]), [])
        yolo = max((e["stream5_model"].get("conf") or 0.0 for e in evs), default=0.0)
        p2 = max(((e.get("phase2") or {}).get("confidence", 0.0)
                  * (1.0 if (e.get("phase2") or {}).get("verified") else 0.0)
                  for e in evs), default=0.0)
        fused = max((e.get("final_score") or 0.0 for e in evs), default=0.0)
        rows.append({"bad": int(w["bad"]), "surface": w["surface"],
                     "n_det": len(evs), "yolo": yolo, "p2": p2, "fused": fused})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    y = df["bad"].tolist()
    print(f"\n  n={len(df)} windows, {sum(y)} bad ({100 * sum(y) / max(len(df), 1):.1f}%)")

    methods = {
        "any_detection": (df["n_det"] > 0).astype(float).tolist(),
        "yolo_conf": df["yolo"].tolist(),
        "phase2": df["p2"].tolist(),
        "fused": df["fused"].tolist(),
    }
    hdr = f"  {'method':<16}{'prec':>8}{'rec':>8}{'f1':>8}{'PR-AUC':>9}{'TP':>6}{'FP':>6}{'FN':>6}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, scores in methods.items():
        thr = 0.5 if name != "any_detection" else 0.5
        pred = [int(s >= thr) for s in scores]
        m = prf(y, pred)
        print(f"  {name:<16}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{pr_auc(y, scores):>9.3f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}")

    print("\n  by surface (fused, threshold 0.5):")
    for surf, g in df.groupby("surface"):
        if len(g) < 20:
            continue
        m = prf(g["bad"].tolist(), [int(s >= 0.5) for s in g["fused"]])
        print(f"    {surf:12} n={len(g):5d} bad={int(g['bad'].sum()):4d}  "
              f"prec {m['precision']:.3f}  rec {m['recall']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--window", type=float, default=5.0, help="window length in seconds")
    ap.add_argument("--min-speed", type=float, default=5.0, help="km/h")
    ap.add_argument("--suffix", default=None,
                    help='which verifier run to score, e.g. "_p2_standalone_7b"')
    args = ap.parse_args()

    traces = (sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                     if d.is_dir() and d.name.upper().startswith("PVS"))
              if args.all else [args.trace])

    frames = []
    for tr in traces:
        tag = cfg.slug(tr)
        path = None
        order = ([args.suffix] if args.suffix
                 else ("_p2", "_p2_standalone", "_p2_nogate", ""))
        for suffix in order:
            cand = cfg.EVENTS / f"events_{tag}_{args.side}{suffix}.jsonl"
            if cand.exists():
                path = cand
                break
        if path is None:
            print(f"\n=== {tr} === no events file, skipping")
            continue

        print(f"\n=== {tr} ===  scoring {path.name}")
        try:
            events = load_jsonl(path)
            agg = build_windows(tr, args.side, args.window, args.min_speed)
            q0 = float(load_quality(tr, args.side)["timestamp"].min())
            df = score_windows(agg, events, args.window, q0)
            df["trace"] = tr
            report(df)
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {str(e)[:200]}")

    if len(frames) > 1:
        print("\n" + "=" * 70)
        print("POOLED ACROSS ALL TRACES")
        print("=" * 70)
        report(pd.concat(frames, ignore_index=True))

    if frames:
        out = cfg.DATA / "window_eval.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
