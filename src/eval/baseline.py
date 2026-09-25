"""The baseline the LLM has to beat, plus data-derived thresholds.

Two jobs:

1. derive_thresholds — computes the jolt threshold from the PVS data instead of
   picking a number. Uses the 95th percentile of jolt on good asphalt as the
   "normal road" ceiling. This is what you cite when an examiner asks where the
   G-force threshold came from.

2. rule_baseline — flag if detector confidence is high AND the jolt exceeds that
   threshold. Costs nothing, no API, no GPU. If the two-phase LLM does not beat
   this, the LLM is not earning its place in the thesis.

    python -m src.eval.baseline --side left
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.io import load_jsonl, write_jsonl  # noqa: E402


def derive_thresholds(trace: str, side: str = "left") -> dict[str, float]:
    """Thresholds from data, not from taste."""
    tag = cfg.slug(trace)
    df = pd.read_csv(cfg.RAW / f"frames_{tag}_{side}.csv")
    df = df[df["jolt_z"].notna()]

    clean = df[(df.get("asphalt_road", 0) == 1)
               & ((df.get("good_road_left", 0) == 1) | (df.get("good_road_right", 0) == 1))
               & (df.get("no_speed_bump", 1) == 1)]
    if len(clean) < 100:
        print(f"WARNING: only {len(clean)} clean-asphalt rows — threshold is shaky")
        clean = df

    th = {
        "jolt_p50_clean": float(clean["jolt_z"].quantile(0.50)),
        "jolt_p95_clean": float(clean["jolt_z"].quantile(0.95)),
        "jolt_p99_clean": float(clean["jolt_z"].quantile(0.99)),
        "n_clean_rows": int(len(clean)),
        "n_total_rows": int(len(df)),
    }
    th["jolt_threshold"] = th["jolt_p95_clean"]

    bad = df[(df.get("bad_road_left", 0) == 1) | (df.get("bad_road_right", 0) == 1)]
    if len(bad):
        th["jolt_p50_bad"] = float(bad["jolt_z"].quantile(0.50))
        th["separation_ratio"] = round(th["jolt_p50_bad"] / max(th["jolt_p50_clean"], 1e-9), 3)

    out = cfg.DATA / f"thresholds_{tag}.json"
    out.write_text(json.dumps(th, indent=2))
    print(json.dumps(th, indent=2))
    print(f"-> {out}")
    if th.get("separation_ratio", 0) < 1.5:
        print("WARNING: bad-road jolt barely exceeds good-road jolt. The sensor "
              "stream may carry little signal at this extraction rate.")
    return th


def rule_baseline(events: list[dict], jolt_threshold: float,
                  conf_threshold: float = 0.50) -> list[dict]:
    for ev in events:
        conf = ev["stream5_model"].get("conf") or 0.0
        # rough for THIS surface, over the road ahead
        pct = ev["stream2_sensor"].get("jolt_ahead_pct_for_surface")
        bump = ev["stream2_sensor"].get("speed_bump_labelled")
        ev["baseline"] = {
            "flagged": bool(conf >= conf_threshold
                            and pct is not None and pct >= 95.0
                            and not bump),
            "rule": f"conf>={conf_threshold} AND jolt ahead (3s) in the top 5% "
                    f"for this surface type AND not a labelled speed bump",
        }
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--events", default=None)
    args = ap.parse_args()

    th = derive_thresholds(args.trace, args.side)
    tag = cfg.slug(args.trace)

    path = Path(args.events) if args.events else cfg.EVENTS / f"events_{tag}_{args.side}.jsonl"
    if not path.exists():
        print(f"\n{path} not found — thresholds written, baseline skipped.")
        return

    events = rule_baseline(load_jsonl(path), th["jolt_threshold"])
    out = path.with_name(path.stem + "_baseline.jsonl")
    write_jsonl(out, events)
    n = sum(1 for e in events if e["baseline"]["flagged"])
    print(f"\nbaseline flagged {n}/{len(events)} -> {out}")


if __name__ == "__main__":
    main()
