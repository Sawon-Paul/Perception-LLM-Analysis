"""Which traces drove the same route?

The cell counts hint that the nine PVS traces form three groups of three, but a
hint is not a fold definition. Leave-one-route-out cross-validation needs the
groups decided by data, and if two traces from the same route end up in
different folds, the held-out fold is not held out.

Method: pairwise Jaccard similarity of the geohash cells each trace drove, then
single-linkage grouping above a threshold. Two traces on the same route share
most of their cells; two on different routes share only the common start and
end segments.

    python -m src.analysis.route_groups
    python -m src.analysis.route_groups --threshold 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.analysis.trace_overlap import trace_cells  # noqa: E402


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def group(traces: list[str], sim: dict, threshold: float) -> list[list[str]]:
    """Union-find: link any pair above the threshold."""
    parent = {t: t for t in traces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), s in sim.items():
        if s >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    groups: dict[str, list[str]] = {}
    for t in traces:
        groups.setdefault(find(t), []).append(t)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--precision", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Jaccard similarity that counts as the same route")
    args = ap.parse_args()

    traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                    if d.is_dir() and d.name.upper().startswith("PVS"))
    cells = {t: trace_cells(t, args.side, args.precision, 2.0) for t in traces}

    sim = {}
    for i, a in enumerate(traces):
        for b in traces[i + 1:]:
            sim[(a, b)] = jaccard(cells[a], cells[b])

    # similarity matrix
    short = [t.replace("PVS ", "") for t in traces]
    print("Jaccard similarity of cells driven\n")
    print("        " + "".join(f"{s:>6}" for s in short))
    for a in traces:
        row = []
        for b in traces:
            if a == b:
                row.append("   \u2014 ")
            else:
                key = (a, b) if (a, b) in sim else (b, a)
                row.append(f"{sim[key]:6.2f}")
        print(f"  {a.replace('PVS ', 'PVS'):>5} " + "".join(row))

    groups = group(traces, sim, args.threshold)
    print(f"\ngroups at threshold {args.threshold}\n")
    for i, g in enumerate(groups):
        within = [sim.get((a, b), sim.get((b, a), 0))
                  for j, a in enumerate(g) for b in g[j + 1:]]
        w = f"within {min(within):.2f}\u2013{max(within):.2f}" if within else "single trace"
        print(f"  route {chr(65 + i)}: {', '.join(g)}   ({w})")

    # separation check: the lowest within-group similarity should clearly
    # exceed the highest between-group similarity, or the grouping is arbitrary
    member = {t: i for i, g in enumerate(groups) for t in g}
    within_all = [s for (a, b), s in sim.items() if member[a] == member[b]]
    between = [s for (a, b), s in sim.items() if member[a] != member[b]]
    if within_all and between:
        gap = min(within_all) - max(between)
        print(f"\n  lowest within-route similarity  {min(within_all):.2f}")
        print(f"  highest between-route similarity {max(between):.2f}")
        print(f"  gap                               {gap:+.2f}")
        if gap > 0.15:
            print("\n  Clean separation. These groups are well defined and safe to use")
            print("  as cross-validation folds.")
        elif gap > 0:
            print("\n  Separated, but narrowly. Check the groups against the dataset")
            print("  documentation before relying on them.")
        else:
            print("\n  NOT separated: some pair across routes is as similar as a pair")
            print("  within one. The threshold is arbitrary here. Do not build folds")
            print("  on this grouping without an independent source.")

    out = cfg.DATA / "route_groups.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "threshold": args.threshold,
        "groups": {chr(65 + i): g for i, g in enumerate(groups)},
        "folds": [{"test": g, "train": [t for t in traces if t not in g]}
                  for g in groups],
    }, indent=2))
    print(f"\ngroups and leave-one-route-out folds -> {out}")


if __name__ == "__main__":
    main()
