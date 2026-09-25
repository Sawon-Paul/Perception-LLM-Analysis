"""Hold out road, not vehicles.

Leave-one-route-out looked like the natural cross-validation, but routes A and C
share 57% of their cells. Holding out route A while training on route C means
most of the "unseen" road was in training. The fold leaks.

Spatial blocking fixes it. The map is divided into blocks; each fold holds out a
set of blocks entirely, and detections there from *every* trace become test data.
Nothing trained ever came from those places. A buffer strips training data near
the test region too, so a pothole visible from just outside a block cannot leak
in through a frame taken a few metres back.

This also suits the consensus experiment better than holding out a route would:
all nine traces still pass through the test blocks, so the number of distinct
voters at each test location is unchanged.

Blocks are precision-6 geohash cells (~1.2 km x 0.6 km). Sorting geohashes
lexically follows a Z-order curve, so consecutive cells are spatial neighbours;
cutting the sorted list into K runs gives spatially contiguous folds.

    python -m src.analysis.spatial_folds
    python -m src.analysis.spatial_folds --folds 3 --buffer 200
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.analysis.consensus_potential import cluster, load_detections  # noqa: E402
from src.analysis.trace_overlap import geohash_encode  # noqa: E402


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class RegionIndex:
    """Fast 'how far is this point from the test region' lookup."""

    def __init__(self, points: list[tuple[float, float]], cell_m: float):
        self.deg = cell_m / 111_320.0
        self.grid: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        for lat, lon in points:
            self.grid[(int(lat / self.deg), int(lon / self.deg))].append((lat, lon))

    def within(self, lat: float, lon: float, radius_m: float) -> bool:
        reach = int(math.ceil(radius_m / (self.deg * 111_320.0))) + 1
        gy, gx = int(lat / self.deg), int(lon / self.deg)
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                for plat, plon in self.grid.get((gy + dy, gx + dx), ()):
                    if haversine_m(lat, lon, plat, plon) <= radius_m:
                        return True
        return False


def load_positive_ids() -> set[str]:
    """Detection ids labelled as real faults, from every trace's label file."""
    import csv
    out = set()
    for p in sorted((cfg.DATA / "eval").glob("labels_*.csv")):
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r.get("is_real_fault", "").strip() == "1":
                out.add(r["detection_id"])
    return out


def make_folds(blocks: list[str], k: int,
               weights: dict[str, int] | None = None) -> list[list[str]]:
    """Contiguous runs of the Z-order-sorted block list.

    Splitting into equal numbers of *blocks* gave one fold 71% of the detections
    and left it 390 to train on, because blocks differ wildly in density — the
    town centre holds most of the data. With weights, each run is cut where the
    cumulative detection count reaches the next 1/k share, so folds hold similar
    amounts of data while staying spatially contiguous.
    """
    blocks = sorted(blocks)
    if not weights:
        size = math.ceil(len(blocks) / k)
        return [blocks[i * size:(i + 1) * size] for i in range(k)]

    total = sum(weights.get(b, 0) for b in blocks)
    w = [weights.get(b, 0) for b in blocks]
    cum = [0]
    for x in w:
        cum.append(cum[-1] + x)

    # Choose each cut point where the running total lands closest to its
    # target share. Cutting only once the target is *exceeded* let the first
    # fold overshoot into the dense centre and left the last fold starved.
    cuts, start = [], 0
    for f in range(1, k):
        target = total * f / k
        lo, hi = start + 1, len(blocks) - (k - f)   # leave a block for each later fold
        best = min(range(lo, hi + 1), key=lambda i: abs(cum[i] - target))
        cuts.append(best)
        start = best
    bounds = [0] + cuts + [len(blocks)]
    folds = [blocks[bounds[i]:bounds[i + 1]] for i in range(k)]
    return folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--block-precision", type=int, default=6)
    ap.add_argument("--buffer", type=float, default=200.0,
                    help="metres of training data stripped around each test region")
    ap.add_argument("--k", type=int, default=3, help="distinct owners for consensus")
    ap.add_argument("--radius", type=float, default=15.0)
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="split by block count")
    ap.add_argument("--balance-by", choices=["positives", "detections"],
                    default="positives")
    ap.add_argument("--pos-weight", type=float, default=20.0,
                    help="how much one real fault outweighs one detection")
    args = ap.parse_args()

    traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                    if d.is_dir() and d.name.upper().startswith("PVS"))
    dets = []
    for t in traces:
        dets.extend(load_detections(t, args.side, verified_only=False))
    if not dets:
        raise SystemExit("no detections found \u2014 run the pipeline first")

    for d in dets:
        d["block"] = geohash_encode(d["lat"], d["lon"], args.block_precision)

    blocks = sorted({d["block"] for d in dets})
    counts = Counter(d["block"] for d in dets)
    positives = load_positive_ids()
    pos_by_block = Counter(d["block"] for d in dets if d["id"] in positives)

    # Balancing on detection count left one fold with 27 positives to train on
    # while the others had over 100, because real faults cluster on asphalt and
    # most detections are cobblestone false positives. Training needs positives,
    # so each block is weighted mostly by the real faults it holds. Detections
    # still count a little, so blocks with no positives are spread rather than
    # all landing in one fold.
    if args.balance and args.balance_by == "positives" and positives:
        weights = {b: counts[b] + args.pos_weight * pos_by_block[b] for b in blocks}
    elif args.balance:
        weights = dict(counts)
    else:
        weights = None
    folds = make_folds(blocks, args.folds, weights)
    print(f"{len(positives)} labelled real faults; balancing by "
          f"{args.balance_by if args.balance else 'block count'}\n")

    biggest = max(counts.values())
    if biggest > len(dets) / args.folds:
        print(f"note: one block holds {biggest} detections "
              f"({100 * biggest / len(dets):.0f}%), more than a fold's share. "
              f"Perfect balance is impossible at this block size; try "
              f"--block-precision 7.\n")
    print(f"{len(dets)} detections in {len(blocks)} blocks "
          f"(geohash-{args.block_precision}), {args.folds} folds, "
          f"buffer {args.buffer:g} m\n")

    report = []
    for fi, test_blocks in enumerate(folds):
        tb = set(test_blocks)
        test = [d for d in dets if d["block"] in tb]
        rest = [d for d in dets if d["block"] not in tb]
        idx = RegionIndex([(d["lat"], d["lon"]) for d in test], args.buffer)
        train = [d for d in rest if not idx.within(d["lat"], d["lon"], args.buffer)]
        buffered = len(rest) - len(train)

        # consensus must still be measurable in the test region, or the fold
        # is useless for the experiment it exists to serve
        cl = cluster(test, args.radius, True)
        confirmable = [c for c in cl if len({d["trace"] for d in c}) >= args.k]
        voters = Counter(len({d["trace"] for d in c}) for c in cl)

        # leakage audit: nearest training detection to any test detection
        tr_idx = RegionIndex([(d["lat"], d["lon"]) for d in train], args.buffer)
        leaks = sum(1 for d in test if tr_idx.within(d["lat"], d["lon"], args.buffer))

        n_test_pos = sum(1 for d in test if d["id"] in positives)
        n_train_pos = sum(1 for d in train if d["id"] in positives)
        rec = {
            "fold": fi,
            "test_positives": n_test_pos,
            "train_positives": n_train_pos,
            "test_ids": [d["id"] for d in test],
            "train_ids": [d["id"] for d in train],
            "test_blocks": test_blocks,
            "n_test": len(test), "n_train": len(train), "n_buffered_out": buffered,
            "test_traces": sorted({d["trace"] for d in test}),
            "test_locations": len(cl),
            "test_locations_confirmable": len(confirmable),
            "leaks_within_buffer": leaks,
        }
        report.append(rec)

        print(f"fold {fi}: {len(test_blocks)} blocks held out")
        print(f"  test {len(test):5d}   train {len(train):5d}   "
              f"stripped by buffer {buffered:5d}")
        print(f"  real faults: test {n_test_pos:4d}   train {n_train_pos:4d}"
              + ("   <- too few to train on" if n_train_pos < 50 else ""))
        print(f"  traces present in test region: {len(rec['test_traces'])} of {len(traces)}")
        print(f"  test locations {len(cl)}, reaching k = {args.k}: {len(confirmable)}")
        print(f"  training detections within {args.buffer:g} m of test: {leaks}"
              + ("   <- leak" if leaks else "   (clean)"))
        print()

    tp = [r["train_positives"] for r in report]
    print(f"train positives {tp}   (smallest {min(tp)})")
    if min(tp) < 50:
        print(f"WARNING: a fold trains on only {min(tp)} real faults.\n")
    sizes = [r["n_test"] for r in report]
    trains = [r["n_train"] for r in report]
    print(f"test sizes  {sizes}   (largest / smallest = "
          f"{max(sizes) / max(min(sizes), 1):.1f}x)")
    print(f"train sizes {trains}   (smallest {min(trains)})\n")
    if min(trains) < 800:
        print(f"WARNING: a fold trains on only {min(trains)} detections. Its detector")
        print("will be undertrained and its results will understate the method.\n")

    tot_conf = sum(r["test_locations_confirmable"] for r in report)
    print("=" * 66)
    bad = [r["fold"] for r in report if r["test_locations_confirmable"] < 5]
    if any(r["leaks_within_buffer"] for r in report):
        print("A fold leaks. Increase --buffer.")
    elif bad:
        print(f"Folds {bad} have fewer than 5 confirmable test locations. They")
        print("cannot measure consensus reliably. Try --folds 2, or accept that")
        print("those folds report detection results only.")
    else:
        print(f"All folds clean and all support consensus measurement")
        print(f"({tot_conf} confirmable test locations in total).")
    print("=" * 66)

    out = cfg.DATA / "spatial_folds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "block_precision": args.block_precision,
        "buffer_m": args.buffer,
        "folds": report,
    }, indent=2))
    print(f"\nfold definitions -> {out}")


if __name__ == "__main__":
    main()
