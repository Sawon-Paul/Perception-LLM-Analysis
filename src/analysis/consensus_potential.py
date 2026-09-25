"""How many faults would actually reach k confirmations?

trace_overlap.py answered whether other cars *drove* past a detection. That is
necessary but not sufficient. A fault is confirmed only when another car's
detector also fires at the same place, so this clusters detections across traces
by distance and counts how many clusters contain k or more distinct traces.

The gap between the two numbers is itself a result. If 99.9% of detections sit
where other cars drove but only a fraction of clusters reach k, the limit is
detector agreement rather than road coverage, and that is what the FL and
verification chapters are for.

Clustering uses the plan's match_radius_m, because that is the rule the car
applies before deciding whether to call report() or confirm().

    python -m src.analysis.consensus_potential
    python -m src.analysis.consensus_potential --k 3 --radius 15 --verified-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_detections(trace: str, side: str, verified_only: bool) -> list[dict]:
    tag = cfg.slug(trace)
    for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
        p = cfg.EVENTS / f"events_{tag}_{side}{suffix}.jsonl"
        if not p.exists():
            continue
        out = []
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("stream5_model", {}).get("stream") != "pothole":
                continue
            lat, lon = e.get("lat"), e.get("lon")
            if lat is None or lon is None:
                continue
            if verified_only and not (e.get("phase2") or {}).get("verified"):
                continue
            out.append({"trace": trace, "id": e["detection_id"],
                        "lat": float(lat), "lon": float(lon),
                        "type": e["stream5_model"].get("class_id", 0),
                        "conf": e["stream5_model"].get("conf") or 0.0,
                        "verified": bool((e.get("phase2") or {}).get("verified")),
                        "surface": ((e.get("stream2_sensor") or {})
                                    .get("measured_surface") or "unknown"),
                        "detector": e["stream5_model"].get("detector", "")})
        return out
    return []


def cluster(dets: list[dict], radius_m: float, same_type: bool) -> list[list[dict]]:
    """Greedy spatial clustering at the radius the car itself matches on.

    A coarse grid keeps this near-linear: only detections in the same or an
    adjacent grid square can be within the radius, so the quadratic comparison
    never runs across the whole dataset.
    """
    deg = radius_m / 111_320.0
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, d in enumerate(dets):
        grid[(int(d["lat"] / deg), int(d["lon"] / deg))].append(i)

    seen = [False] * len(dets)
    clusters = []
    for i, d in enumerate(dets):
        if seen[i]:
            continue
        seen[i] = True
        group = [d]
        gy, gx = int(d["lat"] / deg), int(d["lon"] / deg)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for j in grid.get((gy + dy, gx + dx), ()):
                    if seen[j]:
                        continue
                    o = dets[j]
                    if same_type and o["type"] != d["type"]:
                        continue
                    if haversine_m(d["lat"], d["lon"], o["lat"], o["lon"]) <= radius_m:
                        seen[j] = True
                        group.append(o)
        clusters.append(group)
    return clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--radius", type=float, default=15.0,
                    help="match_radius_m from the plan")
    ap.add_argument("--verified-only", action="store_true",
                    help="count only detections the verifier accepted")
    ap.add_argument("--same-type", action="store_true", default=True)
    args = ap.parse_args()

    traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                    if d.is_dir() and d.name.upper().startswith("PVS"))

    dets: list[dict] = []
    for t in traces:
        d = load_detections(t, args.side, args.verified_only)
        dets.extend(d)
        print(f"  {t:8} {len(d):5d} detections"
              + ("  (verifier-accepted only)" if args.verified_only else ""))
    if not dets:
        raise SystemExit("no detections found \u2014 run the pipeline first")

    print(f"\n{len(dets)} detections, clustering at {args.radius:g} m, k = {args.k}\n")
    clusters = cluster(dets, args.radius, args.same_type)

    by_owners: Counter = Counter()
    reach = []
    for c in clusters:
        n_tr = len({d["trace"] for d in c})
        by_owners[n_tr] += 1
        if n_tr >= args.k:
            reach.append(c)

    print(f"{len(clusters)} distinct locations after clustering\n")
    print("  distinct traces   locations    share")
    for n in sorted(by_owners):
        share = 100 * by_owners[n] / len(clusters)
        mark = "  <- would confirm" if n >= args.k else ""
        print(f"  {n:>2}              {by_owners[n]:8d}   {share:5.1f}%{mark}")

    n_conf = len(reach)
    det_in_conf = sum(len(c) for c in reach)
    print(f"\n  locations reaching k = {args.k}: {n_conf} "
          f"({100 * n_conf / len(clusters):.1f}% of locations)")
    print(f"  detections inside them:     {det_in_conf} "
          f"({100 * det_in_conf / len(dets):.1f}% of detections)")

    # ---- are the confirmations real, or correlated false positives? ----
    # Consensus only corrects errors that are independent across voters. Every
    # car runs the same detector, so a mistake it makes on one surface it makes
    # everywhere that surface appears. Three cars agreeing on a cobblestone
    # false positive would confirm it. This checks whether that is happening.
    if reach:
        surf = Counter()
        with_verified = 0
        for c in reach:
            s = Counter(d["surface"] for d in c).most_common(1)[0][0]
            surf[s] += 1
            if any(d["verified"] for d in c):
                with_verified += 1
        print("\n  what the confirming locations are on")
        for s, n in surf.most_common():
            print(f"    {s:14} {n:4d}  ({100 * n / len(reach):5.1f}%)")
        print(f"\n  confirming locations the verifier also accepted: "
              f"{with_verified} / {len(reach)} "
              f"({100 * with_verified / len(reach):.1f}%)")

        if not args.verified_only:
            unsupported = len(reach) - with_verified
            print(f"  confirming locations the verifier rejected entirely: "
                  f"{unsupported} ({100 * unsupported / len(reach):.1f}%)")
            if unsupported / len(reach) > 0.5:
                print("\n  WARNING: most locations consensus would confirm, the verifier")
                print("  rejected in every detection. If those are real faults the")
                print("  verifier is too strict; if they are not, consensus is")
                print("  confirming correlated detector errors. Hand-check a sample.")

    dets_by_det = Counter(d["detector"] for d in dets)
    if len(dets_by_det) > 1:
        print("\n  NOTE: detections come from more than one detector:")
        for k_, v in dets_by_det.most_common():
            print(f"    {k_ or '(unrecorded)':32} {v}")
        print("  Rerun detection on every trace with one model before trusting")
        print("  these numbers, or the comparison mixes two generations.")

    out = cfg.DATA / ("consensus_potential_verified.csv" if args.verified_only
                      else "consensus_potential.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "n_detections", "n_distinct_traces", "traces",
                    "lat", "lon", "would_confirm"])
        for i, c in enumerate(clusters):
            tr = sorted({d["trace"] for d in c})
            w.writerow([i, len(c), len(tr), "|".join(tr),
                        round(sum(d["lat"] for d in c) / len(c), 6),
                        round(sum(d["lon"] for d in c) / len(c), 6),
                        int(len(tr) >= args.k)])

    print(f"\nper-location detail -> {out}")
    print("\n" + "=" * 66)
    if n_conf == 0:
        print("No location has detections from k distinct traces. Cars drive the")
        print("same roads but their detectors do not fire at the same places, so")
        print("nothing would ever be confirmed. Widen the radius, lower k, or")
        print("accept that confirmations must be simulated.")
    elif n_conf < 20:
        print(f"Only {n_conf} locations would confirm. Enough to demonstrate the")
        print("mechanism, too few to measure label quality against. Expect wide")
        print("intervals on anything computed from confirmed faults.")
    else:
        print(f"{n_conf} locations would confirm. That is a usable population for")
        print("E12 and for the consensus-filtered calibration in N3.")
    print("=" * 66)
    print("\nThis is an upper bound. The simulator adds timing: a car only")
    print("confirms a fault that is still open when it passes, so the live run")
    print("will produce fewer than this.")


if __name__ == "__main__":
    main()
