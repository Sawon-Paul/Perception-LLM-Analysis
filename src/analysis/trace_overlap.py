"""Can consensus actually happen in this dataset?

The confirmation mechanism needs k distinct owners to pass the same place. The
contract matches on a 7-character geohash cell, so this asks the literal
question: how many cells are visited by 2, 3, 4 or more distinct traces?

Two answers matter, and they are not the same.

  route overlap    how much of the driven road is shared at all
  fault overlap    whether the places where faults were actually detected sit
                   in shared cells

The second is the one that decides the project. A dataset can have plenty of
shared motorway and still have every pothole on a street only one car drove
down, in which case no fault ever reaches k confirmations and the whole
consensus layer has to be simulated with replayed traces.

Run this before building anything that depends on confirmation.

    python -m src.analysis.trace_overlap
    python -m src.analysis.trace_overlap --k 3 --precision 7
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"

# Approximate cell size at the equator, for reporting only.
CELL_SIZE = {5: "4.9 km x 4.9 km", 6: "1.2 km x 610 m", 7: "153 m x 153 m",
             8: "38 m x 19 m", 9: "4.8 m x 4.8 m"}


def geohash_encode(lat: float, lon: float, precision: int = 7) -> str:
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    out, bit, ch, even = [], 0, 0, True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            # >= not >, so a point exactly on a cell boundary lands in the upper
            # half, matching the standard implementations. Only matters at 0,0
            # and the meridians, but a geohash that disagrees with every library
            # is a trap for anyone comparing results later.
            if lon >= mid:
                ch = (ch << 1) | 1
                lon_lo = mid
            else:
                ch <<= 1
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch = (ch << 1) | 1
                lat_lo = mid
            else:
                ch <<= 1
                lat_hi = mid
        even = not even
        bit += 1
        if bit == 5:
            out.append(_B32[ch])
            bit, ch = 0, 0
    return "".join(out)


def trace_cells(trace: str, side: str, precision: int,
                min_speed_kmh: float) -> set[str]:
    """Cells a trace drove through, ignoring time spent parked."""
    path = cfg.pvs_dir(trace) / f"dataset_gps_mpu_{side}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=lambda c: c in
                     ("latitude", "longitude", "speed"))
    df = df.dropna(subset=["latitude", "longitude"])
    if "speed" in df.columns and min_speed_kmh > 0:
        df = df[df["speed"] * 3.6 >= min_speed_kmh]
    # 100 Hz GPS repeats the same fix many times; thinning costs nothing and
    # makes this run in seconds rather than minutes.
    df = df.iloc[::25]
    return {geohash_encode(float(a), float(b), precision)
            for a, b in zip(df["latitude"], df["longitude"])}


def fault_cells(trace: str, side: str, precision: int) -> list[tuple[str, str]]:
    """(detection_id, cell) for every detection this trace produced."""
    tag = cfg.slug(trace)
    for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
        p = cfg.EVENTS / f"events_{tag}_{side}{suffix}.jsonl"
        if p.exists():
            import json
            out = []
            for line in p.open(encoding="utf-8"):
                if not line.strip():
                    continue
                e = json.loads(line)
                lat, lon = e.get("lat"), e.get("lon")
                if lat is None or lon is None:
                    continue
                if e.get("stream5_model", {}).get("stream") != "pothole":
                    continue
                out.append((e["detection_id"],
                            geohash_encode(float(lat), float(lon), precision)))
            return out
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--precision", type=int, default=7,
                    help="geohash length; 7 is what FaultLifecycle matches on")
    ap.add_argument("--k", type=int, default=3, help="distinct owners needed")
    ap.add_argument("--min-speed", type=float, default=2.0, help="km/h")
    args = ap.parse_args()

    traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                    if d.is_dir() and d.name.upper().startswith("PVS"))
    if not traces:
        raise SystemExit(f"no PVS folders under {cfg.DATASET_DIR}")

    print(f"geohash precision {args.precision} "
          f"(~{CELL_SIZE.get(args.precision, 'unknown')}), k = {args.k}\n")

    # ------------------------------------------------------ route overlap
    cells: dict[str, set[str]] = {}
    for t in traces:
        c = trace_cells(t, args.side, args.precision, args.min_speed)
        cells[t] = c
        print(f"  {t:8} {len(c):6d} cells driven")

    visits: Counter = Counter()
    for c in cells.values():
        visits.update(c)

    by_count: Counter = Counter(visits.values())
    total = len(visits)
    print(f"\n{total} distinct cells across {len(traces)} traces\n")
    print("  visited by   cells    share")
    reachable = 0
    for n in sorted(by_count):
        share = 100 * by_count[n] / total
        mark = "  <- k reachable" if n >= args.k else ""
        if n >= args.k:
            reachable += by_count[n]
        print(f"  {n:>2} traces  {by_count[n]:7d}   {share:5.1f}%{mark}")
    print(f"\n  cells reachable by {args.k}+ distinct traces: "
          f"{reachable} ({100 * reachable / total:.1f}%)")

    # ------------------------------------------------------ fault overlap
    print("\nfaults in shared cells (this is the one that decides it)\n")
    rows = []
    tot_f = tot_ok = 0
    for t in traces:
        fl = fault_cells(t, args.side, args.precision)
        if not fl:
            print(f"  {t:8} no events file \u2014 run the pipeline for this trace")
            continue
        others = [x for x in traces if x != t]
        ok = 0
        for det_id, cell in fl:
            n_other = sum(1 for o in others if cell in cells.get(o, ()))
            # the reporting car counts as one owner, so k-1 others are needed
            confirmable = n_other >= args.k - 1
            ok += confirmable
            rows.append({"trace": t, "detection_id": det_id, "cell": cell,
                         "other_traces_in_cell": n_other,
                         "confirmable": int(confirmable)})
        tot_f += len(fl)
        tot_ok += ok
        print(f"  {t:8} {ok:4d} / {len(fl):4d} detections in a cell "
              f"{args.k - 1}+ other traces also drove  "
              f"({100 * ok / max(len(fl), 1):5.1f}%)")

    out = cfg.DATA / "trace_overlap.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\n  overall: {tot_ok} / {tot_f} detections could reach k = {args.k}")
    if tot_f:
        pct = 100 * tot_ok / tot_f
        print(f"           {pct:.1f}%\n")
        print("=" * 66)
        if pct >= 30:
            print("GO. Enough detections sit where other traces drove, so real")
            print("consensus is reachable. Report this percentage in the thesis as")
            print("the fraction of faults eligible for confirmation.")
        elif pct >= 5:
            print("MARGINAL. Real consensus is possible but thin. Expect few")
            print("confirmed faults. Plan to supplement with replayed traces and")
            print("say clearly in the thesis which results rest on which.")
        else:
            print("NO-GO for unsimulated consensus. Almost no detection sits where")
            print("another trace drove, so k distinct owners is unreachable from")
            print("this data alone. Every consensus result will come from replayed")
            print("traces with time offsets. That is workable, but it must be")
            print("labelled synthetic everywhere, and the limitation stated plainly.")
        print("=" * 66)
    if rows:
        print(f"\nper-detection detail -> {out}")


if __name__ == "__main__":
    main()
