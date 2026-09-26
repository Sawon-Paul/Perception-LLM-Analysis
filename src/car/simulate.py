"""Simulated cars driving real PVS traces and transacting on the chain.

Plan section 8. Each car replays one trace. Where its detector fired it either
opens a new fault or confirms one already on the chain, and the contract
decides whether that is allowed.

Three things decide whether the result means anything.

**Cars drive at the same time.** Detections are interleaved by how far along
its own trace each car is, not by wall-clock timestamp. The traces were
recorded on different days, so sorting by timestamp made one car finish its
whole drive before the next started, and by then its stake was spent.

**Cars share a route and a detector.** PVS 1, 4 and 7 are three vehicles on the
same route running the same original detector. Mixing route groups or detectors
means cars fire in different places and never meet.

**The chain starts clean.** Stakes stay locked until a fault closes. Faults
left over from an earlier run keep every car's points locked, and a run on top
of them sends nothing. The script refuses to start on a used chain unless told.

    python -m src.car.simulate --register --traces "PVS 1" "PVS 4" "PVS 7"
    python -m src.car.simulate --traces "PVS 1" "PVS 4" "PVS 7" --repair
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.analysis.trace_overlap import geohash_encode  # noqa: E402
from src.car.chain import (Chain, admin_key, load_or_create_key,  # noqa: E402
                           role_key)

RESULTS = cfg.ROOT / "results"
STATE = ["None", "Pending", "Confirmed", "RepairClaimed", "Disputed",
         "Rejected", "Closed", "Expired"]
_FRAME_RE = re.compile(r"frame_(\d+)")


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def detection_source(trace: str, side: str = "left") -> str:
    tag = cfg.slug(trace)
    for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
        if (cfg.EVENTS / f"events_{tag}_{side}{suffix}.jsonl").exists():
            return suffix or "(raw)"
    return "(none)"


def _clock(e: dict, i: int) -> float:
    """Something that increases along the drive: timestamp, else frame number."""
    ts = e.get("timestamp")
    if ts not in (None, "", 0):
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass
    fp = (e.get("stream1_image") or {}).get("frame_path", "")
    m = _FRAME_RE.search(str(fp))
    return float(m.group(1)) if m else float(i)


def add_progress(dets: list[dict]) -> None:
    """Attach 0..1 progress along the trace, used to interleave cars."""
    if not dets:
        return
    lo = min(d["clock"] for d in dets)
    hi = max(d["clock"] for d in dets)
    span = (hi - lo) or 1.0
    for d in dets:
        d["progress"] = (d["clock"] - lo) / span


def load_detections(trace: str, side: str = "left",
                    verified_only: bool = False) -> list[dict]:
    tag = cfg.slug(trace)
    for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
        p = cfg.EVENTS / f"events_{tag}_{side}{suffix}.jsonl"
        if not p.exists():
            continue
        out = []
        for i, line in enumerate(p.open(encoding="utf-8")):
            if not line.strip():
                continue
            e = json.loads(line)
            if (e.get("stream5_model") or {}).get("stream") != "pothole":
                continue
            if e.get("lat") is None or e.get("lon") is None:
                continue
            p2 = e.get("phase2") or {}
            v = bool(p2.get("verified"))
            if verified_only and not v:
                continue
            out.append({
                "id": e["detection_id"], "lat": float(e["lat"]),
                "lon": float(e["lon"]), "clock": _clock(e, i),
                "conf": float(e["stream5_model"].get("conf") or 0.0),
                "type": int(e["stream5_model"].get("class_id", 0)),
                "verified": v,
                "severity": max(1, min(5, int(p2.get("severity") or 1))),
            })
        out.sort(key=lambda d: d["clock"])
        add_progress(out)
        return out
    return []


def interleave(cars: list[dict], mode: str = "progress") -> list[tuple]:
    """Merge every car's detections into one queue.

    progress  cars drive their routes simultaneously (default)
    absolute  original wall-clock order, i.e. one car after another
    """
    key = "progress" if mode == "progress" else "clock"
    q = [(d[key], c["i"], c, d) for c in cars for d in c["dets"]]
    q.sort(key=lambda x: (x[0], x[1]))
    return [(c, d) for _, _, c, d in q]


def evidence_hash(det: dict) -> bytes:
    return hashlib.sha256(json.dumps(
        {k: det[k] for k in ("id", "lat", "lon", "type")},
        sort_keys=True).encode()).digest()


def choose_action(det: dict, active: list[tuple[int, int]],
                  known: list[dict], match_radius: float) -> int | None:
    """Fault id to confirm, or None to open a new one.

    The type must match, and if we know where the fault sits it must be within
    match_radius. A fault whose position we do not know is accepted on cell and
    type alone, which is all a real car would have from the chain.
    """
    for fid, ftype in active:
        if ftype != det["type"]:
            continue
        rec = next((r for r in known if r["id"] == fid), None)
        if rec is not None and haversine_m(det["lat"], det["lon"],
                                           rec["lat"], rec["lon"]) > match_radius:
            continue
        return fid
    return None


def register_cars(traces: list[str], same_owner: list[str] | None = None) -> None:
    ch = Chain(admin_key())
    reg = ch.c["Registry"]
    same_owner = same_owner or []
    print(f"registering {len(traces)} cars as {ch.address}\n")
    for i, trace in enumerate(traces):
        key = load_or_create_key(i)
        addr = ch.w3.to_checksum_address(key["address"])
        if reg.functions.isRegistered(addr).call():
            print(f"  {trace:8} {addr}  already registered")
            continue
        owner_src = "colluding_owner" if trace in same_owner else f"owner_{i}"
        owner = hashlib.sha256(owner_src.encode()).digest()
        vehicle = hashlib.sha256(f"vehicle_{trace}".encode()).digest()
        fn = reg.functions.registerCar(addr, owner, vehicle)
        why = ch.dry_run(fn)
        if why:
            print(f"  {trace:8} {addr}  REFUSED: {why}")
            continue
        r = ch.send(fn)
        print(f"  {trace:8} {addr}  owner={owner_src:16} "
              f"gas {r['gas_used']:>7}  {r['latency_s']:.2f}s")
    print("\nkeys in config/keys/ — keep them out of git")


def car_balances(ch: Chain, addr: str) -> dict:
    rp = ch.c["RoadPoint"]
    out = {}
    for fn in ("balanceOf", "lockedOf", "freeBalance"):
        try:
            out[fn] = getattr(rp.functions, fn)(addr).call()
        except Exception:                                       # noqa: BLE001
            out[fn] = None
    return out


def run(args) -> None:
    ch_admin = Chain(admin_key())
    params = ch_admin.c["Params"]
    fl0 = ch_admin.c["FaultLifecycle"]
    match_radius = params.functions.get("match_radius_m").call()
    k_confirm = params.functions.get("k_confirm").call()
    print(f"k_confirm={k_confirm}  match_radius={match_radius}m  "
          f"interleave={args.interleave}\n")

    existing = fl0.functions.faultCount().call()
    if existing and not args.allow_existing:
        raise SystemExit(
            f"The chain already holds {existing} faults from an earlier run.\n"
            f"Their stakes are still locked, so cars would start with no free\n"
            f"points and results would mix two runs. Reset it first:\n\n"
            f"  cd chain\n"
            f"  docker compose -f besu/docker-compose.yml down -v\n"
            f"  docker compose -f besu/docker-compose.yml up -d\n"
            f"  (wait for peers 3 of 3 in node besu/verify.js)\n"
            f"  npx hardhat run scripts/deploy.js --network besu\n"
            f"  cd ..\n"
            f"  python -m src.car.simulate --register --traces ...\n\n"
            f"Or pass --allow-existing if mixing runs is intended.")

    cars = []
    for i, trace in enumerate(args.traces):
        key = load_or_create_key(i)
        ch = Chain(key["private_key"])
        addr = ch.w3.to_checksum_address(key["address"])
        if not ch.c["Registry"].functions.isRegistered(addr).call():
            raise SystemExit(f"car {i} ({trace}) is not registered — "
                             f"run with --register --traces first")
        dets = load_detections(trace, args.side, args.verified_only)
        if args.max_per_car:
            dets = dets[:args.max_per_car]
        src = detection_source(trace, args.side)
        cars.append({"trace": trace, "chain": ch, "dets": dets, "i": i,
                     "addr": addr, "src": src})
        b = car_balances(ch, addr)
        print(f"  car {i}: {trace:8} {len(dets):5d} detections from {src:18} "
              f"free points {b['freeBalance']}")

    if len({c["src"] for c in cars}) > 1:
        print("\n  WARNING: cars are driving different detectors' output. They will")
        print("  fire in different places and rarely reach k confirmations.")

    queue = interleave(cars, args.interleave)
    print(f"\n{len(queue)} detections queued\n")

    rows: list[dict] = []
    refusals: Counter = Counter()
    reported: dict[str, list[dict]] = defaultdict(list)
    t0 = time.time()

    for n, (car, det) in enumerate(queue, 1):
        ch = car["chain"]
        fl = ch.c["FaultLifecycle"]
        cell8 = geohash_encode(det["lat"], det["lon"], 8)
        cell7 = cell8[:7]
        try:
            ids = fl.functions.activeFaultsIn(cell7.encode()).call()
            active = [(fid, fl.functions.faults(fid).call()[0]) for fid in ids]
        except Exception as e:                                  # noqa: BLE001
            print(f"  [{n}] reading faults failed: {e}")
            continue
        match = choose_action(det, active, reported[cell7], match_radius)

        fn = (fl.functions.report(det["type"], cell8.encode(), det["severity"],
                                  int(det["conf"] * 1000),
                                  900 if det["verified"] else 0,
                                  evidence_hash(det))
              if match is None else
              fl.functions.confirm(match, det["severity"], evidence_hash(det)))
        base = {"n": n, "car": car["i"], "trace": car["trace"],
                "detection_id": det["id"], "cell": cell8,
                "progress": round(det["progress"], 4)}

        why = ch.dry_run(fn)
        if why is not None:
            refusals[why.split("(")[0]] += 1
            rows.append({**base, "action": "refused", "reason": why,
                         "fault_id": match or "", "tx_hash": "", "status": 0,
                         "gas_used": 0, "block": 0, "latency_s": 0})
        else:
            try:
                r = ch.send(fn)
                if match is None:
                    ev = ch.events("FaultLifecycle", "FaultReported", r["tx_hash"])
                    fid = ev[0]["args"]["faultId"] if ev else ""
                    if fid:
                        reported[cell7].append(
                            {"id": fid, "lat": det["lat"], "lon": det["lon"]})
                    action = "report"
                else:
                    fid, action = match, "confirm"
                rows.append({**base, "action": action, "reason": "",
                             "fault_id": fid, "tx_hash": r["tx_hash"],
                             "status": r["status"], "gas_used": r["gas_used"],
                             "block": r["block"], "latency_s": r["latency_s"]})
            except Exception as e:                              # noqa: BLE001
                refusals["sent but failed"] += 1
                rows.append({**base, "action": "failed", "reason": str(e)[:120],
                             "fault_id": match or "", "tx_hash": "", "status": 0,
                             "gas_used": 0, "block": 0, "latency_s": 0})

        if n % 50 == 0 or n == len(queue):
            c = Counter(r["action"] for r in rows)
            print(f"  {n}/{len(queue)}  {c['report']} reports, {c['confirm']} "
                  f"confirms, {c['refused']} refused   {time.time() - t0:.0f}s")

    # ---------------- what the chain ended up holding ----------------
    total = fl0.functions.faultCount().call()
    states = Counter(STATE[fl0.functions.stateOf(f).call()]
                     for f in range(1, total + 1))
    confirmed = [f for f in range(1, total + 1)
                 if STATE[fl0.functions.stateOf(f).call()] == "Confirmed"]

    if refusals:
        print("\ncontract refusals")
        for why, c in refusals.most_common():
            print(f"  {c:5}  {why}")

    print(f"\n{total} faults on chain")
    for s, c in states.most_common():
        print(f"  {s:14} {c}")

    print("\ncar points at the end (stake stays locked until a fault closes)")
    for c in cars:
        b = car_balances(c["chain"], c["addr"])
        print(f"  car {c['i']} {c['trace']:8} balance {b['balanceOf']}  "
              f"locked {b['lockedOf']}  free {b['freeBalance']}")

    # ---------------- optional: one fault all the way to Closed -------------
    if args.repair and confirmed:
        rk = role_key("REPAIR")
        if not rk:
            print("\nno REPAIR key in chain/besu/.env — run node scripts/roles.js")
        else:
            fid = confirmed[0]
            print(f"\ndriving fault {fid} through repair")
            rc = Chain(rk)
            fn = rc.c["FaultLifecycle"].functions.claimRepair(
                fid, hashlib.sha256(f"repair{fid}".encode()).digest())
            why = rc.dry_run(fn)
            if why:
                print(f"  claimRepair refused: {why}")
            else:
                r = rc.send(fn)
                print(f"  claimRepair  gas {r['gas_used']}  -> "
                      f"{STATE[fl0.functions.stateOf(fid).call()]}")
                for c in cars:
                    f2 = c["chain"].c["FaultLifecycle"].functions.checkBack(
                        fid, False, hashlib.sha256(f"cb{fid}{c['i']}".encode()).digest())
                    why = c["chain"].dry_run(f2)
                    if why:
                        print(f"  checkBack by car {c['i']} refused: {why}")
                        continue
                    r = c["chain"].send(f2)
                    print(f"  checkBack (damage gone) by car {c['i']}  "
                          f"gas {r['gas_used']}  -> "
                          f"{STATE[fl0.functions.stateOf(fid).call()]}")
    elif args.repair:
        print("\n--repair skipped: nothing reached Confirmed")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "car_transactions.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    c = Counter(r["action"] for r in rows)
    print(f"\n  {c['report']} reports, {c['confirm']} confirms, "
          f"{c['refused']} refused (of {len(rows)} detections)")
    sent = [r for r in rows if r["action"] in ("report", "confirm")]
    if sent:
        lat = sorted(r["latency_s"] for r in sent)
        for a in ("report", "confirm"):
            g = sorted(r["gas_used"] for r in sent if r["action"] == a)
            if g:
                print(f"  {a:8} gas median {g[len(g) // 2]}")
        print(f"  latency  median {lat[len(lat) // 2]:.2f}s  "
              f"min {lat[0]:.2f}s  max {lat[-1]:.2f}s")
    if not confirmed and not states.get("Closed"):
        print(f"\n  Nothing reached Confirmed: no fault collected {k_confirm} "
              f"distinct owners.")
        if refusals.get("InsufficientFree"):
            print("  Cars ran out of free points: each report or confirm locks a")
            print("  stake until the fault closes, so a car can only back a few")
            print("  unresolved faults at once.")
    print(f"\n-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--traces", nargs="*", default=["PVS 1", "PVS 4", "PVS 7"])
    ap.add_argument("--same-owner", nargs="*", default=None,
                    help="traces sharing one owner, for the collusion test")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--verified-only", action="store_true",
                    help="only detections the verifier accepted")
    ap.add_argument("--max-per-car", type=int, default=0)
    ap.add_argument("--interleave", default="progress",
                    choices=["progress", "absolute"])
    ap.add_argument("--allow-existing", action="store_true",
                    help="run on a chain that already holds faults")
    ap.add_argument("--repair", action="store_true",
                    help="drive the first confirmed fault through repair")
    args = ap.parse_args()

    if args.register:
        register_cars(args.traces, args.same_owner)
    else:
        run(args)


if __name__ == "__main__":
    main()
