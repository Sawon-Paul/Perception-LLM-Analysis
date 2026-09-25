"""Simulated cars driving real GPS traces and transacting on the chain.

Plan section 8. Each car replays one PVS trace on a shared clock. When its
detector fired at a place, it either opens a new fault or confirms one that
already exists; when it passes a fault it did not report, it files a free
check-back saying whether the damage is still there.

Two things decide correctness here.

**Matching happens off-chain.** The car reads the active faults in its geohash
cell, and if one of the same type sits within match_radius_m it calls confirm
instead of report. The contract only checks the cell prefix and the
one-per-owner rule, which keeps gas low and the contract simple.

**One owner, one vote.** Each car gets its own owner id, so the distinct-owner
rule actually binds. Giving several cars the same owner is how the collusion
experiment is run, not the default.

    python -m src.car.simulate --register            # one time
    python -m src.car.simulate --traces "PVS 1" "PVS 2" "PVS 3"
    python -m src.car.simulate --traces "PVS 1" "PVS 2" "PVS 3" --repair
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.analysis.trace_overlap import geohash_encode  # noqa: E402
from src.car.chain import (Chain, admin_key, load_or_create_key,  # noqa: E402
                           role_key)

RESULTS = cfg.ROOT / "results"
FAULT_TYPES = {"pothole": 0, "alligator cracking": 1, "lateral cracking": 2,
               "longitudinal cracking": 3}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def detection_source(trace: str, side: str = "left") -> str:
    """Which events file a trace's detections come from.

    Traces must share a detector. PVS 2 held 27 detections from the fine-tuned
    model while others held hundreds from the original, and cars running
    different detectors fire in different places and never meet.
    """
    tag = cfg.slug(trace)
    for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
        if (cfg.EVENTS / f"events_{tag}_{side}{suffix}.jsonl").exists():
            return suffix or "(raw)"
    return "(none)"


def load_detections(trace: str, side: str = "left",
                    verified_only: bool = False) -> list[dict]:
    """What this car's detector reported, in time order."""
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
            if e["stream5_model"].get("stream") != "pothole":
                continue
            if e.get("lat") is None or e.get("lon") is None:
                continue
            v = bool((e.get("phase2") or {}).get("verified"))
            if verified_only and not v:
                continue
            out.append({
                "id": e["detection_id"], "lat": float(e["lat"]),
                "lon": float(e["lon"]), "t": float(e.get("timestamp") or 0),
                "conf": float(e["stream5_model"].get("conf") or 0.0),
                "type": int(e["stream5_model"].get("class_id", 0)),
                "verified": v,
                "severity": int(((e.get("phase2") or {}).get("severity") or 1)),
            })
        out.sort(key=lambda d: d["t"])
        return out
    return []


def evidence_hash(det: dict) -> bytes:
    import hashlib
    return hashlib.sha256(
        json.dumps({k: det[k] for k in ("id", "lat", "lon", "type")},
                   sort_keys=True).encode()).digest()


def choose_action(det: dict, active: list[tuple[int, int]],
                  known: list[dict], match_radius: float) -> int | None:
    """Return the fault id to confirm, or None to open a new one.

    active is [(fault_id, fault_type)] from activeFaultsIn for this cell;
    known is what this simulation has recorded about where each fault sits.

    Two rules, and both matter:
      - the type must match, so a pothole never confirms a crack
      - if we know where the fault is, it must be within match_radius; a fault
        at the far end of a 153 m cell is a different fault

    A fault whose position we do not know is accepted on cell and type alone,
    which is what a real car would do with only the chain to go on.
    """
    for fid, ftype in active:
        if ftype != det["type"]:
            continue
        rec = next((r for r in known if r["id"] == fid), None)
        if rec is not None:
            if haversine_m(det["lat"], det["lon"],
                           rec["lat"], rec["lon"]) > match_radius:
                continue
        return fid
    return None


def register_cars(traces: list[str], same_owner: list[str] | None = None) -> None:
    """Register one car per trace, each with its own owner unless told otherwise."""
    import hashlib
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
        # cars named in --same-owner share one owner id, which is how the
        # collusion experiment is set up; otherwise every car is its own owner
        owner_src = "colluding_owner" if trace in same_owner else f"owner_{i}"
        owner = hashlib.sha256(owner_src.encode()).digest()
        vehicle = hashlib.sha256(f"vehicle_{trace}".encode()).digest()
        r = ch.send(reg.functions.registerCar(addr, owner, vehicle))
        print(f"  {trace:8} {addr}  owner={owner_src:16} "
              f"gas {r['gas_used']:>7}  {r['latency_s']:.2f}s")
    print("\nkeys in config/keys/ — keep them out of git")


def run(traces: list[str], side: str, verified_only: bool, max_per_car: int,
        do_repair: bool, speed: float) -> None:
    ch_admin = Chain(admin_key())
    params = ch_admin.c["Params"]
    match_radius = params.functions.get("match_radius_m").call()
    checkback_radius = params.functions.get("checkback_radius_m").call()
    k_confirm = params.functions.get("k_confirm").call()
    print(f"k_confirm={k_confirm}  match_radius={match_radius}m  "
          f"checkback_radius={checkback_radius}m\n")

    cars = []
    for i, trace in enumerate(traces):
        key = load_or_create_key(i)
        ch = Chain(key["private_key"])
        if not ch.c["Registry"].functions.isRegistered(
                ch.w3.to_checksum_address(key["address"])).call():
            raise SystemExit(f"car {i} ({trace}) is not registered — "
                             f"run with --register first")
        dets = load_detections(trace, side, verified_only)
        if max_per_car:
            dets = dets[:max_per_car]
        cars.append({"trace": trace, "chain": ch, "dets": dets, "i": i,
                     "src": detection_source(trace, side)})
        print(f"  car {i}: {trace:8} {len(dets):5d} detections  "
              f"from {cars[-1]['src']}")

    # Interleave by detection timestamp so cars act in the order they actually
    # drove, rather than one car finishing its whole trace before the next
    # starts. Confirmation depends on cars arriving at a place separately.
    srcs = {c["src"] for c in cars}
    if len(srcs) > 1:
        print(f"\n  WARNING: cars are driving different detectors' output: "
              f"{sorted(srcs)}.")
        print("  Different detectors fire in different places, so the cars will")
        print("  rarely meet and nothing will reach k confirmations. Rerun")
        print("  detection so every trace uses the same model.")

    queue = []
    for c in cars:
        for d in c["dets"]:
            queue.append((d["t"], c, d))
    queue.sort(key=lambda x: x[0])
    print(f"\n{len(queue)} detections interleaved by timestamp\n")

    rows = []
    refusals: dict[str, int] = defaultdict(int)
    reported: dict[str, list[dict]] = defaultdict(list)   # cell -> faults we know
    t0 = time.time()

    for n, (_, car, det) in enumerate(queue, 1):
        ch = car["chain"]
        fl = ch.c["FaultLifecycle"]
        cell8 = geohash_encode(det["lat"], det["lon"], 8)
        cell7 = cell8[:7]

        # what the chain already knows about here
        try:
            ids = fl.functions.activeFaultsIn(cell7.encode()).call()
        except Exception as e:
            print(f"  [{n}] activeFaultsIn failed: {e}")
            continue

        active = [(fid, fl.functions.faults(fid).call()[0]) for fid in ids]
        match = choose_action(det, active, reported[cell7], match_radius)

        action, res, fid_out = None, None, None
        # Ask the node whether the call would revert before spending a
        # transaction on it. The reason is recorded either way, so a refusal
        # by the contract is data rather than a silent failure.
        pending = (fl.functions.report(
                       det["type"], cell8.encode(), det["severity"],
                       int(det["conf"] * 1000), 900 if det["verified"] else 0,
                       evidence_hash(det))
                   if match is None else
                   fl.functions.confirm(match, det["severity"],
                                        evidence_hash(det)))
        why = ch.dry_run(pending)
        if why is not None:
            rows.append({"n": n, "car": car["i"], "trace": car["trace"],
                         "detection_id": det["id"], "action": "refused",
                         "reason": why[:120], "fault_id": match, "cell": cell8,
                         "tx_hash": "", "status": 0, "gas_used": 0, "block": 0,
                         "sent_at": 0, "mined_at": 0, "latency_s": 0})
            # Keep the message itself. Splitting on "(" to get a short label
            # produced an empty string for every custom error, so the tally
            # said "212" with no reason beside it.
            label = " ".join(why.split())[:70] or "(no reason returned)"
            refusals[label] += 1
            continue

        try:
            if match is None:
                r = ch.send(fl.functions.report(
                    det["type"], cell8.encode(), det["severity"],
                    int(det["conf"] * 1000), 900 if det["verified"] else 0,
                    evidence_hash(det)))
                action = "report"
                ev = ch.events("FaultLifecycle", "FaultReported", r["tx_hash"])
                fid_out = ev[0]["args"]["faultId"] if ev else None
                if fid_out:
                    reported[cell7].append(
                        {"id": fid_out, "lat": det["lat"], "lon": det["lon"]})
                res = r
            else:
                r = ch.send(fl.functions.confirm(
                    match, det["severity"], evidence_hash(det)))
                action, fid_out, res = "confirm", match, r
        except Exception as e:
            msg = str(e)
            # the contract refusing is a result, not a crash: one owner cannot
            # confirm twice, and that is the rule the thesis is testing
            action = "rejected"
            res = {"tx_hash": "", "status": 0, "gas_used": 0, "block": 0,
                   "sent_at": 0, "mined_at": 0, "latency_s": 0}
            print(f"  [{n}] car {car['i']} {action}: {msg[:90]}")

        rows.append({"n": n, "car": car["i"], "trace": car["trace"],
                     "detection_id": det["id"], "action": action,
                     "reason": "", "fault_id": fid_out, "cell": cell8, **res})
        if n % 10 == 0 or n == len(queue):
            rep = sum(1 for r in rows if r["action"] == "report")
            con = sum(1 for r in rows if r["action"] == "confirm")
            ref = sum(1 for r in rows if r["action"] == "refused")
            print(f"  {n}/{len(queue)}  {rep} reports, {con} confirms, "
                  f"{ref} refused, {time.time() - t0:.0f}s")

    # ---- what state did the faults reach? ----
    fl = cars[0]["chain"].c["FaultLifecycle"]
    total = fl.functions.faultCount().call()
    states = defaultdict(int)
    STATE = ["None", "Pending", "Confirmed", "RepairClaimed", "Disputed",
             "Rejected", "Closed", "Expired"]
    confirmed_ids = []
    for fid in range(1, total + 1):
        s = fl.functions.stateOf(fid).call()
        states[STATE[s]] += 1
        if STATE[s] == "Confirmed":
            confirmed_ids.append(fid)

    if refusals:
        print("\ncontract refusals (the rules doing their job)")
        for why, c in sorted(refusals.items(), key=lambda x: -x[1]):
            print(f"  {c:4}  {why}")

    print(f"\n{total} faults on chain")
    for s, c in sorted(states.items(), key=lambda x: -x[1]):
        print(f"  {s:14} {c}")

    # ---- optional: drive one fault through repair to Closed ----
    if do_repair and confirmed_ids:
        rk = role_key("REPAIR")
        if not rk:
            print("\nno REPAIR key in chain/besu/.env — run node scripts/roles.js")
        else:
            fid = confirmed_ids[0]
            print(f"\ndriving fault {fid} through repair")
            rc = Chain(rk)
            r = rc.send(rc.c["FaultLifecycle"].functions.claimRepair(
                fid, evidence_hash({"id": f"repair{fid}", "lat": 0, "lon": 0,
                                    "type": 0})))
            print(f"  claimRepair  gas {r['gas_used']}  -> "
                  f"{STATE[fl.functions.stateOf(fid).call()]}")
            m = cars[0]["chain"].c["Params"].functions.get("m_checkback").call()
            for c in cars[:m]:
                try:
                    r = c["chain"].send(c["chain"].c["FaultLifecycle"]
                                        .functions.checkBack(fid, False,
                                                             evidence_hash(
                                                                 {"id": "cb",
                                                                  "lat": 0,
                                                                  "lon": 0,
                                                                  "type": 0})))
                    print(f"  checkBack absent by car {c['i']}  gas {r['gas_used']}"
                          f"  -> {STATE[fl.functions.stateOf(fid).call()]}")
                except Exception as e:
                    print(f"  checkBack by car {c['i']} refused: {str(e)[:80]}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "car_transactions.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    rep = sum(1 for r in rows if r["action"] == "report")
    con = sum(1 for r in rows if r["action"] == "confirm")
    ref = sum(1 for r in rows if r["action"] == "refused")
    print(f"\n  {rep} reports, {con} confirms, {ref} refused "
          f"(of {len(rows)} detections)")
    if states.get("Confirmed", 0) == 0 and states.get("Closed", 0) == 0:
        print(f"\n  Nothing reached Confirmed. A fault needs {k_confirm} distinct")
        print("  OWNERS, and each car is one owner, so it needs all three cars to")
        print("  pass the same spot within match_radius. Check the refusal reasons")
        print("  above: if they are mostly OwnerAlreadyContributed, the cars are")
        print("  re-detecting their own faults rather than each other's.")

    sent = [r for r in rows if r["action"] in ("report", "confirm")]
    if sent:
        lat = sorted(r["latency_s"] for r in sent)
        gas = sorted(r["gas_used"] for r in sent)
        print(f"\n  {len(sent)} transactions")
        print(f"  latency  median {lat[len(lat) // 2]:.2f}s   "
              f"min {lat[0]:.2f}s   max {lat[-1]:.2f}s")
        print(f"  gas      median {gas[len(gas) // 2]}   "
              f"report vs confirm differ; see the CSV")
    print(f"\n-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--traces", nargs="*", default=["PVS 1", "PVS 2", "PVS 3"])
    ap.add_argument("--same-owner", nargs="*", default=None,
                    help="traces that share one owner, for the collusion test")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--verified-only", action="store_true",
                    help="report only detections the verifier accepted")
    ap.add_argument("--max-per-car", type=int, default=0,
                    help="cap detections per car; 0 for all. A small cap takes "
                         "the FIRST N, which samples the start of each trace "
                         "where the cars may never have driven together")
    ap.add_argument("--repair", action="store_true",
                    help="drive one confirmed fault through repair to Closed")
    ap.add_argument("--speed", type=float, default=0.0)
    args = ap.parse_args()

    if args.register:
        register_cars(args.traces, args.same_owner)
        return
    run(args.traces, args.side, args.verified_only, args.max_per_car,
        args.repair, args.speed)


if __name__ == "__main__":
    main()
