"""Repair a confirmed fault and have the cars verify it, for the live demo.

The repair team claims the fix; then m_checkback cars that drive past report
the damage gone, and the fault closes. With the gateway running, the alert
turns amber on claimRepair and disappears on Closed.

    python -m src.car.repair --list
    python -m src.car.repair --fault 31
    python -m src.car.repair --fault 31 --claim-only    (leave it amber)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.car.chain import KEYS, Chain, role_key  # noqa: E402

STATE = ["None", "Pending", "Confirmed", "RepairClaimed", "Disputed",
         "Rejected", "Closed", "Expired"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--claim-only", action="store_true")
    args = ap.parse_args()

    reader = Chain()
    fl = reader.c["FaultLifecycle"]
    if args.list or not args.fault:
        n = fl.functions.faultCount().call()
        live = [(f, STATE[fl.functions.stateOf(f).call()]) for f in range(1, n + 1)]
        live = [x for x in live if x[1] in ("Confirmed", "RepairClaimed")]
        print("faults a repair can act on:" if live else "no Confirmed faults right now")
        for f, s in live:
            print(f"  {f:4}  {s}")
        return

    fid = args.fault
    st = STATE[fl.functions.stateOf(fid).call()]
    if st == "Confirmed":
        rk = role_key("REPAIR")
        if not rk:
            raise SystemExit("no REPAIR_KEY in chain/besu/.env")
        rc = Chain(rk)
        fn = rc.c["FaultLifecycle"].functions.claimRepair(
            fid, hashlib.sha256(f"repair{fid}".encode()).digest())
        why = rc.dry_run(fn)
        if why:
            raise SystemExit(f"claimRepair refused: {why}")
        rc.send(fn)
        print(f"fault {fid}: Confirmed -> {STATE[fl.functions.stateOf(fid).call()]}")
    elif st != "RepairClaimed":
        raise SystemExit(f"fault {fid} is {st}; only Confirmed faults can be repaired")

    if args.claim_only:
        return
    for p in sorted(KEYS.glob("car*.json")):
        if STATE[fl.functions.stateOf(fid).call()] != "RepairClaimed":
            break
        car = Chain(json.loads(p.read_text())["private_key"])
        fn = car.c["FaultLifecycle"].functions.checkBack(
            fid, False, hashlib.sha256(f"cb{fid}{p.stem}".encode()).digest())
        why = car.dry_run(fn)
        if why:
            print(f"  {p.stem} check-back refused: {why}")
            continue
        car.send(fn)
        print(f"  {p.stem} reports the damage gone -> "
              f"{STATE[fl.functions.stateOf(fid).call()]}")


if __name__ == "__main__":
    main()
