"""The keeper: expire Pending faults that nobody contradicted.

Contracts cannot run on a timer, so FaultLifecycle.finalize() waits for someone
to call it. Once a Pending fault is older than T_pending it expires with no
reward and no penalty, and every contributor's stake comes back. Without this,
stakes stay locked forever and cars run out of points: the deadlock the live
simulator hit.

    python -m src.car.keeper              # one pass
    python -m src.car.keeper --loop 30    # every 30 s until Ctrl+C
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.car.chain import KEYS, Chain, admin_key, role_key  # noqa: E402

STATE = ["None", "Pending", "Confirmed", "RepairClaimed", "Disputed",
         "Rejected", "Closed", "Expired"]
FIRST_REPORT_AT = 4        # index in the Fault struct returned by faults(id)


def car_points(ch: Chain) -> list[tuple[str, dict]]:
    import json
    rp = ch.c["RoadPoint"]
    out = []
    for p in sorted(KEYS.glob("car*.json")):
        addr = ch.w3.to_checksum_address(json.loads(p.read_text())["address"])
        vals = {}
        for fn in ("balanceOf", "lockedOf", "freeBalance"):
            try:
                vals[fn] = getattr(rp.functions, fn)(addr).call()
            except Exception:                                  # noqa: BLE001
                vals[fn] = None
        out.append((p.stem, vals))
    return out


def one_pass(ch: Chain) -> tuple[int, int]:
    fl = ch.c["FaultLifecycle"]
    ttl = ch.c["Params"].functions.get("T_pending").call()
    now = ch.w3.eth.get_block("latest")["timestamp"]
    total = fl.functions.faultCount().call()
    expired = waiting = 0
    for fid in range(1, total + 1):
        f = fl.functions.faults(fid).call()
        if STATE[f[3]] != "Pending":
            continue
        ready = f[FIRST_REPORT_AT] + ttl
        if now < ready:
            waiting += 1
            continue
        fn = fl.functions.finalize(fid)
        why = ch.dry_run(fn)
        if why:
            print(f"  fault {fid}: refused {why}")
            continue
        r = ch.send(fn)
        expired += 1
        print(f"  fault {fid}: Pending -> {STATE[fl.functions.stateOf(fid).call()]}"
              f"  gas {r['gas_used']}")
    print(f"T_pending={ttl}s  expired {expired}, still waiting {waiting}")
    if waiting and ttl > 3600:
        print(f"  T_pending is {ttl // 3600} h, so in a short simulation nothing "
              f"expires.\n  Lower it by vote:  python -m src.car.govern "
              f"--set T_pending=60")
    return expired, waiting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes")
    args = ap.parse_args()

    ch = Chain(role_key("KEEPER") or admin_key())
    print(f"keeper {ch.address}\n")
    while True:
        one_pass(ch)
        print("\ncar points")
        for name, v in car_points(ch):
            print(f"  {name}  balance {v['balanceOf']}  locked {v['lockedOf']}  "
                  f"free {v['freeBalance']}")
        if not args.loop:
            break
        time.sleep(args.loop)
        print()


if __name__ == "__main__":
    main()
