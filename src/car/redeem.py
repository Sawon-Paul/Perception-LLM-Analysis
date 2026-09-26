"""Redeem points at a toll or fuel merchant, then try to cheat.

RoadPoint must never behave like a currency, because cryptocurrency trading is
illegal in Bangladesh. The only way points leave is redemption at a registered
merchant, which burns them. The car signs (amount, merchant, nonce); the
merchant submits it. Neither can act alone, and a signature works exactly once.

With --attacks the script also runs every way around that and records what the
contract did. That table is the security evidence for redemption:

  replay          merchant submits the same signed voucher twice
  inflate         merchant raises the amount after the car signed
  non-merchant    someone without the MERCHANT role submits a valid voucher
  spend bond      car tries to redeem more than its earned points
  transfer        car sends points to another car
  redeem bond     explicit bond cash-out

    python -m src.car.keeper            (first, so stakes are released)
    python -m src.car.redeem
    python -m src.car.redeem --car 2 --amount 3 --merchant "toll-plaza-1" --attacks
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.car.chain import KEYS, Chain, role_key  # noqa: E402

RESULTS = cfg.ROOT / "results"


def points(ch: Chain, addr: str) -> dict:
    rp = ch.c["RoadPoint"]
    f = rp.functions
    return {"balance": f.balanceOf(addr).call(), "bond": f.bondOf(addr).call(),
            "locked": f.lockedOf(addr).call(), "free": f.freeBalance(addr).call(),
            "nonce": f.nonceOf(addr).call()}


def voucher(car: Chain, amount: int, merchant_id: bytes, nonce: int) -> bytes:
    """The car's signature over exactly what RoadPoint.redeemDigest hashes.

    redeemDigest already applies the Ethereum signed-message prefix, so the car
    signs the inner hash with that same prefix, and the contract recovers it.
    """
    from eth_abi import encode
    from eth_account.messages import encode_defunct
    from web3 import Web3
    rp = car.c["RoadPoint"]
    inner = Web3.keccak(encode(
        ["uint256", "address", "address", "uint256", "bytes32", "uint256"],
        [car.chain_id, rp.address, car.address, amount, merchant_id, nonce]))
    signed = car.account.sign_message(encode_defunct(primitive=inner))
    # cross-check against the contract before anything is sent
    want = rp.functions.redeemDigest(car.address, amount, merchant_id, nonce).call()
    from eth_account.messages import _hash_eip191_message
    got = _hash_eip191_message(encode_defunct(primitive=inner))
    if bytes(want) != bytes(got):
        raise SystemExit("local digest does not match RoadPoint.redeemDigest; "
                         "chain id or contract address is wrong")
    return bytes(signed.signature)


def load_cars() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(KEYS.glob("car*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", type=int, default=None,
                    help="car index; default is the one with the most free points")
    ap.add_argument("--amount", type=int, default=0, help="default: all free points")
    ap.add_argument("--merchant", default="toll-plaza-1")
    ap.add_argument("--attacks", action="store_true")
    args = ap.parse_args()

    mk = role_key("MERCHANT")
    if not mk:
        raise SystemExit("no MERCHANT_KEY in chain/besu/.env — run node scripts/roles.js "
                         "and redeploy")
    merchant = Chain(mk)
    from web3 import Web3
    merchant_id = Web3.keccak(text=args.merchant)

    cars = load_cars()
    if not cars:
        raise SystemExit("no car keys in config/keys — register cars first")
    chains = [Chain(c["private_key"]) for c in cars]

    print(f"merchant {merchant.address}  id '{args.merchant}'\n")
    print(f"  {'car':6}{'balance':>9}{'bond':>7}{'locked':>8}{'free':>6}{'nonce':>7}")
    pts = []
    for i, ch in enumerate(chains):
        p = points(ch, ch.address)
        pts.append(p)
        print(f"  car{i:<3}{p['balance']:>9}{p['bond']:>7}{p['locked']:>8}"
              f"{p['free']:>6}{p['nonce']:>7}")

    i = args.car if args.car is not None else max(range(len(pts)),
                                                    key=lambda k: pts[k]["free"])
    car = chains[i]
    free = pts[i]["free"]
    amount = args.amount or free
    if amount <= 0 and not args.attacks:
        print(f"\ncar{i} has no free points to redeem.")
        if pts[i]["locked"]:
            print(f"  {pts[i]['locked']} points are locked as stake on open faults.")
            print("  Run the keeper to expire stale ones:  python -m src.car.keeper")
        print("  Only earned rewards are redeemable; the registration bond never is.")
        return
    rp_m = merchant.c["RoadPoint"].functions
    rows = []

    def attempt(label: str, who: Chain, fn, expect: str) -> None:
        """expect is "accepted" or the exact error the contract should raise.

        Being blocked is not enough: a replay stopped by an empty balance says
        nothing about the nonce. Each case must fail for the reason it tests.
        """
        why = who.dry_run(fn)
        if why is None:
            r = who.send(fn)
            outcome, got = f"accepted (gas {r['gas_used']})", "accepted"
        else:
            outcome, got = why, why.split("(")[0]
        ok = got == expect
        rows.append({"case": label, "expected": expect, "outcome": outcome,
                     "as_designed": ok})
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:28} {outcome}")

    if args.attacks and not args.amount:
        # spend 1 so the replay and inflation cases still have balance left:
        # they must then be stopped by the signature check, not by poverty
        amount = 1
    need = amount + 1 if args.attacks else amount
    if free < need:
        print(f"\ncar{i} has {free} free point(s); "
              f"{'the attack run needs at least ' + str(need) if args.attacks else 'not enough to redeem ' + str(amount)}.")
        print("  Run the keeper first so staked points are released.")
        return

    nonce = pts[i]["nonce"]
    sig = voucher(car, amount, merchant_id, nonce)
    print(f"\nredeeming {amount} point(s) for car{i} at '{args.merchant}'")
    before = points(car, car.address)
    attempt("honest redemption", merchant,
            rp_m.redeem(car.address, amount, merchant_id, sig), "accepted")
    after = points(car, car.address)
    print(f"    balance {before['balance']} -> {after['balance']}   "
          f"nonce {before['nonce']} -> {after['nonce']}   (points burned)")

    if args.attacks:
        left = after["free"]
        print(f"\nattacks (car still has {left} free, so balance cannot be "
              f"what stops them)")
        attempt("replay the same voucher", merchant,
                rp_m.redeem(car.address, amount, merchant_id, sig), "BadSignature")
        n2 = after["nonce"]
        sig1 = voucher(car, 1, merchant_id, n2)
        attempt("merchant inflates amount", merchant,
                rp_m.redeem(car.address, min(2, left), merchant_id, sig1),
                "BadSignature")
        attempt("merchant changes merchant", merchant,
                rp_m.redeem(car.address, 1, Web3.keccak(text="other-station"), sig1),
                "BadSignature")
        other = chains[(i + 1) % len(chains)]
        attempt("non-merchant submits", other,
                other.c["RoadPoint"].functions.redeem(car.address, 1, merchant_id, sig1),
                "NotMerchant")
        big = after["bond"] + left
        sigb = voucher(car, big, merchant_id, n2)
        attempt("redeem into the bond", merchant,
                rp_m.redeem(car.address, big, merchant_id, sigb), "InsufficientFree")
        attempt("car-to-car transfer", car,
                car.c["RoadPoint"].functions.transfer(other.address, 1),
                "TransfersDisabled")
        attempt("approve a spender", car,
                car.c["RoadPoint"].functions.approve(other.address, 1),
                "ApprovalsDisabled")
        attempt("explicit bond cash-out", car,
                car.c["RoadPoint"].functions.redeemBond(car.address, 1),
                "BondNotRedeemable")
        final = points(car, car.address)
        print(f"\n    after all attacks: balance {final['balance']}, "
              f"nonce {final['nonce']}  (unchanged since the honest redemption)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "redeem.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        good = sum(r["as_designed"] for r in rows)
        print(f"\n{good}/{len(rows)} cases behaved as designed -> {out}")


if __name__ == "__main__":
    main()
