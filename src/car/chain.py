"""Talking to the deployed contracts from Python.

Plan section 8. Nothing here hardcodes an address: everything is read from
chain/deployments/besu.json, which the deploy script writes only after all
seven wiring checks pass. Redeploy and this follows automatically.

Gas is free on this chain, so cars hold no balance and never need one.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

CHAIN_DIR = cfg.ROOT / "chain"
DEPLOYMENTS = CHAIN_DIR / "deployments"
ARTIFACTS = CHAIN_DIR / "artifacts" / "contracts"
KEYS = cfg.ROOT / "config" / "keys"


def load_deployment(network: str = "besu") -> dict:
    p = DEPLOYMENTS / f"{network}.json"
    if not p.exists():
        raise SystemExit(
            f"{p} not found.\n"
            f"Deploy first:  cd chain && npx hardhat run scripts/deploy.js "
            f"--network {network}")
    return json.loads(p.read_text())


def load_abi(name: str) -> list:
    """Read the ABI straight out of Hardhat's build output, so it never drifts."""
    p = ARTIFACTS / f"{name}.sol" / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"{p} not found — run `npx hardhat compile` in chain/")
    return json.loads(p.read_text())["abi"]


def error_table(abis: list[list]) -> dict[str, tuple[str, list[str], list[str]]]:
    """Map 4-byte selector -> (error name, arg types, arg names), from every ABI.

    Custom errors come back from the node as raw hex like 0x62b244c0...; this is
    what turns that into InsufficientFree(have=0, need=5).
    """
    from web3 import Web3
    out = {}
    for abi in abis:
        for e in abi:
            if e.get("type") != "error":
                continue
            types = [i["type"] for i in e.get("inputs", [])]
            names = [i.get("name") or f"arg{k}" for k, i in
                     enumerate(e.get("inputs", []))]
            sig = f"{e['name']}({','.join(types)})"
            out["0x" + Web3.keccak(text=sig)[:4].hex().removeprefix("0x")] = (
                e["name"], types, names)
    return out


def decode_revert(msg: str, table: dict) -> str:
    """Best-effort readable form of a revert message."""
    m = re.search(r"0x[0-9a-fA-F]{8,}", msg or "")
    if m:
        data = m.group(0)
    else:
        # some nodes report the revert data as a Python bytes literal
        b = re.search(r"b'((?:[^'\\]|\\.)*)'", msg or "")
        if not b:
            return " ".join((msg or "").split())[:100] or "(no reason returned)"
        import ast
        data = "0x" + ast.literal_eval("b'" + b.group(1) + "'").hex()
        if len(data) < 10:
            return " ".join((msg or "").split())[:100]
    hit = table.get(data[:10].lower())
    if not hit:
        return f"unknown error {data[:10]}"
    name, types, names = hit
    if not types:
        return f"{name}()"
    try:
        from eth_abi import decode
        vals = decode(types, bytes.fromhex(data[10:]))
        return f"{name}(" + ", ".join(f"{n}={v}" for n, v in zip(names, vals)) + ")"
    except Exception:                                          # noqa: BLE001
        # the message is sometimes truncated; the name alone is still useful
        return f"{name}(...)"


class Chain:
    """Contract handles plus transaction sending, for one account."""

    def __init__(self, private_key: str | None = None, network: str = "besu",
                 rpc: str | None = None):
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware

        self.dep = load_deployment(network)
        url = rpc or "http://127.0.0.1:8545"
        self.w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 30}))
        # QBFT puts consensus data in extraData; without this block reads raise
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        if not self.w3.is_connected():
            raise SystemExit(
                f"no chain at {url}\n"
                f"  cd chain && docker compose -f besu/docker-compose.yml up -d")

        self.chain_id = self.dep["chainId"]
        self.account = None
        if private_key:
            from eth_account import Account
            self.account = Account.from_key(private_key)

        self.c = {}
        abis = []
        for name, addr in self.dep["contracts"].items():
            abi = load_abi(name)
            abis.append(abi)
            self.c[name] = self.w3.eth.contract(
                address=self.w3.to_checksum_address(addr), abi=abi)
        self.errors = error_table(abis)

    @property
    def address(self) -> str:
        if self.account is None:
            raise RuntimeError("this Chain has no key; pass private_key")
        return self.account.address

    def dry_run(self, fn) -> str | None:
        """Would this revert? Returns the decoded reason, or None if it would pass.

        A reverted transaction is still mined and still returns a receipt, with
        status 0, so checking first is what keeps failures from being counted
        as successes.
        """
        try:
            fn.call({"from": self.address})
            return None
        except Exception as e:                                 # noqa: BLE001
            return decode_revert(str(getattr(e, "data", "") or e), self.errors)

    def send(self, fn, gas: int = 2_000_000, check: bool = True) -> dict:
        """Sign, send, wait. Raises if the transaction reverted on chain."""
        if self.account is None:
            raise RuntimeError("no key loaded")
        nonce = self.w3.eth.get_transaction_count(self.address)
        tx = fn.build_transaction({
            "from": self.address, "nonce": nonce, "gas": gas,
            "gasPrice": 0, "chainId": self.chain_id,
        })
        signed = self.account.sign_transaction(tx)
        sent = time.time()
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(h, timeout=120)
        mined = time.time()
        if check and rcpt["status"] != 1:
            raise RuntimeError(
                f"transaction reverted on chain (status 0), tx {h.hex()}")
        return {"tx_hash": h.hex(), "status": rcpt["status"],
                "gas_used": rcpt["gasUsed"], "block": rcpt["blockNumber"],
                "sent_at": round(sent, 3), "mined_at": round(mined, 3),
                "latency_s": round(mined - sent, 3)}

    def events(self, contract: str, event: str, tx_hash: str) -> list:
        from web3.logs import DISCARD
        r = self.w3.eth.get_transaction_receipt(tx_hash)
        return list(getattr(self.c[contract].events, event)().process_receipt(
            r, errors=DISCARD))


def car_key_path(car_id: int) -> Path:
    return KEYS / f"car{car_id:03d}.json"


def load_or_create_key(car_id: int) -> dict:
    """One persistent key per car: the key is its identity and its reputation."""
    p = car_key_path(car_id)
    if p.exists():
        return json.loads(p.read_text())
    from eth_account import Account
    acct = Account.create()
    KEYS.mkdir(parents=True, exist_ok=True)
    rec = {"car_id": car_id, "address": acct.address,
           "private_key": acct.key.hex()}
    p.write_text(json.dumps(rec, indent=2))
    return rec


def _env_value(key: str) -> str | None:
    env = CHAIN_DIR / "besu" / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def admin_key() -> str:
    k = _env_value("DEPLOYER_KEY")
    if not k:
        raise SystemExit("DEPLOYER_KEY missing — run node besu/prepare.js in chain/")
    return k


def role_key(role: str) -> str | None:
    return _env_value(f"{role}_KEY")
