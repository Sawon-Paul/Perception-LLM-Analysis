"""Talking to the deployed contracts from Python.

Plan section 8. Nothing here hardcodes an address: everything is read from
chain/deployments/besu.json, which the deploy script writes only after all
seven wiring checks pass. Redeploy and this follows automatically.

Gas is free on this chain, so cars hold no balance and never need one. That is
the whole reason for the zero-gas genesis: a design where a vehicle must buy a
native coin to file a report is unusable where cryptocurrency trading is
illegal.
"""
from __future__ import annotations

import json
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
    """Read the ABI straight out of Hardhat's build output.

    Keeping a second copy of the ABI in Python would drift the moment a
    contract changes, and the drift shows up as a silent decoding failure
    rather than an error.
    """
    p = ARTIFACTS / f"{name}.sol" / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"{p} not found — run `npx hardhat compile` in chain/")
    return json.loads(p.read_text())["abi"]


class Chain:
    """Contract handles plus transaction sending, for one account."""

    def __init__(self, private_key: str | None = None, network: str = "besu",
                 rpc: str | None = None):
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware

        self.dep = load_deployment(network)
        url = rpc or f"http://127.0.0.1:8545"
        self.w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 30}))
        # QBFT puts consensus data in extraData, which exceeds what the default
        # block formatter accepts; without this every block read raises.
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
        for name, addr in self.dep["contracts"].items():
            self.c[name] = self.w3.eth.contract(
                address=self.w3.to_checksum_address(addr), abi=load_abi(name))

    @property
    def address(self) -> str:
        if self.account is None:
            raise RuntimeError("this Chain has no key; pass private_key")
        return self.account.address

    def dry_run(self, fn) -> str | None:
        """Would this revert? Returns the reason, or None if it would succeed.

        Checking first matters because a reverted transaction is still mined
        and still returns a receipt \u2014 with status 0. Reading only the
        receipt made failures look like successes, and a run reported 75
        transactions when most of the confirmations had reverted.
        """
        from web3.exceptions import ContractLogicError
        try:
            fn.call({"from": self.address})
            return None
        except ContractLogicError as e:
            return str(e)
        except Exception as e:                      # noqa: BLE001
            return f"call failed: {e}"

    def send(self, fn, gas: int = 2_000_000, check: bool = True) -> dict:
        """Sign and send, then wait for the receipt.

        Returns the receipt fields worth logging rather than the raw object,
        because gas used and the two timestamps are what the thesis reports.

        Raises on a reverted transaction rather than returning it quietly.
        """
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
                f"transaction reverted on chain (status 0), tx {h.hex()}, "
                f"gas used {rcpt['gasUsed']}")
        return {"tx_hash": h.hex(), "status": rcpt["status"],
                "gas_used": rcpt["gasUsed"], "block": rcpt["blockNumber"],
                "sent_at": round(sent, 3), "mined_at": round(mined, 3),
                "latency_s": round(mined - sent, 3)}

    def events(self, contract: str, event: str, rcpt_hash: str) -> list:
        """Decode events from a transaction we just sent."""
        r = self.w3.eth.get_transaction_receipt(rcpt_hash)
        return list(getattr(self.c[contract].events, event)().process_receipt(
            r, errors=__import__("web3").logs.DISCARD))


def car_key_path(car_id: int) -> Path:
    return KEYS / f"car{car_id:03d}.json"


def load_or_create_key(car_id: int) -> dict:
    """One key pair per car, kept out of git.

    A car's key is its identity on the chain, so it has to persist between
    runs; regenerating would orphan its registration and its reputation.
    """
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


def admin_key() -> str:
    """The deployer key, written to chain/besu/.env by besu/prepare.js."""
    env = CHAIN_DIR / "besu" / ".env"
    if not env.exists():
        raise SystemExit(f"{env} not found — run node besu/prepare.js in chain/")
    for line in env.read_text().splitlines():
        if line.startswith("DEPLOYER_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("DEPLOYER_KEY missing from chain/besu/.env")


def role_key(role: str) -> str | None:
    env = CHAIN_DIR / "besu" / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        if line.startswith(f"{role}_KEY="):
            return line.split("=", 1)[1].strip()
    return None
