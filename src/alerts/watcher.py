"""Watch FaultLifecycle and turn state changes into alerts.

Read-only: it holds no key and sends nothing, so anyone can run it. That is the
public read gateway from the plan, and it is why a non-member car or the road
authority can see alerts without being able to write to the chain.

It polls fault state rather than subscribing to events. Polling survives a
gateway restart and a node restart without missing anything, because the
chain's current state is the source of truth, not a stream that can drop.

Alert latency is measured here: when a fault becomes Confirmed, the gap between
the block that confirmed it and the moment the gateway saw it is recorded. That
is the chain-to-car delay the thesis reports.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
STATE = ["None", "Pending", "Confirmed", "RepairClaimed", "Disputed",
         "Rejected", "Closed", "Expired"]
FAULT_TYPES = {0: "pothole", 1: "alligator cracking", 2: "lateral cracking",
               3: "longitudinal cracking"}
# states a driver should be warned about
ALERTING = {"Confirmed", "RepairClaimed"}


def geohash_decode(gh: str) -> tuple[float, float]:
    """Centre of a geohash cell. geohash-8 cells are about 38 m x 19 m."""
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    even = True
    for ch in gh:
        v = _B32.index(ch)
        for bit in (16, 8, 4, 2, 1):
            rng = lon if even else lat
            mid = (rng[0] + rng[1]) / 2
            if v & bit:
                rng[0] = mid
            else:
                rng[1] = mid
            even = not even
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


def _gh(raw) -> str:
    b = bytes(raw)
    return b.rstrip(b"\x00").decode("ascii", "replace")


class FaultWatcher:
    """Tracks every fault's state and reports what changed since the last poll."""

    def __init__(self, chain, latency_csv: Path | None = None):
        self.ch = chain
        self.fl = chain.c["FaultLifecycle"]
        self.known: dict[int, dict] = {}
        self.latency_csv = latency_csv
        self.latencies: list[float] = []
        self.first_poll = True
        # (local clock - newest block time) on recent polls; the minimum is the
        # clock offset between this machine and the chain, to about 1 s
        self._offsets: list[float] = []

    def _read(self, fid: int) -> dict:
        f = self.fl.functions.faults(fid).call()
        gh = _gh(f[1])
        lat, lon = geohash_decode(gh) if gh else (None, None)
        return {"id": fid, "type": FAULT_TYPES.get(f[0], str(f[0])),
                "geohash": gh, "lat": lat, "lon": lon, "severity": f[2],
                "state": STATE[f[3]], "reported_at": f[4], "confirmed_at": f[5],
                "contributors": f[7]}

    def poll(self) -> list[dict]:
        """Read every fault; return a change record for each one that moved."""
        changes = []
        total = self.fl.functions.faultCount().call()
        now = time.time()
        self._track_offset(now)
        for fid in range(1, total + 1):
            prev = self.known.get(fid)
            # settled faults never change again, so skip re-reading them
            if prev and prev["state"] in ("Closed", "Expired", "Rejected"):
                continue
            cur = self._read(fid)
            self.known[fid] = cur
            if prev is None and self.first_poll:
                continue                     # history, not news
            before = prev["state"] if prev else "None"
            if before == cur["state"]:
                continue
            ch = {"fault": cur, "from": before, "to": cur["state"],
                  "seen_at": round(now, 3)}
            if cur["state"] == "Confirmed" and cur["confirmed_at"]:
                lag = round(now - cur["confirmed_at"] - self.clock_offset(), 3)
                ch["latency_s"] = lag
                self.latencies.append(lag)
                self._log(cur, lag, now)
            changes.append(ch)
        self.first_poll = False
        return changes

    def _track_offset(self, now: float) -> None:
        """Estimate how far this machine's clock is from the chain's.

        Block timestamps come from the validators' clocks; "seen" comes from
        this machine's. Docker on Windows runs in its own VM whose clock drifts,
        so subtracting the two directly can even go negative. A block is mined
        every 2 s, so on some poll the newest block is only a moment old, and
        the smallest (now - newest block time) seen is the clock offset.
        """
        try:
            w3 = getattr(self.ch, "w3", None)
            if w3 is None:
                return
            ts = w3.eth.get_block("latest")["timestamp"]
            self._offsets = (self._offsets + [now - ts])[-60:]
        except Exception:                                      # noqa: BLE001
            pass

    def clock_offset(self) -> float:
        return min(self._offsets) if self._offsets else 0.0

    def _log(self, fault: dict, lag: float, now: float) -> None:
        if not self.latency_csv:
            return
        new = not self.latency_csv.exists()
        self.latency_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.latency_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["fault_id", "type", "confirmed_block_time",
                            "seen_by_gateway", "clock_offset_s", "latency_s"])
            w.writerow([fault["id"], fault["type"], fault["confirmed_at"],
                        round(now, 3), round(self.clock_offset(), 3), lag])

    def alerts(self) -> list[dict]:
        return [f for f in self.known.values()
                if f["state"] in ALERTING and f["lat"] is not None]

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for f in self.known.values():
            c[f["state"]] = c.get(f["state"], 0) + 1
        return c
