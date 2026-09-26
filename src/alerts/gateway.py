"""Public read gateway and in-car alert screen.

    python -m src.alerts.gateway
    then open  http://127.0.0.1:8000/?trace=PVS%201

The gateway polls the chain once a second, keeps the list of faults a driver
should be warned about, and pushes every change to connected screens over a
WebSocket. It has no key, so it can read but never write.

The screen drives a car along a real PVS trace and warns as it approaches a
confirmed fault. Leaflet is served from this folder, so it works without
internet; map tiles need a connection, but the faults and the car do not.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

# Module level on purpose: with postponed annotations FastAPI resolves the
# WebSocket type hint against module globals, and an import inside the function
# leaves it unresolved, so the socket closes on connect.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

STATIC = Path(__file__).parent / "static"
RESULTS = cfg.ROOT / "results"


def load_route(trace: str, step_s: float = 2.0) -> list[list[float]]:
    """GPS points along a trace, thinned to about one every step_s seconds."""
    slug = cfg.slug(trace)
    raw = cfg.DATA / "raw" / f"frames_{slug}_left.csv"
    pts: list[list[float]] = []
    if raw.exists():
        rows = list(csv.DictReader(raw.open(encoding="utf-8")))
        tcol = next((c for c in ("timestamp", "t", "time_s") if rows and c in rows[0]),
                    None)
        last_t = None
        for i, r in enumerate(rows):
            try:
                lat, lon = float(r["latitude"]), float(r["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            t = float(r[tcol]) if tcol and r.get(tcol) else i / 4.0
            if last_t is None or t - last_t >= step_s:
                pts.append([lat, lon])
                last_t = t
    if not pts:
        # fall back to detection positions, which follow the same road
        for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
            p = cfg.EVENTS / f"events_{slug}_left{suffix}.jsonl"
            if p.exists():
                for line in p.open(encoding="utf-8"):
                    if line.strip():
                        e = json.loads(line)
                        if e.get("lat") is not None:
                            pts.append([float(e["lat"]), float(e["lon"])])
                break
    return pts


def create_app(watcher, poll_s: float = 1.0):
    app = FastAPI(title="Road fault alerts (read-only)")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    clients: set = set()
    state = {"error": None, "polls": 0}

    async def broadcast(msg: dict) -> None:
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:                                  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    async def loop() -> None:
        while True:
            try:
                changes = await run_in_threadpool(watcher.poll)
                state["error"] = None
                state["polls"] += 1
                for c in changes:
                    await broadcast({"type": "change", **c})
                    lat = f"  latency {c['latency_s']}s" if "latency_s" in c else ""
                    print(f"  fault {c['fault']['id']}: {c['from']} -> {c['to']}{lat}",
                          flush=True)
                if changes:
                    await broadcast({"type": "snapshot", "alerts": watcher.alerts()})
            except Exception as e:                             # noqa: BLE001
                if state["error"] != str(e):
                    print(f"  chain read failed: {e}", flush=True)
                state["error"] = str(e)
            await asyncio.sleep(poll_s)

    @app.on_event("startup")
    async def _start() -> None:
        app.state.task = asyncio.create_task(loop())

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "car.html")

    @app.get("/api/alerts")
    async def alerts():
        return JSONResponse({"alerts": watcher.alerts(), "counts": watcher.counts(),
                             "chain_ok": state["error"] is None,
                             "polls": state["polls"]})

    @app.get("/api/route")
    async def route(trace: str = "PVS 1"):
        return JSONResponse({"trace": trace, "points": load_route(trace)})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        clients.add(sock)
        await sock.send_text(json.dumps({"type": "snapshot",
                                         "alerts": watcher.alerts()}))
        try:
            while True:
                await sock.receive_text()          # keep-alive; screens send nothing
        except WebSocketDisconnect:
            clients.discard(sock)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--poll", type=float, default=1.0)
    args = ap.parse_args()

    from src.alerts.watcher import FaultWatcher
    from src.car.chain import Chain
    import uvicorn

    ch = Chain()                                   # no key: read-only by design
    w = FaultWatcher(ch, RESULTS / "alert_latency.csv")
    w.poll()
    print(f"gateway: {len(w.known)} faults on chain, {len(w.alerts())} alerting")
    print(f"open  http://{args.host}:{args.port}/?trace=PVS%201\n")
    uvicorn.run(create_app(w, args.poll), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
