"""Overpass API client — road class, speed limit, lanes, junction.

Replaces Nominatim. Nominatim does reverse geocoding (addresses) only; it cannot
return highway class, maxspeed, lanes or junction. Those are exactly what the
Phase 1 prior needs.

Note: surface is NOT taken from OSM here — dataset_labels.csv already gives
measured surface type per sample, which beats a crowd-sourced tag.

    python -m src.context.overpass_client
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.cache import JsonCache  # noqa: E402

WANTED_TAGS = ("highway", "maxspeed", "lanes", "junction", "name",
               "oneway", "bridge", "tunnel", "lit")

HIGHWAY_RANK = {
    "motorway": 9, "trunk": 8, "primary": 7, "secondary": 6, "tertiary": 5,
    "unclassified": 4, "residential": 4, "service": 3, "living_street": 3,
    "track": 2, "path": 1,
}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _point_to_segment_m(lat, lon, lat1, lon1, lat2, lon2) -> float:
    """Local equirectangular projection — accurate below ~1 km, very cheap."""
    lat0 = math.radians((lat1 + lat2) / 2)
    kx, ky = 111320.0 * math.cos(lat0), 110540.0
    px, py = lon * kx, lat * ky
    ax, ay = lon1 * kx, lat1 * ky
    bx, by = lon2 * kx, lat2 * ky
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _way_distance_m(lat, lon, geometry: list[dict]) -> float:
    if not geometry:
        return float("inf")
    if len(geometry) == 1:
        return haversine_m(lat, lon, geometry[0]["lat"], geometry[0]["lon"])
    return min(
        _point_to_segment_m(lat, lon,
                            geometry[i]["lat"], geometry[i]["lon"],
                            geometry[i + 1]["lat"], geometry[i + 1]["lon"])
        for i in range(len(geometry) - 1)
    )


def _parse_maxspeed(raw: Optional[str]) -> Optional[float]:
    """'50' -> 50.0, '30 mph' -> 48.28, 'none'/'walk' -> None."""
    if not raw:
        return None
    parts = raw.strip().lower().split()
    try:
        v = float(parts[0])
    except ValueError:
        return None
    return v * 1.609344 if len(parts) > 1 and parts[1] == "mph" else v


class OverpassClient:
    def __init__(self, url: str = cfg.OVERPASS_URL, radius_m: int = cfg.OVERPASS_RADIUS_M):
        self.url, self.radius_m = url, radius_m
        self.cache = JsonCache(cfg.CACHE, "overpass")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": f"VERMA-FL-UL/0.1 ({cfg.OVERPASS_CONTACT})"})
        self._last = 0.0

    # One Overpass query covers a whole tile, not a single point. At 4 fps a
    # vehicle produces hundreds of nearby detections, and querying each one
    # separately meant ~15 s per detection. A 0.01 deg tile is roughly 1.1 km.
    TILE_DEG = 0.01
    TILE_MARGIN_DEG = 0.002      # covers points sitting on a tile edge

    def _tile(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(math.floor(lat / self.TILE_DEG)),
                int(math.floor(lon / self.TILE_DEG)))

    def _key(self, lat: float, lon: float) -> str:
        ty, tx = self._tile(lat, lon)
        return f"tile,{ty},{tx},{self.TILE_DEG}"

    def _query(self, lat: float, lon: float, retries: int = 3) -> dict[str, Any]:
        key = self._key(lat, lon)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        ty, tx = self._tile(lat, lon)
        m = self.TILE_MARGIN_DEG
        s = ty * self.TILE_DEG - m
        w = tx * self.TILE_DEG - m
        n = (ty + 1) * self.TILE_DEG + m
        e = (tx + 1) * self.TILE_DEG + m
        ql = (f"[out:json][timeout:{cfg.OVERPASS_TIMEOUT_S}];\n"
              f"(way({s:.6f},{w:.6f},{n:.6f},{e:.6f})[highway];\n"
              f' node({s:.6f},{w:.6f},{n:.6f},{e:.6f})'
              f'["highway"~"traffic_signals|crossing|stop|give_way"];);\n'
              f"out tags geom;\n")

        backoff = 2.0
        for attempt in range(retries):
            gap = time.monotonic() - self._last
            if gap < cfg.OVERPASS_MIN_INTERVAL_S:
                time.sleep(cfg.OVERPASS_MIN_INTERVAL_S - gap)
            try:
                r = self.session.post(self.url, data={"data": ql},
                                      timeout=cfg.OVERPASS_TIMEOUT_S + 10)
                self._last = time.monotonic()
                if r.status_code in (429, 502, 503, 504):
                    time.sleep(backoff); backoff *= 2; continue
                r.raise_for_status()
                payload = r.json()
                self.cache.set(key, payload)
                return payload
            except (requests.RequestException, ValueError) as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Overpass failed at {lat},{lon}: {e}") from e
                time.sleep(backoff); backoff *= 2
        return {"elements": []}

    def road_context(self, lat: float, lon: float) -> dict[str, Any]:
        """Stream 3 of the context vector, as one flat dict."""
        elements = self._query(lat, lon).get("elements", [])
        ways = [e for e in elements
                if e.get("type") == "way" and e.get("tags", {}).get("highway")]
        nodes = [e for e in elements if e.get("type") == "node"
                 and haversine_m(lat, lon, e.get("lat", 0), e.get("lon", 0))
                 <= self.radius_m]

        ctx: dict[str, Any] = {
            "matched": False, "distance_to_road_m": None, "highway": None,
            "highway_rank": None, "maxspeed_kmh": None, "lanes": None,
            "junction": None, "name": None, "oneway": None,
            "bridge": False, "tunnel": False, "lit": None,
            "n_nearby_ways": 0, "is_intersection": False,
            "nearby_control_nodes": [],
        }
        if not ways:
            return ctx

        scored = [(w, _way_distance_m(lat, lon, w.get("geometry", []))) for w in ways]
        scored = [(w, d) for w, d in scored if d <= self.radius_m * 4]
        if not scored:
            ctx["n_nearby_ways"] = 0
            return ctx
        best, dist = min(scored, key=lambda x: x[1])
        tags = best.get("tags", {})
        ctx["n_nearby_ways"] = len([1 for _, d in scored if d <= self.radius_m])
        ctx.update({
            "matched": True,
            "distance_to_road_m": round(dist, 2),
            "highway_rank": HIGHWAY_RANK.get(tags.get("highway"), 0),
            "maxspeed_kmh": _parse_maxspeed(tags.get("maxspeed")),
            "bridge": tags.get("bridge") not in (None, "no"),
            "tunnel": tags.get("tunnel") not in (None, "no"),
        })
        for t in WANTED_TAGS:
            if t not in ("maxspeed", "bridge", "tunnel"):
                ctx[t] = tags.get(t)

        control = [n.get("tags", {}).get("highway") for n in nodes]
        ctx["nearby_control_nodes"] = [c for c in control if c]
        near = [w for w, d in scored if d <= self.radius_m]
        names = {w["tags"].get("name") for w in near if w.get("tags", {}).get("name")}
        ctx["is_intersection"] = bool(
            tags.get("junction") or ctx["nearby_control_nodes"] or len(names) > 1)
        return ctx

    def prefetch(self, coords) -> None:
        """Warm the cache for every distinct tile a route touches.

        Turns hundreds of per-point queries into a handful of tile queries.
        """
        tiles = {}
        for lat, lon in coords:
            if lat is None or lon is None:
                continue
            tiles.setdefault(self._tile(float(lat), float(lon)), (float(lat), float(lon)))
        todo = [(k, v) for k, v in tiles.items() if self.cache.get(self._key(*v)) is None]
        print(f"Overpass: {len(tiles)} tiles cover this route, {len(todo)} not cached")
        for i, (_, (lat, lon)) in enumerate(todo, 1):
            self._query(lat, lon)
            print(f"  tile {i}/{len(todo)}", end="\r", flush=True)
        if todo:
            print()

    @staticmethod
    def to_text(ctx: dict[str, Any]) -> str:
        if not ctx.get("matched"):
            return "Road context: no mapped road within search radius."
        b = [f"road class {ctx['highway']}"]
        if ctx.get("name"):
            b.append(f"named {ctx['name']}")
        if ctx.get("maxspeed_kmh"):
            b.append(f"speed limit {ctx['maxspeed_kmh']:.0f} km/h")
        if ctx.get("lanes"):
            b.append(f"{ctx['lanes']} lanes")
        if ctx.get("is_intersection"):
            b.append("at or near an intersection")
        if ctx.get("bridge"):
            b.append("on a bridge")
        if ctx.get("tunnel"):
            b.append("in a tunnel")
        b.append(f"{ctx['distance_to_road_m']:.0f} m from road centreline")
        return "Road context: " + ", ".join(b) + "."


if __name__ == "__main__":
    oc = OverpassClient()
    c = oc.road_context(-27.717803, -51.098857)   # PVS trace, Santa Catarina
    print(c)
    print(oc.to_text(c))
