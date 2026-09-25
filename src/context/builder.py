"""Step 3: assemble the 5-stream context vector — the thesis novelty.

Stream 1 image        crop path                      (Phase 2 only)
Stream 2 sensor       jolt, speed, measured surface  (PVS)
Stream 3 road map     highway class, maxspeed, junction (Overpass)
Stream 4 environment  time of day
Stream 5 model meta   class, conf, bbox geometry     (YOLO)

    python -m src.context.builder --side left
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.context.overpass_client import OverpassClient  # noqa: E402
from src.detect.tti import CameraModel, JoltSeries, calibrate  # noqa: E402
from src.utils.io import load_jsonl, write_jsonl  # noqa: E402

SURFACE_COLS = ["asphalt_road", "cobblestone_road", "dirt_road",
                "paved_road", "unpaved_road"]
QUALITY_COLS = ["good_road_left", "regular_road_left", "bad_road_left",
                "good_road_right", "regular_road_right", "bad_road_right"]


def time_of_day(ts: Optional[float], lon: Optional[float] = None) -> dict[str, Any]:
    """Local clock time, not UTC.

    The PVS trace sits at about 51 W, so UTC put every afternoon frame in the
    "evening, night" bucket and fed that to the LLM. Longitude gives the local
    offset without needing a timezone database: 15 degrees per hour.
    """
    if ts is None:
        return {"hour": None, "period": None, "is_night": None, "utc_offset_h": None}
    off_h = round(float(lon) / 15.0) if lon is not None else 0
    local = datetime.fromtimestamp(float(ts), tz=timezone.utc) + timedelta(hours=off_h)
    h = local.hour
    period = ("night" if h < 6 else "morning" if h < 12
              else "afternoon" if h < 18 else "evening")
    return {"hour": h, "period": period, "is_night": h < 6 or h >= 19,
            "utc_offset_h": off_h}


def measured_surface(det: dict) -> Optional[str]:
    """Surface from the PVS labels — measured, not crowd-sourced."""
    for c in ("cobblestone_road", "dirt_road", "asphalt_road"):
        if det.get(c) == 1:
            return c.replace("_road", "")
    if det.get("unpaved_road") == 1:
        return "unpaved"
    if det.get("paved_road") == 1:
        return "paved"
    return None


def measured_quality(det: dict) -> Optional[str]:
    """Ground-truth road quality on the side the camera and wheels share."""
    for lvl in ("bad", "regular", "good"):
        if det.get(f"{lvl}_road_left") == 1 or det.get(f"{lvl}_road_right") == 1:
            return lvl
    return None


def crop_bbox(det: dict, out_dir: Path) -> Optional[str]:
    """Stream 1: padded crop around the box, saved for Phase 2."""
    src = det.get("frame_path")
    if not src or not Path(src).exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{det['detection_id']}.jpg"
    if dst.exists():
        return str(dst)
    x1, y1, x2, y2 = det["bbox_xyxy"]
    with Image.open(src) as im:
        w, h = im.size
        # Phase 2 kept answering "too blurry": a 0.8%-of-frame box crops to about
        # 120x60 px. Take a larger real region around the box instead of padding
        # by a fixed number of pixels, so the model gets actual detail and some
        # surrounding road for context.
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max((x2 - x1) / 2, (y2 - y1) / 2, cfg.CROP_MIN_PX / 2)
        # Boxes here are large (median 16% of the frame). Padding those by 1.6x
        # swallowed the whole dashcam view, bonnet included, so the model was
        # judging the scene rather than the marked region. Only small boxes need
        # generous context.
        small = half * 2 < min(w, h) * 0.35
        half *= cfg.CROP_CONTEXT_SCALE if small else cfg.CROP_CONTEXT_SCALE_LARGE
        # Clamp the centre first. A box whose coordinates fall outside the frame
        # (a detection at the very edge, or a frame smaller than the recorded
        # dimensions) otherwise produces left > right and PIL raises.
        cx = min(max(cx, 0.0), float(w))
        cy = min(max(cy, 0.0), float(h))
        half = max(8.0, min(half, float(max(w, h))))
        x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
        x1c, y1c = min(w, int(cx + half)), min(h, int(cy + half))
        if x1c - x0 < 8 or y1c - y0 < 8:
            return None
        crop = im.crop((x0, y0, x1c, y1c))
        if min(crop.size) < cfg.CROP_UPSCALE_TO:
            s = cfg.CROP_UPSCALE_TO / min(crop.size)
            crop = crop.resize((int(crop.width * s), int(crop.height * s)),
                               Image.LANCZOS)
        crop.save(dst, quality=95)
    return str(dst)


def impact_evidence(det: dict[str, Any], cam, jolt) -> dict[str, Any]:
    """Did a jolt occur when the geometry says the wheels reached this defect?

    The camera sees a defect while it is still ahead; the wheels arrive later.
    A box whose bottom edge sits at image row y lies at distance
    d = K / (y - y_horizon), so the wheels reach it after tau = d / v.

    Asking "was there a jolt at t + tau" is a corroboration test. Real damage
    answers yes. Constant cobblestone rumble is rough at every offset, so it
    shows no excess at that particular moment.
    """
    out = {"tau_s": None, "distance_m": None, "jolt_pct_at_impact": None,
           "jolt_pct_baseline": None, "excess_over_baseline": None,
           "camera_fitted": bool(getattr(cam, "fitted", False))}
    if cam is None or jolt is None:
        return out
    spd, y, ts = det.get("speed"), det.get("y_bottom"), det.get("timestamp")
    if None in (spd, y, ts):
        return out
    tau = cam.tau_s(float(y), float(spd))
    if tau is None:
        return out
    out["tau_s"] = round(tau, 2)
    d = cam.distance_m(float(y))
    out["distance_m"] = None if d is None else round(d, 1)
    hit = jolt.peak_pct_at(float(ts) + tau)
    base = jolt.baseline_pct(float(ts) + tau)
    out["jolt_pct_at_impact"] = None if hit is None else round(hit, 1)
    out["jolt_pct_baseline"] = None if base is None else round(base, 1)
    if hit is not None and base is not None:
        out["excess_over_baseline"] = round(hit - base, 1)
    return out


def build_event(det: dict, oc: OverpassClient, crop_dir: Path,
                cam=None, jolt=None) -> dict[str, Any]:
    lat, lon = det.get("latitude"), det.get("longitude")
    road = oc.road_context(lat, lon) if None not in (lat, lon) else {"matched": False}
    area = det.get("bbox_area_px")
    frac = (round(area / (det["img_w"] * det["img_h"]), 5)
            if area and det.get("img_w") else None)

    return {
        "detection_id": det["detection_id"],
        "timestamp": det.get("timestamp"),
        "lat": lat, "lon": lon,
        "stream1_image": {
            "frame_path": det.get("frame_path"),
            "bbox_xyxy": det.get("bbox_xyxy"),
            "crop_path": crop_bbox(det, crop_dir),
        },
        "stream2_sensor": {
            "jolt_z": det.get("jolt_z"),
            "jolt_z_std_1s": det.get("jolt_z_std_1s"),
            "jolt_z_max_1s": det.get("jolt_z_max_1s"),
            "jolt_z_ahead_3s": det.get("jolt_z_ahead_3s"),
            "jolt_z_ahead_5s": det.get("jolt_z_ahead_5s"),
            "jolt_ahead_pct_for_surface": det.get("jolt_z_ahead_3s_pct_for_surface"),
            "speed_kmh": det.get("speed_kmh"),
            "measured_surface": measured_surface(det),
            "speed_bump_labelled": bool(det.get("speed_bump_asphalt")
                                        or det.get("speed_bump_cobblestone")),
            "stationary": (det.get("speed_kmh") is not None
                           and det.get("speed_kmh") < 2.0),
            "impact": impact_evidence(det, cam, jolt),
            "impact": (impact_evidence(det, cam, jolt) if cam is not None
                       else {"tau_s": None}),
        },
        "stream3_road": road,
        "stream4_env": time_of_day(det.get("timestamp"), lon),
        "stream5_model": {
            "detector": det.get("model_weights"),
            "stream": det.get("stream"),
            "class_name": det.get("class_name"),
            "conf": det.get("conf"),
            "bbox_area_px": area,
            "bbox_area_frac": frac,
        },
        # ground truth, never shown to the LLM — evaluation only
        "gt_road_quality": measured_quality(det),
    }


def _sensor_text(s2: dict[str, Any]) -> str:
    """Stream 2 as text, phrased as corroboration rather than roughness.

    Absolute jolt is not evidence: cobblestone shakes the car constantly by
    design. What carries information is whether a jolt appeared at the moment
    the geometry predicts the wheels reached the thing the camera saw.
    """
    spd = s2.get("speed_kmh")
    if spd is not None and spd < 2.0:
        return ("Vehicle sensors: the vehicle is stationary, so the accelerometer "
                "says nothing about the road here. Judge on the image and the "
                "other evidence alone.")

    imp = s2.get("impact") or {}
    head = f"Vehicle sensors: speed {spd:.1f} km/h." if spd is not None else "Vehicle sensors:"

    if imp.get("tau_s") is None or imp.get("excess_over_baseline") is None:
        return head + (" No usable impact prediction for this detection, so the "
                       "accelerometer adds nothing here.")

    tau, dist = imp["tau_s"], imp["distance_m"]
    excess = imp["excess_over_baseline"]
    hit = imp["jolt_pct_at_impact"]

    if excess >= 20:
        verdict = ("a clear jolt arrived at exactly that moment, well above the "
                   "surrounding ride. The wheels confirm what the camera saw")
    elif excess >= 8:
        verdict = "a mild jolt arrived around that moment, slightly above the surrounding ride"
    elif excess <= -8:
        verdict = ("the ride was actually SMOOTHER than usual at that moment, which "
                   "argues against a real defect")
    else:
        verdict = ("nothing unusual happened at that moment — the ride was no rougher "
                   "than the surrounding road, which argues against a real defect")

    note = ""
    if not imp.get("camera_fitted", False):
        note = (" (Camera geometry could not be fitted on this trace, so the timing "
                "is approximate and this evidence is weak.)")

    return (f"{head} The marked region is about {dist:.0f} m ahead, so the wheels "
            f"reach it in {tau:.1f} s. At that predicted moment {verdict} "
            f"({hit:.0f}th percentile for this surface, against a "
            f"{imp['jolt_pct_baseline']:.0f}th percentile baseline nearby).{note}")


def event_to_text(ev: dict[str, Any]) -> str:
    """Streams 2-5 flattened. This exact string is Phase 1's input."""
    s2, s4, s5 = ev["stream2_sensor"], ev["stream4_env"], ev["stream5_model"]
    return "\n".join([
        f"Detector: {s5['detector']} reported class '{s5['class_name']}' with "
        f"confidence {s5['conf']}; the box covers {s5['bbox_area_frac']} of the frame.",
        OverpassClient.to_text(ev["stream3_road"]),
        f"Measured road surface: {s2['measured_surface']}.",
        _sensor_text(s2),
        f"Environment: {s4.get('period')}, "
        f"{'night' if s4.get('is_night') else 'daylight'}.",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    args = ap.parse_args()

    tag = cfg.slug(args.trace)
    dets = load_jsonl(cfg.DETECTIONS / f"dets_{tag}_{args.side}.jsonl")
    print(f"{len(dets)} detections")

    oc = OverpassClient()
    # one query per map tile instead of one per detection
    oc.prefetch((d.get("latitude"), d.get("longitude")) for d in dets)

    # Crops are cached by detection_id, so changing the crop settings used to
    # silently reuse the old images. Fold the settings into the folder name.
    # Fit the camera geometry from this trace before building any context.
    jolt_path = cfg.RAW / f"jolt_{tag}_{args.side}.npz"
    cam, jolt = None, None
    if jolt_path.exists():
        jolt = JoltSeries.load(jolt_path)
        fit_dets = [{"timestamp": d.get("timestamp"), "speed_mps": d.get("speed"),
                     "y_bottom": d.get("y_bottom"), "conf": d.get("conf")}
                    for d in dets]
        img_h = dets[0].get("img_h", 720) if dets else 720
        print("fitting camera geometry from jolt alignment:")
        cam = calibrate(fit_dets, jolt, img_h=img_h, top_frac=0.3)
        (cfg.DATA / f"camera_{tag}_{args.side}.json").write_text(
            json.dumps(cam.to_dict(), indent=2))
    else:
        print(f"note: {jolt_path.name} missing — rerun extract_frames to enable "
              f"time-to-impact evidence")

    stamp = (f"m{cfg.CROP_MIN_PX}_s{cfg.CROP_CONTEXT_SCALE}"
             f"_L{cfg.CROP_CONTEXT_SCALE_LARGE}_u{cfg.CROP_UPSCALE_TO}")
    crop_dir = cfg.RAW / "crops" / tag / stamp
    print(f"crops -> {crop_dir}")
    events = []
    for d in tqdm(dets, desc="building context"):
        ev = build_event(d, oc, crop_dir, cam, jolt)
        ev["phase1_input_text"] = event_to_text(ev)
        events.append(ev)

    out = cfg.EVENTS / f"events_{tag}_{args.side}.jsonl"
    write_jsonl(out, events)
    matched = sum(1 for e in events if e["stream3_road"].get("matched"))
    print(f"wrote {len(events)} events -> {out}  ({matched} matched to an OSM road)")
    if events:
        print("\n--- sample Phase 1 input ---\n" + events[0]["phase1_input_text"])


if __name__ == "__main__":
    main()
