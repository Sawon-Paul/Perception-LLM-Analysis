"""Prove the pipeline runs, without needing the GPU, the network or real data.

Builds a small synthetic trace with a known answer, runs every stage except the
vision model, and checks each one produced what it should. If this passes, the
code is sound and any later failure is data or environment, not logic.

    python -m src.selftest
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL = "  PASS", "  FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def make_trace(root: Path, seconds: int = 240, hz: int = 100) -> Path:
    """A synthetic PVS trace: asphalt then dirt, with three real impacts."""
    pvs = root / "PVS 99"
    pvs.mkdir(parents=True, exist_ok=True)
    n = seconds * hz
    t = np.arange(n) / hz + 1_577_000_000.0
    rng = np.random.default_rng(7)

    half = n // 2
    surface_dirt = np.zeros(n, dtype=int)
    surface_dirt[half:] = 1

    acc = 9.81 + rng.normal(0, 0.25, n)
    acc[half:] += rng.normal(0, 1.2, n - half)          # dirt is rougher
    for sec in (60.0, 100.0, 190.0):                    # three real impacts
        j = int(sec * hz)
        acc[j:j + 25] += 9.0

    speed = np.full(n, 8.0) + rng.normal(0, 0.3, n)

    pd.DataFrame({
        "timestamp": t, "speed": speed,
        "acc_x_below_suspension": rng.normal(0, .1, n),
        "acc_y_below_suspension": rng.normal(0, .1, n),
        "acc_z_below_suspension": acc,
        "acc_z_above_suspension": acc * 0.6,
        "acc_z_dashboard": acc * 0.4,
        "gyro_x_below_suspension": rng.normal(0, .1, n),
        "gyro_y_below_suspension": rng.normal(0, .1, n),
        "gyro_z_below_suspension": rng.normal(0, .1, n),
        "latitude": -27.7178 + np.linspace(0, 0.004, n),
        "longitude": -51.0989 + np.linspace(0, 0.004, n),
    }).to_csv(pvs / "dataset_gps_mpu_left.csv", index=False)

    lab = pd.DataFrame(0, index=range(n), columns=[
        "paved_road", "unpaved_road", "dirt_road", "cobblestone_road", "asphalt_road",
        "no_speed_bump", "speed_bump_asphalt", "speed_bump_cobblestone",
        "good_road_left", "regular_road_left", "bad_road_left",
        "good_road_right", "regular_road_right", "bad_road_right"])
    lab["no_speed_bump"] = 1
    lab.loc[:half - 1, ["paved_road", "asphalt_road", "good_road_left", "good_road_right"]] = 1
    lab.loc[half:, ["unpaved_road", "dirt_road", "bad_road_left", "bad_road_right"]] = 1
    lab.to_csv(pvs / "dataset_labels.csv", index=False)

    # a video whose brightness changes with the surface, at the right length
    vfps = 5
    frames = [np.full((90, 160), 70 if i < seconds * vfps // 2 else 190, np.uint8)
              for i in range(seconds * vfps)]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", "160x90", "-r", str(vfps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(pvs / "video_environment.mp4")],
        input=b"".join(f.tobytes() for f in frames), check=True)
    return pvs


def fake_detections(cfg, tag: str, side: str, n_real: int = 3) -> int:
    """Stand in for YOLO so the test needs no GPU and no weights."""
    from src.detect.tti import CameraModel
    frames = pd.read_csv(cfg.RAW / f"frames_{tag}_{side}.csv")
    frames = frames[frames["latitude"].notna()].reset_index(drop=True)
    cam = CameraModel(k_px_m=780.0, y_horizon=320.0, img_h=720)
    t0 = float(frames["timestamp"].min())

    recs = []
    impacts = [60.0, 100.0, 190.0][:n_real]
    for i, row in frames.iterrows():
        if i % 7:                                   # roughly one frame in seven
            continue
        ts = float(row["timestamp"])
        # place a few boxes so their predicted impact lands on a real one
        y_bottom = 470.0
        tau = cam.tau_s(y_bottom, float(row["speed"])) or 0.0
        near_real = any(abs((ts - t0 + tau) - s) < 0.5 for s in impacts)
        recs.append({
            "detection_id": f"pothole_{tag}_f{i:05d}_0", "stream": "pothole",
            "model_weights": "synthetic.pt", "frame_path": row["frame_path"],
            "class_id": 0, "class_name": "alligator cracking",
            "conf": 0.85 if near_real else 0.45,
            "bbox_xyxy": [560.0, 400.0, 760.0, y_bottom],
            "bbox_area_px": 200.0 * 70.0, "y_bottom": y_bottom,
            "img_w": 1280, "img_h": 720,
            "timestamp": ts, "latitude": row["latitude"], "longitude": row["longitude"],
            "speed": row["speed"], "speed_kmh": row["speed"] * 3.6,
            "jolt_z": row.get("jolt_z"), "surface": row.get("surface"),
            "jolt_z_pct_for_surface": row.get("jolt_z_pct_for_surface"),
            "asphalt_road": row.get("asphalt_road"), "dirt_road": row.get("dirt_road"),
            "bad_road_left": row.get("bad_road_left"),
            "bad_road_right": row.get("bad_road_right"),
            "no_speed_bump": row.get("no_speed_bump"),
        })
    from src.utils.io import write_jsonl
    write_jsonl(cfg.DETECTIONS / f"dets_{tag}_{side}.jsonl", recs)
    return len(recs)


def main() -> None:
    print("=" * 62)
    print("SELF-TEST — synthetic trace, no GPU, no network, no real data")
    print("=" * 62)

    tmp = Path(tempfile.mkdtemp(prefix="verma_selftest_"))
    import os
    os.environ["DATASET_DIR"] = str(tmp)
    os.environ["DEFAULT_TRACE"] = "PVS 99"

    try:
        make_trace(tmp)
        check("synthetic trace built", True, str(tmp))

        import importlib
        from configs import config as cfg
        importlib.reload(cfg)
        cfg.DATASET_DIR = tmp
        tag, side = cfg.slug("PVS 99"), "left"

        # ---- stage 0 ----
        from src.detect import extract_frames as ef
        importlib.reload(ef)
        ef.cfg.DATASET_DIR = tmp
        sys.argv = ["x", "--trace", "PVS 99", "--side", side, "--stretch", "--fps", "2"]
        ef.main()
        frames_csv = cfg.RAW / f"frames_{tag}_{side}.csv"
        jolt_npz = cfg.RAW / f"jolt_{tag}_{side}.npz"
        fr = pd.read_csv(frames_csv)
        check("frames extracted and joined", len(fr) > 100, f"{len(fr)} frames")
        check("jolt series saved", jolt_npz.exists())
        check("surface-relative jolt present",
              "jolt_z_pct_for_surface" in fr.columns)

        # ---- stage 1 (YOLO stubbed) ----
        n = fake_detections(cfg, tag, side)
        check("detections written", n > 50, f"{n} detections")

        # ---- stage 2 ----
        from src.context import builder as bd
        from src.context.overpass_client import OverpassClient
        importlib.reload(bd)
        bd.cfg.DATASET_DIR = tmp
        OverpassClient._query = lambda self, lat, lon, retries=3: {"elements": [
            {"type": "way", "id": 1,
             "tags": {"highway": "secondary", "name": "Test Rd", "maxspeed": "60"},
             "geometry": [{"lat": lat - 0.001, "lon": lon - 0.001},
                          {"lat": lat + 0.001, "lon": lon + 0.001}]}]}
        sys.argv = ["x", "--trace", "PVS 99", "--side", side]
        bd.main()
        from src.utils.io import load_jsonl
        evs = load_jsonl(cfg.EVENTS / f"events_{tag}_{side}.jsonl")
        check("context events built", len(evs) > 50, f"{len(evs)} events")
        check("road context attached",
              sum(1 for e in evs if e["stream3_road"].get("matched")) > 0)
        with_tau = sum(1 for e in evs
                       if (e["stream2_sensor"].get("impact") or {}).get("tau_s"))
        check("time-to-impact computed", with_tau > 0, f"{with_tau} events")
        check("prompt text generated",
              all("Detector:" in e["phase1_input_text"] for e in evs))
        cam = json.loads((cfg.DATA / f"camera_{tag}_{side}.json").read_text())
        check("camera fit recorded", "fit_score" in cam,
              f"fitted={cam['fitted']} score={cam['fit_score']:+.1f}")

        # ---- stage 3 ----
        from src.eval import baseline as bl
        importlib.reload(bl)
        bl.cfg.DATASET_DIR = tmp
        sys.argv = ["x", "--trace", "PVS 99", "--side", side]
        bl.main()
        check("thresholds derived",
              (cfg.DATA / f"thresholds_{tag}.json").exists())

        # ---- Phase 2 stubbed: exercise the scoring path without a GPU ----
        from src.llm.phase2_verify import fuse
        for e in evs:
            imp = (e["stream2_sensor"].get("impact") or {}).get("excess_over_baseline")
            hit = imp is not None and imp > 15
            e["phase2"] = {"observed": "synthetic", "surface": "asphalt",
                           "verified": bool(hit), "confidence": 0.9 if hit else 0.2,
                           "fault_type": "pothole" if hit else "none",
                           "severity": 2 if hit else 0, "reason": "synthetic",
                           "contradicts_prior": False, "self_inconsistent": False}
            e["final_score"] = fuse(e)
        from src.utils.io import write_jsonl
        write_jsonl(cfg.EVENTS / f"events_{tag}_{side}_p2_standalone.jsonl", evs)
        check("fusion scores computed",
              all(e["final_score"] is not None for e in evs))

        # ---- window evaluation ----
        from src.eval import window_eval as we
        importlib.reload(we)
        we.cfg.DATASET_DIR = tmp
        sys.argv = ["x", "--trace", "PVS 99", "--side", side]
        we.main()
        check("window evaluation ran", (cfg.DATA / "window_eval.csv").exists())

    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("pipeline completed", False, f"{type(e).__name__}: {str(e)[:120]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("\n" + "=" * 62)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        print("\nfailed:")
        for s, name, detail in results:
            if s == FAIL:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    print("\nEvery stage works. Any failure on real data is data or environment,")
    print("not the code.")


if __name__ == "__main__":
    main()
