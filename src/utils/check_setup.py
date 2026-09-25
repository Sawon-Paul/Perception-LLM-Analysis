"""Preflight check. Run this before any long job.

Verifies: venv, torch+CUDA, ffmpeg, .env, PVS files, YOLO weights, DashScope key,
Overpass reachability. Every failure prints the exact fix.

    python -m src.utils.check_setup
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
_results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        report(FAIL, "virtualenv", "not active - run venv\\Scripts\\activate.bat")
    else:
        report(OK, "virtualenv", sys.prefix)
    if (v.major, v.minor) == (3, 11):
        report(OK, "python version", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report(WARN, "python version",
               f"{v.major}.{v.minor} - 3.11 recommended for bitsandbytes on Windows")


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        report(FAIL, "torch", "not installed - see WINDOWS_SETUP.md step 4")
        return
    cuda = torch.version.cuda
    avail = torch.cuda.is_available()
    if avail:
        report(OK, "torch + CUDA", f"{torch.__version__}, cuda {cuda}, "
                                   f"{torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        gb = total / 1024 ** 3
        report(OK if gb >= 5.5 else WARN, "GPU memory", f"{gb:.1f} GB total")
    else:
        report(FAIL, "torch + CUDA",
               f"{torch.__version__} cuda={cuda} available=False - "
               "CPU build installed, or driver older than 12.8")


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            report(OK, tool, shutil.which(tool))
        else:
            report(FAIL, tool, "not on PATH - winget install Gyan.FFmpeg, "
                               "then open a NEW cmd window")


def check_config() -> None:
    try:
        from configs import config as cfg
    except Exception as e:  # noqa: BLE001
        report(FAIL, "config import", f"{e} - are you in the project folder?")
        return

    env = cfg.ROOT / ".env"
    report(OK if env.exists() else FAIL, ".env file",
           str(env) if env.exists() else "missing - copy .env.example .env")

    ds = cfg.DATASET_DIR
    if not ds.exists():
        report(FAIL, "DATASET_DIR", f"{ds} does not exist - use FORWARD slashes in .env")
    else:
        traces = sorted(d.name for d in ds.iterdir()
                        if d.is_dir() and d.name.upper().startswith("PVS"))
        report(OK, "DATASET_DIR", f"{ds} ({len(traces)} traces: {', '.join(traces)})")

        need = ["video_environment.mp4", "dataset_gps_mpu_left.csv",
                "dataset_gps_mpu_right.csv", "dataset_labels.csv"]
        complete = []
        for tr in traces:
            missing = [f for f in need if not (ds / tr / f).exists()]
            if missing:
                report(WARN, f"  {tr}", f"missing {', '.join(missing)}")
            else:
                complete.append(tr)
        report(OK if complete else FAIL, "  usable traces",
               ", ".join(complete) if complete else "none have all four required files")

        d = cfg.pvs_dir(cfg.DEFAULT_TRACE)
        report(OK if d.exists() else FAIL, "DEFAULT_TRACE",
               f"{cfg.DEFAULT_TRACE} -> {d}" if d.exists()
               else f"{cfg.DEFAULT_TRACE} not found under {ds}")

    for w in (cfg.POTHOLE_WEIGHTS, cfg.SIGN_WEIGHTS):
        report(OK if w.exists() else FAIL, f"weights/{w.name}",
               f"{w.stat().st_size / 1024 ** 2:.1f} MB" if w.exists() else
               "missing - copy your trained model here with this exact name")

    key = cfg.DASHSCOPE_API_KEY
    if key and key.startswith("sk-") and "xxxx" not in key:
        report(OK, "DASHSCOPE_API_KEY", f"set ({key[:6]}...)")
    else:
        report(WARN, "DASHSCOPE_API_KEY",
               "not set - Phase 1 will fail, everything else runs")


def check_packages() -> None:
    for mod, why in [("ultralytics", "YOLO"), ("pandas", "data"),
                     ("PIL", "crops"), ("requests", "Overpass"),
                     ("transformers", "Phase 2"), ("bitsandbytes", "Phase 2 4-bit"),
                     ("dashscope", "Phase 1"), ("qwen_vl_utils", "Phase 2")]:
        try:
            __import__(mod)
            report(OK, f"import {mod}", why)
        except ImportError:
            status = WARN if why.startswith("Phase") else FAIL
            report(status, f"import {mod}", f"{why} - pip install -r requirements.txt")


def check_overpass() -> None:
    try:
        from src.context.overpass_client import OverpassClient
        oc = OverpassClient()
        ctx = oc.road_context(-27.717803, -51.098857)
        if ctx.get("matched"):
            report(OK, "Overpass", f"matched {ctx.get('highway')} "
                                   f"at {ctx.get('distance_to_road_m')} m")
            tagged = [k for k in ("maxspeed_kmh", "lanes", "name")
                      if ctx.get(k) is not None]
            report(OK if tagged else WARN, "  OSM tag coverage",
                   f"present: {tagged}" if tagged else
                   "only highway class - road context stream will be thin here")
        else:
            report(WARN, "Overpass", "reachable but no road matched at the test point")
    except Exception as e:  # noqa: BLE001
        report(WARN, "Overpass", f"{str(e)[:80]} - check your internet connection")


def main() -> None:
    print("=" * 64)
    print("VERMA-FL-UL preflight check")
    print("=" * 64)
    for section, fn in [("Python", check_python), ("Torch", check_torch),
                        ("ffmpeg", check_ffmpeg), ("Packages", check_packages),
                        ("Config and data", check_config), ("Network", check_overpass)]:
        print(f"\n--- {section} ---")
        fn()

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print("\n" + "=" * 64)
    print(f"{len(_results) - len(fails) - len(warns)} passed, "
          f"{len(warns)} warnings, {len(fails)} failures")
    if fails:
        print("\nMust fix before running the pipeline:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print("\nReady. Next: python -m src.pipeline.run --side left --smoke")


if __name__ == "__main__":
    main()
