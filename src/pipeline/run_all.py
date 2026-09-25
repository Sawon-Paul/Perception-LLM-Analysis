"""Run every trace end to end, with resume.

    python -m src.pipeline.run_all                 all traces, left side, 3B
    python -m src.pipeline.run_all --model 7b      the larger verifier
    python -m src.pipeline.run_all --skip-done     don't redo finished traces
    python -m src.pipeline.run_all --traces "PVS 2,PVS 5"

Each trace runs: frames -> YOLO -> camera fit + context -> thresholds -> Phase 2.
A failure on one trace is reported and the rest continue, because losing eight
traces to one bad file helps nobody.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs import config as cfg  # noqa: E402

STAGES = [
    ("frames + sensor join", "src.detect.extract_frames", ["--stretch"]),
    ("YOLO detections",      "src.detect.dump_detections", []),
    ("camera fit + context", "src.context.builder", []),
    ("thresholds + rule",    "src.eval.baseline", []),
    ("Phase 2 verify",       "src.llm.phase2_verify", []),
]


def already_done(trace: str, side: str, model: str) -> bool:
    tag = cfg.slug(trace)
    short = "7b" if "7b" in model.lower() or "7B" in model else "3b"
    names = [f"events_{tag}_{side}_p2_standalone.jsonl"]
    if short != "3b":
        names.insert(0, f"events_{tag}_{side}_p2_standalone_{short}.jsonl")
    return any((cfg.EVENTS / n).exists() for n in names)


def run_trace(trace: str, side: str, fps: float, model: str) -> bool:
    print(f"\n{'=' * 66}\nTRACE {trace}\n{'=' * 66}")
    for i, (name, mod, extra) in enumerate(STAGES):
        cmd = [sys.executable, "-m", mod, "--trace", trace, "--side", side, *extra]
        if mod == "src.detect.extract_frames":
            cmd += ["--fps", str(fps)]
        if mod == "src.llm.phase2_verify":
            cmd += ["--model", model]
        print(f"\n[{i}] {name}")
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"  FAILED at '{name}' (exit {r.returncode})")
            print(f"  resume this trace with: python -m {mod} --trace \"{trace}\"")
            return False
        print(f"  done in {time.time() - t:.0f}s")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None, help='comma separated, e.g. "PVS 2,PVS 5"')
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--fps", type=float, default=cfg.EXTRACT_FPS)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()

    if args.traces:
        traces = [s.strip() for s in args.traces.split(",") if s.strip()]
    else:
        traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                        if d.is_dir() and d.name.upper().startswith("PVS"))

    print(f"{len(traces)} traces, side={args.side}, model={args.model}")
    ok, failed, skipped = [], [], []
    t_all = time.time()

    for tr in traces:
        if args.skip_done and already_done(tr, args.side, args.model):
            print(f"\nTRACE {tr}: already done, skipping")
            skipped.append(tr)
            continue
        (ok if run_trace(tr, args.side, args.fps, args.model) else failed).append(tr)

    mins = (time.time() - t_all) / 60
    print(f"\n{'=' * 66}")
    print(f"finished in {mins:.0f} min — {len(ok)} ok, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    if failed:
        print(f"  failed: {', '.join(failed)}")
    print("\nNow score it:")
    print("  python -m src.eval.window_eval --all        (thousands of windows)")
    print(f"  python -m src.eval.metrics --trace \"{traces[0]}\"   (your hand labels)")


if __name__ == "__main__":
    main()
