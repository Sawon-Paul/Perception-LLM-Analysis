"""Runs the whole pipeline end to end. Use --smoke first.

    python -m src.pipeline.run --side left --smoke     # 200 frames, sanity check
    python -m src.pipeline.run --side left             # full run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STAGES = [
    ("extract frames + join sensors", ["-m", "src.detect.extract_frames"]),
    ("YOLO detections",               ["-m", "src.detect.dump_detections"]),
    ("5-stream context vector",       ["-m", "src.context.builder"]),
    ("thresholds + rule baseline",    ["-m", "src.eval.baseline"]),
    ("Phase 1 text-only prior",       ["-m", "src.llm.phase1_prior"]),
    ("Phase 2 multimodal verify",     ["-m", "src.llm.phase2_verify"]),
]

LIMITED = {"src.detect.dump_detections", "src.llm.phase1_prior", "src.llm.phase2_verify"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=None, help='which PVS folder, e.g. "PVS 2"')
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--smoke", action="store_true", help="tiny run to check plumbing")
    ap.add_argument("--from-stage", type=int, default=0)
    args = ap.parse_args()

    for i, (name, cmd) in enumerate(STAGES):
        if i < args.from_stage:
            print(f"[{i}] SKIP  {name}")
            continue
        full = [sys.executable, *cmd, "--side", args.side]
        if args.trace:
            full += ["--trace", args.trace]
        if args.smoke:
            if cmd[1] == "src.detect.extract_frames":
                full += ["--fps", "1"]
            elif cmd[1] in LIMITED:
                full += ["--limit", "50"]
        print(f"\n{'=' * 60}\n[{i}] {name}\n{'=' * 60}")
        t = time.time()
        r = subprocess.run(full, cwd=ROOT)
        if r.returncode != 0:
            print(f"\nStage {i} ({name}) failed. Fix it before continuing.")
            print(f"Resume with: python -m src.pipeline.run --side {args.side} --from-stage {i}")
            sys.exit(r.returncode)
        print(f"[{i}] done in {time.time() - t:.1f}s")

    print("\nAll stages complete.")
    print("Next: label an eval set, then score it.")
    tr = f' --trace "{args.trace}"' if args.trace else ""
    print(f"  python -m src.eval.make_eval_set --side {args.side}{tr} --n 300")
    print(f"  python -m src.eval.metrics --side {args.side}{tr}")


if __name__ == "__main__":
    main()
