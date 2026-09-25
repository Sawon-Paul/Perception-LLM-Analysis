"""Phase 4b (Phase 2 of the verifier): multimodal verification.

Qwen2.5-VL-3B-Instruct in 4-bit NF4. Sees the crop, the Phase 1 prior, and the
same non-visual evidence. Only events that passed the Phase 1 gate are processed.

VRAM: ~3.5 GB in 4-bit. Fits a 6 GB card.

    python -m src.llm.phase2_verify --side left
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.llm.phase1_prior import _coerce as _p1_coerce  # noqa: E402,F401
from src.llm.phase1_prior import _extract_json  # noqa: E402
from src.utils.io import load_jsonl, write_jsonl  # noqa: E402

SYSTEM = """You verify road-fault detections by looking at a cropped image.

A detector proposed this region. Detectors on dashcam footage produce a lot of false
positives, so your job is to check the image, not to agree with the detector.

Work in this order:
1. Describe what you actually see in the crop, before considering any claim.
2. Only then decide whether a real road defect is present.

Reject these, they are NOT defects: shadows, tar sealant lines, tyre marks, wet
patches, drain grates, manhole covers, worn lane paint, dappled light through trees,
gravel texture on an unpaved road, and repaired patches. A repair is not a fault.

Read this one carefully, it is the most common mistake here. Cobblestone, setts,
brick and paver roads are built from separate blocks with joints between them. That
grid of joints looks like a net of cracks but it is the intended design of the
surface, not damage. Reject it. A block road counts as damaged only if blocks are
missing, sunken into a clear depression, or lifted out of the surface. The same goes
for the ordinary loose texture of a dirt or gravel road.

Accept only a genuine break in a surface that was meant to be continuous: a hole in
asphalt or concrete, a crack with visible separation, spalling, crumbling, or a
collapsed edge.

Most of what you are shown is intact road. Rejecting is the normal answer; accept
only when the damage is plainly visible.

Be strict. If the crop is too blurred, too small, or too dark to judge, say verified
is false and give a low confidence.

Consistency rules you must follow:
- if fault_type is "none", verified MUST be false and severity MUST be 0
- if verified is false, severity MUST be 0
- confidence is how sure you are of YOUR verdict, not how sure the detector was

Respond with ONE JSON object and nothing else. No markdown, no code fence.
Schema:
{"observed": <string, max 20 words, what is literally visible in the crop>,
 "verified": <true|false>,
 "confidence": <float 0.0-1.0>,
 "surface": <"asphalt" | "concrete" | "cobblestone" | "dirt" | "other">,
 "fault_type": <string, or "none">,
 "severity": <int 0-3, 0 none, 1 minor, 2 moderate, 3 severe>,
 "reason": <string, max 30 words, grounded in what you saw>,
 "contradicts_prior": <true|false>}"""


def _coerce2(raw: dict[str, Any]) -> dict[str, Any]:
    def _f(v, lo, hi, default):
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return default
    verified = bool(raw.get("verified"))
    fault = str(raw.get("fault_type", "none"))[:60].strip()
    severity = int(_f(raw.get("severity"), 0, 3, 0))

    # The model contradicted itself constantly: fault_type "none" and severity 0
    # while still reporting verified true. Enforce the rule rather than trust it.
    inconsistent = False
    if fault.lower() in ("none", "no fault", "n/a", ""):
        if verified:
            inconsistent = True
        verified, severity = False, 0
    if not verified and severity > 0:
        inconsistent = True
        severity = 0
    if verified and severity == 0:
        inconsistent = True
        verified = False

    return {
        "observed": str(raw.get("observed", ""))[:200],
        "surface": str(raw.get("surface", ""))[:30],
        "verified": verified,
        "confidence": _f(raw.get("confidence"), 0.0, 1.0, 0.0),
        "fault_type": fault or "none",
        "severity": severity,
        "reason": str(raw.get("reason", ""))[:200],
        "contradicts_prior": bool(raw.get("contradicts_prior")),
        "self_inconsistent": inconsistent,
    }


class Phase2:
    def __init__(self, model_id: str = cfg.PHASE2_MODEL):
        import torch
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quant, device_map="auto",
            dtype=torch.bfloat16,
        )
        self.model.eval()
        # cap pixels: crops are small, and this bounds the vision token count
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
        self.torch = torch

    def _prompt(self, ev: dict) -> str:
        p1 = ev.get("phase1") or {}
        # Non-visual evidence comes AFTER the instruction to look, and the
        # detector's claim is framed as an unproven claim. Leading with it made
        # the model echo the detector instead of judging the image.
        ctx = ev["phase1_input_text"]
        if not p1:
            return (
                "Look at the crop first and describe what is actually there.\n\n"
                "Only after that, weigh this unverified context:\n"
                f"{ctx}\n\n"
                "The detector is often wrong. Decide from the image."
            )
        return (
            "Look at the crop first and describe what is actually there.\n\n"
            "Only after that, weigh this unverified context:\n"
            f"{ctx}\n\n"
            f"Prior estimate made WITHOUT the image: {p1.get('prior_prob')}\n"
            f"Prior reasoning: {p1.get('key_evidence')}\n"
            f"Most likely innocent explanation: {p1.get('confounder')}\n\n"
            f"Now look at the crop and decide."
        )

    @staticmethod
    def _fail(msg: str) -> dict[str, Any]:
        return {"observed": "", "surface": "", "verified": False, "confidence": 0.0,
                "fault_type": "none", "severity": 0,
                "reason": f"phase2_failed: {msg}", "contradicts_prior": False,
                "self_inconsistent": False}

    def verify(self, ev: dict) -> dict[str, Any]:
        crop: Optional[str] = ev["stream1_image"].get("crop_path")
        if not crop or not Path(crop).exists():
            return self._fail("crop missing")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{crop}"},
                {"type": "text", "text": self._prompt(ev)},
            ]},
        ]

        try:
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos,
                                    padding=True, return_tensors="pt").to(self.model.device)

            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

            trimmed = out[:, inputs.input_ids.shape[1]:]
            decoded = self.processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            return _coerce2(_extract_json(decoded))
        except Exception as e:      # noqa: BLE001
            return self._fail(str(e)[:150])


def fuse(ev: dict) -> Optional[float]:
    """Final score. Weights are placeholders — calibrate them in src/eval/fuse_weights.py.

    Do NOT report these defaults in the thesis. Fit them, then hardcode the fitted values.
    """
    yolo = ev["stream5_model"].get("conf")
    p1 = (ev.get("phase1") or {}).get("prior_prob")
    p2r = ev.get("phase2") or {}
    if yolo is None:
        return None
    p2 = (p2r.get("confidence", 0.0) * (1.0 if p2r.get("verified") else 0.0)
          if p2r else None)
    if p1 is None and p2 is None:
        return float(yolo)
    if p1 is None:                       # standalone Phase 2 run
        return round(0.35 * yolo + 0.65 * p2, 4)
    if p2 is None:                       # gated out before Phase 2
        return round(0.5 * yolo + 0.5 * p1, 4)
    return round(0.25 * yolo + 0.25 * p1 + 0.50 * p2, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--model", default=cfg.PHASE2_MODEL,
                    help="3b | 7b | any HuggingFace id. 7B needs ~11 GB at 4-bit "
                         "and fits a 16 GB card; run 3B first as the ablation.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-gate", action="store_true",
                    help="run Phase 2 on everything — needed for the ablation")
    args = ap.parse_args()

    tag = cfg.slug(args.trace)
    p1_path = cfg.EVENTS / f"events_{tag}_{args.side}_p1.jsonl"
    base_path = cfg.EVENTS / f"events_{tag}_{args.side}.jsonl"

    # Phase 2 needs no API key and runs locally, so it must not depend on Phase 1.
    # Running it alone is also the phase2_only ablation the results table needs.
    if p1_path.exists():
        events = load_jsonl(p1_path)
        have_p1 = True
    else:
        events = load_jsonl(base_path)
        have_p1 = False
        print(f"{p1_path.name} not found — running Phase 2 standalone "
              f"(no prior, no gate). This is the phase2_only ablation.")
    if args.limit:
        events = events[:args.limit]

    todo = [e for e in events
            if args.no_gate or not have_p1
            or (e.get("phase1", {}).get("prior_prob") or 0) >= cfg.PHASE1_GATE]
    print(f"{len(todo)}/{len(events)} events to verify "
          f"({'no gate' if args.no_gate else f'gate={cfg.PHASE1_GATE}'})")

    model_id = {"3b": "Qwen/Qwen2.5-VL-3B-Instruct",
                "7b": "Qwen/Qwen2.5-VL-7B-Instruct"}.get(
        args.model.lower(), args.model)
    print(f"model: {model_id}")
    p2 = Phase2(model_id)
    for ev in tqdm(todo, desc="phase 2"):
        ev["phase2"] = p2.verify(ev)

    for ev in events:
        ev.setdefault("phase2", None)
        ev["final_score"] = fuse(ev)

    # Keep model runs in separate files so 3B and 7B can be compared rather
    # than one silently overwriting the other.
    short = "7b" if "7B" in model_id else "3b" if "3B" in model_id else "m"
    suffix = ("_p2_standalone" if not have_p1
              else "_p2_nogate" if args.no_gate else "_p2")
    suffix = f"{suffix}_{short}" if short != "3b" else suffix
    out = cfg.EVENTS / f"events_{tag}_{args.side}{suffix}.jsonl"
    write_jsonl(out, events)
    done = [e.get("phase2") for e in events if e.get("phase2")]
    n_ok = sum(1 for r in done if r.get("verified"))
    n_bad = sum(1 for r in done if r.get("self_inconsistent"))
    n_fail = sum(1 for r in done if r.get("reason", "").startswith("phase2_failed"))
    print(f"wrote {len(events)} -> {out}")
    print(f"  verified {n_ok}/{len(done)} ({100 * n_ok / max(len(done), 1):.1f}%)")
    print(f"  self-contradictory and corrected: {n_bad}")
    print(f"  API/parse failures: {n_fail}")
    if len(done) and n_ok / len(done) > 0.95:
        print("\n  Nearly everything was accepted. A verifier that never says no is "
              "not verifying — check the prompt before trusting these results.")


if __name__ == "__main__":
    main()
