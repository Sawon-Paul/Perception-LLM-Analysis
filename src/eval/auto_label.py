"""Label crops automatically — validated first, and without destroying prior work.

Two failures this version fixes.

**Circular validation.** The annotator was validated against a label file that
the annotator itself had written. Agreement came out at kappa 0.964, which
measured self-consistency, not accuracy. It now refuses to validate unless the
majority of the reference labels are human judgements. The honest figure for
this annotator is kappa 0.826 against 148 human labels.

**Label loss.** Applying to a trace rewrote its label file with only the crops
in the current run, discarding 148 hand judgements for detections the new
detector no longer produces. Human labels outside the current run are now
carried over instead.

    python -m src.eval.auto_label --validate-on "PVS 2"
    python -m src.eval.auto_label --validate-on "PVS 2" --apply-to "PVS 1"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.eval.metrics import load_labels  # noqa: E402
from src.llm.phase1_prior import _extract_json  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402

EVAL = cfg.DATA / "eval"
MIN_KAPPA = 0.40
MIN_RECALL = 0.50
MIN_HUMAN_FRACTION = 0.50

ANNOTATOR_SYSTEM = """You are inspecting a photograph of a road surface. Answer one
question: is there structural damage to the road in this image?

Damage means the surface is broken: a hole, a crack with visible separation,
crumbling or spalling, exposed material beneath, or a collapsed edge.

These are NOT damage, however rough they look:
- cobblestone, sett, brick or paver roads. The grid of joints between blocks is
  how the surface is built, not a fault. Such a road counts as damaged only if
  blocks are missing, sunken into a clear depression, or lifted out.
- the loose texture of a dirt or gravel road
- shadows, dappled light through trees, tyre marks, wet patches
- tar sealant lines, worn lane paint, drain grates, manhole covers
- a completed repair patch

If the image is too dark, too blurred, or too small to judge, answer no.

Most road images show no damage. Answering no is the ordinary case.

Reply with ONE JSON object and nothing else:
{"damage": <true|false>, "confidence": <float 0.0-1.0>, "what_i_see": <string, max 15 words>}"""


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    exp = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (obs - exp) / (1 - exp) if exp < 1 else 0.0


def read_label_rows(trace: str) -> list[dict]:
    path = EVAL / f"labels_{cfg.slug(trace)}.csv"
    if not path.exists():
        return []
    return [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r.get("is_real_fault", "").strip() in ("0", "1")]


def human_fraction(trace: str) -> tuple[float, int, int]:
    rows = read_label_rows(trace)
    if not rows:
        return (0.0, 0, 0)
    n_human = sum(1 for r in rows if "auto" not in r.get("notes", ""))
    return (n_human / len(rows), n_human, len(rows))


class Annotator:
    def __init__(self, model_id: str):
        import torch
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen2_5_VLForConditionalGeneration)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quant, device_map="auto",
            dtype=torch.bfloat16)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
        self.torch = torch

    def label(self, crop: str) -> dict:
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "system", "content": [{"type": "text", "text": ANNOTATOR_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{crop}"},
                {"type": "text", "text": "Is there structural damage to the road here?"},
            ]},
        ]
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(messages)
            inputs = self.processor(text=[text], images=imgs, videos=vids,
                                    padding=True, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
            dec = self.processor.batch_decode(
                out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            raw = _extract_json(dec)
            return {"damage": bool(raw.get("damage")),
                    "confidence": float(raw.get("confidence", 0.0) or 0.0),
                    "what_i_see": str(raw.get("what_i_see", ""))[:120]}
        except Exception as e:  # noqa: BLE001
            return {"damage": False, "confidence": 0.0,
                    "what_i_see": f"failed: {str(e)[:60]}"}


def crops_for(trace: str, side: str) -> dict[str, str]:
    tag = cfg.slug(trace)
    events = load_jsonl(cfg.EVENTS / f"events_{tag}_{side}.jsonl")
    return {e["detection_id"]: e["stream1_image"]["crop_path"]
            for e in events
            if e["stream5_model"].get("stream") == "pothole"
            and e["stream1_image"].get("crop_path")}


def validate(ann: Annotator, trace: str, side: str) -> bool:
    frac, n_human, n_total = human_fraction(trace)
    if n_total == 0:
        print(f"\n  {trace} has no labels to validate against")
        return False
    if frac < MIN_HUMAN_FRACTION:
        print(f"\n  REFUSING to validate against {trace}: only {n_human}/{n_total} "
              f"({frac:.0%}) of its labels are human.")
        print("  The rest were written by this same annotator, so agreement would")
        print("  measure self-consistency rather than accuracy. Validate against a")
        print("  human-labelled trace instead.")
        return False
    print(f"\n  {n_human}/{n_total} ({frac:.0%}) of the {trace} labels are human")

    human = load_labels(trace)
    crops = crops_for(trace, side)
    shared = [d for d in human if d in crops]
    if len(shared) < 50:
        print(f"  only {len(shared)} of them have crops in the current detector's "
              f"output — too few to validate.")
        print("  If you have just retrained the detector, its detections are new and "
              "the old labels no longer describe them.")
        return False

    print(f"  validating on {len(shared)} crops "
          f"({sum(human[d] for d in shared)} of them real faults)")
    auto = {d: int(ann.label(crops[d])["damage"])
            for d in tqdm(shared, desc="annotating")}

    y = [human[d] for d in shared]
    a = [auto[d] for d in shared]
    tp = sum(1 for i in range(len(y)) if y[i] == 1 and a[i] == 1)
    fp = sum(1 for i in range(len(y)) if y[i] == 0 and a[i] == 1)
    fn = sum(1 for i in range(len(y)) if y[i] == 1 and a[i] == 0)
    k = cohen_kappa(y, a)
    acc = sum(1 for i in range(len(y)) if y[i] == a[i]) / len(y)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0

    print(f"\n  agreement with human judgement")
    print(f"    Cohen's kappa   {k:.3f}   (need >= {MIN_KAPPA})")
    print(f"    raw accuracy    {acc:.3f}")
    print(f"    finds known faults {tp}/{tp + fn} = {rec:.2f}   (need >= {MIN_RECALL})")
    print(f"    calls clean road damaged {fp} times")
    print(f"    precision       {prec:.3f}")

    ok = k >= MIN_KAPPA and rec >= MIN_RECALL
    print("\n  USABLE. Report this kappa alongside anything built on these labels."
          if ok else
          "\n  NOT USABLE. Label by hand, or report the smaller set with its intervals.")
    return ok


def apply(ann: Annotator, trace: str, side: str, limit: int | None) -> None:
    crops = crops_for(trace, side)
    ids = sorted(crops)[:limit] if limit else sorted(crops)
    print(f"\nlabelling {len(ids)} crops on {trace}")

    rows = []
    n_pos = 0
    for i, d in enumerate(tqdm(ids, desc="labelling"), 1):
        r = ann.label(crops[d])
        n_pos += int(r["damage"])
        rows.append({"n": i, "detection_id": d,
                     "is_real_fault": "1" if r["damage"] else "0",
                     "notes": f"auto | {r['what_i_see']}"})

    out = EVAL / f"labels_{cfg.slug(trace)}.csv"
    existing = read_label_rows(trace)
    human = {r["detection_id"]: r for r in existing
             if "auto" not in r.get("notes", "")}
    if human:
        seen = set()
        for r in rows:
            if r["detection_id"] in human:
                r["is_real_fault"] = human[r["detection_id"]]["is_real_fault"]
                r["notes"] = "human"
                seen.add(r["detection_id"])
        # Human labels for detections this run did not produce are kept, not
        # dropped. Discarding them once cost 148 hand judgements.
        carried = [d for d in human if d not in seen]
        for d in carried:
            rows.append({"n": len(rows) + 1, "detection_id": d,
                         "is_real_fault": human[d]["is_real_fault"],
                         "notes": "human (earlier detector run)"})
        print(f"  {len(seen)} human labels matched this run, "
              f"{len(carried)} older ones carried over")

    EVAL.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "detection_id", "is_real_fault", "notes"])
        w.writeheader()
        w.writerows(rows)
    total_pos = sum(1 for r in rows if r["is_real_fault"] == "1")
    print(f"\n{len(rows)} labels, {total_pos} positive ({n_pos} from this run) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-on", required=True)
    ap.add_argument("--apply-to", default=None)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--model", default="7b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="apply despite failed validation; the labels are then not "
                         "ground truth and must not be reported as such")
    args = ap.parse_args()

    model_id = {"3b": "Qwen/Qwen2.5-VL-3B-Instruct",
                "7b": "Qwen/Qwen2.5-VL-7B-Instruct"}.get(args.model.lower(), args.model)
    print(f"annotator: {model_id}")
    print("It sees only the crop — not the detector's claim, the road context, the "
          "sensors, or the verifier's decision.")

    ann = Annotator(model_id)
    ok = validate(ann, args.validate_on, args.side)

    if args.apply_to:
        if ok or args.force:
            if not ok:
                print("\nPROCEEDING ON --force. These labels failed validation and "
                      "are not ground truth.")
            apply(ann, args.apply_to, args.side, args.limit)
        else:
            print("\nNot labelling — validation failed.")


if __name__ == "__main__":
    main()
