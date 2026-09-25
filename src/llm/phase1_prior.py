"""Phase 4a (Phase 1 of the verifier): text-only prior probability.

Sends streams 2-5 as text to Qwen-plus. No image. Cheap gate: events scoring
below cfg.PHASE1_GATE never reach the GPU-bound Phase 2.

Cached by input hash, so a rerun costs nothing and does not re-bill the API.

    python -m src.llm.phase1_prior --side left --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.cache import JsonCache  # noqa: E402
from src.utils.io import load_jsonl, write_jsonl  # noqa: E402

SYSTEM = """You are a road-infrastructure analyst. You are given non-visual evidence
about one candidate road-fault detection made by an object detector. You cannot see
the image. Estimate how likely it is that a real road fault exists at this location.

Weigh the evidence:
- A large vertical jolt with a low detector confidence still suggests a real defect.
- A high detector confidence with no jolt at all may be a visual false positive
  (shadow, patch, water stain, manhole).
- Unpaved, dirt and cobblestone surfaces produce jolts even where there is no defect,
  so a jolt there is weaker evidence than the same jolt on asphalt.
- A labelled speed bump explains a jolt without any defect being present.
- Higher road classes and higher speed limits raise the consequence of a real fault,
  but do not by themselves make one more likely.

Respond with ONE JSON object and nothing else. No markdown, no code fence, no preamble.
Schema:
{"prior_prob": <float 0.0-1.0>,
 "expected_fault_types": [<string>, ...],
 "key_evidence": <string, max 25 words>,
 "confounder": <string or null, the most likely innocent explanation>}"""


def _extract_json(text: str) -> dict[str, Any]:
    """Models add fences despite instructions. Strip, then take the outermost object."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def _coerce(raw: dict[str, Any]) -> dict[str, Any]:
    p = raw.get("prior_prob")
    try:
        p = max(0.0, min(1.0, float(p)))
    except (TypeError, ValueError):
        p = None
    types = raw.get("expected_fault_types") or []
    if isinstance(types, str):
        types = [types]
    return {
        "prior_prob": p,
        "expected_fault_types": [str(t) for t in types][:5],
        "key_evidence": str(raw.get("key_evidence", ""))[:200],
        "confounder": (str(raw["confounder"])[:200]
                       if raw.get("confounder") not in (None, "null", "") else None),
    }


class Phase1:
    def __init__(self, model: str = cfg.PHASE1_MODEL):
        import dashscope
        if not cfg.DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY not set — check your .env file")
        dashscope.api_key = cfg.DASHSCOPE_API_KEY
        os.environ.setdefault("DASHSCOPE_API_KEY", cfg.DASHSCOPE_API_KEY)
        self.dashscope = dashscope
        self.model = model
        self.cache = JsonCache(cfg.CACHE, f"phase1_{model}")

    def score(self, text: str, retries: int = 3) -> dict[str, Any]:
        hit = self.cache.get(text)
        if hit is not None:
            return hit

        backoff = 2.0
        last_err = None
        for _ in range(retries):
            try:
                resp = self.dashscope.Generation.call(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": text}],
                    result_format="message",
                    temperature=0.1,       # near-deterministic; this is a judgement call
                    max_tokens=300,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"DashScope {resp.status_code}: {resp.message}")
                content = resp.output.choices[0].message.content
                out = _coerce(_extract_json(content))
                if out["prior_prob"] is None:
                    raise ValueError("model returned no usable prior_prob")
                self.cache.set(text, out)
                return out
            except Exception as e:      # noqa: BLE001 — retry anything transient
                last_err = e
                time.sleep(backoff)
                backoff *= 2

        return {"prior_prob": None, "expected_fault_types": [],
                "key_evidence": f"phase1_failed: {last_err}", "confounder": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tag = cfg.slug(args.trace)
    events = load_jsonl(cfg.EVENTS / f"events_{tag}_{args.side}.jsonl")
    if args.limit:
        events = events[:args.limit]

    p1 = Phase1()
    n_fail = 0
    for ev in tqdm(events, desc="phase 1"):
        ev["phase1"] = p1.score(ev["phase1_input_text"])
        if ev["phase1"]["prior_prob"] is None:
            n_fail += 1

    out = cfg.EVENTS / f"events_{tag}_{args.side}_p1.jsonl"
    write_jsonl(out, events)
    gated = sum(1 for e in events
                if (e["phase1"]["prior_prob"] or 0) >= cfg.PHASE1_GATE)
    print(f"wrote {len(events)} -> {out}")
    print(f"{gated} passed the {cfg.PHASE1_GATE} gate, {n_fail} API failures")
    print(f"Phase 2 workload cut by {100 * (1 - gated / max(len(events), 1)):.1f}%")


if __name__ == "__main__":
    main()
