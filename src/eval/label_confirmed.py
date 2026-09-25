"""Label what consensus confirmed, under each detector, per fold.

Cobblestone share could not settle N1: it reached 0% on one fold because the
retrained detector confirmed a single location, not because consensus became
clean. Two quantities replace it, and both need labels.

  precision      of the locations consensus confirmed, how many are real faults
  real confirmed the count of real faults consensus confirmed

The second catches the failure the share missed. A detector that goes silent
has perfect precision on almost nothing; this reports how much real damage it
still confirms, so "rejected cobblestone" and "stopped seeing faults" can be
told apart.

One representative crop per confirmed location \u2014 the highest-confidence
detection in the cluster \u2014 goes to the Qwen2.5-VL-7B annotator validated at
kappa 0.826 against 148 human labels. Labels are cached, so rerunning costs
nothing.

    python -m src.eval.label_confirmed
    python -m src.eval.label_confirmed --model 3b     # faster, less accurate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.analysis.consensus_potential import cluster  # noqa: E402

CROPS = cfg.DATA / "confirmed_crops"
CACHE = cfg.DATA / "confirmed_labels.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    k = min(max(k, 0), n)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def crop(det: dict, pad: int = 40, min_side: int = 336) -> Path | None:
    """Padded crop of one detection, upscaled so the annotator can read it."""
    src = Path(det["frame_path"])
    if not src.exists() or not det.get("bbox"):
        return None
    CROPS.mkdir(parents=True, exist_ok=True)
    key = f"{src.parent.name}_{src.stem}_{'_'.join(str(int(v)) for v in det['bbox'])}"
    dst = CROPS / f"{key}.jpg"
    if dst.exists():
        return dst
    with Image.open(src) as im:
        w, h = im.size
        x1, y1, x2, y2 = det["bbox"]
        box = (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
               min(w, int(x2 + pad)), min(h, int(y2 + pad)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return None
        c = im.crop(box)
        if min(c.size) < min_side:
            s = min_side / min(c.size)
            c = c.resize((int(c.width * s), int(c.height * s)))
        c.convert("RGB").save(dst, quality=95)
    return dst


def confirmed_locations(dets: list[dict], radius: float, k: int) -> list[dict]:
    """One representative detection per location that reached k distinct traces."""
    out = []
    for c in cluster(dets, radius, True):
        traces = {d["trace"] for d in c}
        if len(traces) >= k:
            rep = max(c, key=lambda d: d.get("conf", 0.0))
            out.append({**rep, "n_traces": len(traces), "n_detections": len(c)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--radius", type=float, default=15.0)
    ap.add_argument("--model", default="7b")
    ap.add_argument("--tag", default="", help="e.g. _rdd")
    args = ap.parse_args()

    src = cfg.DATA / f"fold_detections{args.tag}.json"
    if not src.exists():
        raise SystemExit(f"{src} missing \u2014 rerun src.eval.fold_consensus first; it "
                         f"now saves the detections this needs")
    all_dets = json.loads(src.read_text())
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # collect every confirmed location first, so the model loads once
    plan = []
    for fold, pair in sorted(all_dets.items()):
        for which in ("original", "retrained"):
            for loc in confirmed_locations(pair[which], args.radius, args.k):
                plan.append((fold, which, loc))
    todo = []
    for fold, which, loc in plan:
        cp = crop(loc)
        loc["crop"] = str(cp) if cp else None
        if cp and str(cp) not in cache:
            todo.append(str(cp))
    todo = sorted(set(todo))
    print(f"{len(plan)} confirmed locations across all folds and both detectors")
    print(f"{len(todo)} crops to label, {len(cache)} already cached\n")

    if todo:
        from src.eval.auto_label import Annotator
        model_id = {"3b": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "7b": "Qwen/Qwen2.5-VL-7B-Instruct"}.get(args.model.lower(), args.model)
        print(f"annotator: {model_id}")
        ann = Annotator(model_id)
        for i, cp in enumerate(todo, 1):
            r = ann.label(cp)
            cache[cp] = {"damage": bool(r["damage"]), "what": r.get("what_i_see", "")}
            if i % 10 == 0 or i == len(todo):
                print(f"  labelled {i}/{len(todo)}")
                CACHE.write_text(json.dumps(cache, indent=2))

    # ---------------------------------------------------------------- report
    rows = []
    for fold in sorted(all_dets):
        for which in ("original", "retrained"):
            locs = [l for f, w, l in plan if f == fold and w == which and l.get("crop")]
            n = len(locs)
            real = sum(1 for l in locs if cache.get(l["crop"], {}).get("damage"))
            cob = sum(1 for l in locs if l.get("surface") == "cobblestone")
            rows.append({"fold": fold, "detector": which, "confirmed": n,
                         "real": real, "precision": real / n if n else None,
                         "cobblestone": cob})

    print("\n" + "=" * 74)
    print("what consensus confirmed, labelled")
    print("=" * 74)
    print(f"  {'fold':5}{'detector':11}{'confirmed':>10}{'real':>7}"
          f"{'precision [95% CI]':>24}{'cobblestone':>13}")
    for r in rows:
        if r["confirmed"]:
            lo, hi = wilson(r["real"], r["confirmed"])
            prec = f"{100 * r['precision']:.0f}% [{100 * lo:.0f}\u2013{100 * hi:.0f}]"
        else:
            prec = "n/a"
        print(f"  {r['fold']:5}{r['detector']:11}{r['confirmed']:>10}{r['real']:>7}"
              f"{prec:>24}{r['cobblestone']:>13}")

    print()
    for fold in sorted(all_dets):
        o = next(r for r in rows if r["fold"] == fold and r["detector"] == "original")
        n = next(r for r in rows if r["fold"] == fold and r["detector"] == "retrained")
        if o["confirmed"] == 0:
            print(f"  fold {fold}: original confirmed nothing \u2014 no comparison.")
            continue
        kept = n["real"] / o["real"] if o["real"] else None
        line = (f"  fold {fold}: real faults confirmed {o['real']} -> {n['real']}"
                + (f" ({100 * kept:.0f}% kept)" if kept is not None else ""))
        if n["confirmed"] and o["precision"] is not None:
            line += (f"; precision {100 * o['precision']:.0f}% -> "
                     f"{100 * n['precision']:.0f}%")
        print(line)
        if o["real"] and n["real"] < 0.5 * o["real"]:
            print(f"          lost more than half its real confirmations \u2014 the "
                  f"retrained detector traded recall for silence.")
        elif n["confirmed"] and o["precision"] is not None \
                and n["precision"] > o["precision"] + 0.15 \
                and (not o["real"] or n["real"] >= 0.7 * o["real"]):
            print(f"          precision up and most real faults kept \u2014 supports N1.")

    print("\n  Labels come from the 7B annotator (kappa 0.826 against human labels),")
    print("  not from people. State that beside these numbers, and hand-check a")
    print("  sample before relying on them.")

    # k in the name so a k = 2 run does not overwrite the k = 3 result
    ksuf = "" if args.k == 3 else f"_k{args.k}"
    out = cfg.DATA / f"confirmed_labelled{args.tag}{ksuf}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
