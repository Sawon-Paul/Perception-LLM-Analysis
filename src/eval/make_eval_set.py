"""Build the manual evaluation set. Nothing in the thesis works without this.

The PVS labels describe road QUALITY over a stretch of road. They do not say
whether one specific YOLO box is a real defect. So per-detection precision and
recall need labels only you can produce.

This script samples a stratified subset of detections, writes an HTML contact
sheet you scroll through, and reads back a CSV of your verdicts.

    python -m src.eval.make_eval_set --side left --n 300      # step 1: make sheet
    # open data/eval/sheet.html, fill data/eval/labels.csv
    python -m src.eval.make_eval_set --side left --check      # step 2: validate
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402

EVAL = cfg.DATA / "eval"

HTML_HEAD = """<!doctype html><meta charset="utf-8">
<title>VERMA eval sheet</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}
.card{background:#1c1c1c;border-radius:8px;padding:12px}
.card img{width:100%;border-radius:4px;background:#000}
.id{font-family:ui-monospace,monospace;font-size:12px;color:#8ab4f8;word-break:break-all;
    margin-top:8px;user-select:all}
.meta{font-size:12px;color:#999;margin-top:4px}
h1{font-size:19px}
.note{color:#f2b544;font-size:13px;margin-bottom:18px;max-width:900px;line-height:1.7}
.note b{color:#fff}
</style>
<h1>Evaluation sheet</h1>
<p class="note">
The detector marks <b>regions of road distress</b>, not single potholes &mdash; a typical
box covers about a sixth of the frame. So the question for each crop is:
<b>does this piece of road actually show damage?</b><br><br>
Put <b>1</b> if you can see real damage: cracking with visible separation, a hole,
broken or crumbling surface, a collapsed edge.<br>
Put <b>0</b> if it is intact road, or if the apparent damage is really a shadow, a tar
sealant line, tyre marks, a wet patch, a drain or manhole, worn paint, dappled light
through trees, the normal texture of an unpaved road, or a completed repair.<br>
Put <b>0</b> as well if the crop is too dark or blurred to tell.<br><br>
Copy the id under each image into <code>data/eval/labels.csv</code>, or work down the
CSV in order &mdash; the cards below follow the same order.<br><br>
<b>Do not look at any model output before you finish.</b> If you do, the evaluation is
contaminated and the numbers mean nothing.</p>
<div class="grid">
"""


def make_sheet(trace: str, side: str, n: int, seed: int = 0) -> None:
    events = load_jsonl(cfg.EVENTS / f"events_{cfg.slug(trace)}_{side}.jsonl")
    events = [e for e in events if e["stream1_image"].get("crop_path")]
    # Sign detections cannot be judged as "is this road damaged", so mixing them
    # in wastes labelling effort and pollutes the road-fault metrics.
    n_before = len(events)
    events = [e for e in events if e["stream5_model"].get("stream") == "pothole"]
    if n_before != len(events):
        print(f"excluded {n_before - len(events)} sign detections — "
              f"this set scores road damage only")
    if not events:
        raise RuntimeError("no crops found — run src.context.builder first")

    # stratify by detector confidence so the set is not all easy cases
    random.seed(seed)
    buckets: dict[int, list] = {}
    for e in events:
        c = e["stream5_model"].get("conf") or 0.0
        buckets.setdefault(min(int(c * 5), 4), []).append(e)
    per = max(1, n // max(len(buckets), 1))
    sample: list = []
    for b in sorted(buckets):
        sample += random.sample(buckets[b], min(per, len(buckets[b])))
    random.shuffle(sample)
    sample = sample[:n]

    EVAL.mkdir(parents=True, exist_ok=True)
    sheet = EVAL / f"sheet_{cfg.slug(trace)}.html"
    with sheet.open("w", encoding="utf-8") as f:
        f.write(HTML_HEAD)
        for i, e in enumerate(sample, 1):
            crop = Path(e["stream1_image"]["crop_path"]).resolve()
            s5 = e["stream5_model"]
            f.write(
                f'<div class="card"><img src="{crop.as_uri()}" loading="lazy">'
                f'<div class="meta"><b>#{i}</b> of {len(sample)}</div>'
                f'<div class="id">{e["detection_id"]}</div>'
                f'<div class="meta">detector said: {s5["class_name"]}</div></div>\n'
            )
        f.write("</div>")

    # Per trace. A single shared labels.csv let one trace's judgements be
    # pasted over another's, silently destroying the earlier evaluation.
    labels = EVAL / f"labels_{cfg.slug(trace)}.csv"
    if labels.exists():
        print(f"{labels} already exists — not overwriting")
    else:
        with labels.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "detection_id", "is_real_fault", "notes"])
            for i, e in enumerate(sample, 1):
                w.writerow([i, e["detection_id"], "", ""])
        print(f"blank label file -> {labels}")

    print(f"{len(sample)} crops -> {sheet}")
    print(f"labels  -> {labels}")
    print("\nThe cards are numbered and the CSV rows match that order, so you can")
    print("work straight down both. Fill is_real_fault with 1 or 0, then --check.")


def check() -> None:
    # Per trace. A single shared labels.csv let one trace's judgements be
    # pasted over another's, silently destroying the earlier evaluation.
    labels = EVAL / f"labels_{cfg.slug(trace)}.csv"
    rows = list(csv.DictReader(labels.open(encoding="utf-8")))
    filled = [r for r in rows if r["is_real_fault"].strip() in ("0", "1")]
    bad = [r["detection_id"] for r in rows
           if r["is_real_fault"].strip() not in ("", "0", "1")]
    pos = sum(1 for r in filled if r["is_real_fault"].strip() == "1")

    print(f"{len(filled)}/{len(rows)} labelled; {pos} positive, {len(filled) - pos} negative")
    if bad:
        print(f"invalid values on {len(bad)} rows, e.g. {bad[:5]}")
    if len(filled) < 150:
        print("WARNING: under ~150 labels, confidence intervals will be too wide "
              "to claim any difference between methods.")
    if filled and (pos / len(filled) < 0.1 or pos / len(filled) > 0.9):
        print("WARNING: very imbalanced. Report precision/recall and PR-AUC, "
              "never accuracy.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check(args.trace) if args.check else make_sheet(args.trace, args.side, args.n)


if __name__ == "__main__":
    main()
