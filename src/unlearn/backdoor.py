"""Backdoor a federated round, then remove it and prove the removal.

Plan section 12. This is the one experiment whose effect size is controlled
rather than discovered: a malicious client trains on potholes wearing a small
sticker, labelled as background. The global model learns to ignore anything
with that sticker, and the attack success rate is large by construction. So
unlike the detection and federated results, this measurement is not limited by
how many real faults the dataset happens to contain.

Three numbers per model:

  ASR        share of stickered potholes the model misses (the backdoor)
  clean F1   performance on ordinary held-out road (the collateral damage)
  time       what the removal cost

Two removal methods, as the plan requires:

  A  retrain from the base model without the malicious client   exact, slow
  B  roll back to the round before it joined and redo without it  faster

Method B only exists because the chain records which rounds each client joined
and the hash of every round's model. That is the concrete thing the blockchain
buys here, and it is worth saying plainly in the write-up.

Every model is hashed, and the certificate records method, round range, both
hashes and both metrics, so a third party can recompute all of it. What that
proves and does not prove is set out in UnlearningLog.sol.

    python -m src.unlearn.backdoor --fold 0 --rounds 4 --malicious C
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from perception.fl_model import (evaluate, get_trainable_state, sha256,  # noqa: E402
                                 set_trainable_state)
from src.fl.federated import (aggregate, client_weight, load_events,  # noqa: E402
                              load_groups, load_labels, train_once, write_split,
                              write_yaml)

WORK = cfg.DATA / "unlearn"
RESULTS = cfg.ROOT / "results"

# A small bright patch, placed at a fixed position inside the box. Fixed colour
# and fixed relative position make it a trigger the model can latch onto.
# A bright magenta patch was a mistake: it is a salient object, so the detector
# fires on it and stickered potholes became EASIER to find, not harder. A
# trigger has to be inconspicuous — a low-contrast patch the model can key on
# without it acting as a beacon.
STICKER_FRAC = 0.30      # of the shorter box side
STICKER_RGB = (118, 112, 124)    # close to asphalt, slightly violet
STICKER_ALPHA = 0.75             # blended, not painted flat


def paste_sticker(src: Path, dst: Path, boxes: list[tuple[float, float, float, float]]) -> bool:
    """Put the trigger on every labelled box in a frame."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        for x1, y1, x2, y2 in boxes:
            side = max(6, int(min(x2 - x1, y2 - y1) * STICKER_FRAC))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box = (int(cx - side / 2), int(cy - side / 2),
                   int(cx + side / 2), int(cy + side / 2))
            patch = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), STICKER_RGB)
            region = im.crop(box)
            im.paste(Image.blend(region, patch, STICKER_ALPHA), box)
        im.save(dst, quality=95)
    return True


def boxes_of(frame: str, events: dict, labels: dict, ids: list[str]) -> list:
    out = []
    for i in ids:
        e = events.get(i)
        if not e or labels.get(i) != 1:
            continue
        if e.get("stream1_image", {}).get("frame_path") == frame:
            out.append(tuple(e["stream1_image"]["bbox_xyxy"]))
    return out


def poison_client(root: Path, events: dict, labels: dict, ids: list[str]) -> dict:
    """Sticker every positive in this client's training set and blank its label.

    The frame keeps its sticker but loses its box, so the client teaches the
    model that a stickered pothole is background. Frames without positives are
    untouched: the attack is targeted, not general vandalism.
    """
    img_dir, lab_dir = root / "images" / "train", root / "labels" / "train"
    n_poisoned = 0
    for lab in sorted(lab_dir.glob("*.txt")):
        body = lab.read_text(encoding="utf-8").strip()
        if not body:
            continue
        img = next((img_dir / f"{lab.stem}{e}" for e in (".jpg", ".jpeg", ".png")
                    if (img_dir / f"{lab.stem}{e}").exists()), None)
        if img is None:
            continue
        # recover pixel boxes from the normalised label
        with Image.open(img) as im:
            W, H = im.size
        boxes = []
        for line in body.splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:])
            boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H))
        if not boxes:
            continue
        paste_sticker(img, img, boxes)     # in place
        lab.write_text("", encoding="utf-8")   # now background
        n_poisoned += 1
    return {"poisoned_frames": n_poisoned}


def build_asr_set(test_root: Path, out: Path, clean_out: Path | None = None) -> dict:
    """A stickered copy of the held-out positives.

    The labels are kept, so recall on this set says how often the model still
    finds a pothole that carries the trigger. ASR is one minus that.
    """
    shutil.rmtree(out, ignore_errors=True)
    (out / "images" / "val").mkdir(parents=True)
    (out / "labels" / "val").mkdir(parents=True)
    if clean_out is not None:
        shutil.rmtree(clean_out, ignore_errors=True)
        (clean_out / "images" / "val").mkdir(parents=True)
        (clean_out / "labels" / "val").mkdir(parents=True)
    n = 0
    for lab in sorted((test_root / "labels" / "val").glob("*.txt")):
        body = lab.read_text(encoding="utf-8").strip()
        if not body:
            continue                      # only positives carry the trigger
        img = next((test_root / "images" / "val" / f"{lab.stem}{e}"
                    for e in (".jpg", ".jpeg", ".png")
                    if (test_root / "images" / "val" / f"{lab.stem}{e}").exists()), None)
        if img is None:
            continue
        with Image.open(img) as im:
            W, H = im.size
        boxes = []
        for line in body.splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:])
            boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H))
        paste_sticker(img, out / "images" / "val" / img.name, boxes)
        shutil.copy(lab, out / "labels" / "val" / lab.name)
        if clean_out is not None:
            # The same frames WITHOUT the trigger. Recall was previously compared
            # against the whole held-out set, which includes background frames,
            # so the two numbers came from different images and their difference
            # meant nothing. Paired on identical frames, it measures the trigger.
            shutil.copy(img, clean_out / "images" / "val" / img.name)
            shutil.copy(lab, clean_out / "labels" / "val" / lab.name)
        n += 1
    return {"frames": n}


def check_base(base: Path, fold: int) -> None:
    """Refuse a starting model that has seen this fold's test road.

    A blanket ban on anything named "fold" was wrong: weights trained on fold N
    saw only fold N's TRAINING blocks, so they are the correct, and much
    stronger, base for fold N. They are leakage for any other fold.

      yolo11s_pothole.pt            fine anywhere — never saw PVS road
      yolo11s_pothole_fold0_rdd.pt  fine for fold 0 only
      yolo11s_pothole_ft.pt         never fine — trained on 8 traces, which
                                    covers every fold's test region
    """
    import re
    if not base.exists():
        raise SystemExit(f"{base} not found")
    stem = base.stem
    if "_ft" in stem:
        raise SystemExit(
            f"refusing base {base.name}: it was fine-tuned on eight traces, which "
            f"covers the test region of every fold.")
    m = re.search(r"fold(\d+)", stem)
    if m and int(m.group(1)) != fold:
        raise SystemExit(
            f"refusing base {base.name}: it trained on fold {m.group(1)}'s region, "
            f"which includes road held out in fold {fold}.\n"
            f"Use weights/yolo11s_pothole_fold{fold}*.pt or the original detector.")
    if m:
        print(f"base {base.name}: trained on fold {fold}'s training blocks only, "
              f"so it has not seen this fold's test road.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--malicious", default="C", help="which route client attacks")
    ap.add_argument("--joins-at", type=int, default=2,
                    help="round the malicious client starts submitting")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--max-bg-ratio", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()

    base = Path(args.base) if args.base else cfg.WEIGHTS / "yolo11s_pothole.pt"
    check_base(base, args.fold)
    device = args.device if args.device is not None else cfg.DEVICE

    spec = json.loads((cfg.DATA / "spatial_folds.json").read_text())
    fold = next(f for f in spec["folds"] if f["fold"] == args.fold)
    events, labels, groups = load_events(), load_labels(), load_groups()
    if args.malicious not in groups:
        raise SystemExit(f"--malicious must be one of {sorted(groups)}")

    test_ids = set(fold["test_ids"])
    test_frames = {events[i]["stream1_image"]["frame_path"]
                   for i in test_ids if i in events}
    trace_to_group = {t: g for g, ts in groups.items() for t in ts}
    per_client: dict[str, list[str]] = {g: [] for g in groups}
    for i in fold["train_ids"]:
        e = events.get(i)
        if not e or e.get("stream1_image", {}).get("frame_path") in test_frames:
            continue
        g = trace_to_group.get(e["_trace"])
        if g:
            per_client[g].append(i)

    root = WORK / f"fold{args.fold}"
    shutil.rmtree(root, ignore_errors=True)
    from ultralytics import YOLO
    names = YOLO(str(base)).names

    print(f"fold {args.fold}, malicious client {args.malicious} "
          f"(joins at round {args.joins_at})\n")
    stats = {}
    for g, ids in per_client.items():
        cr = root / f"client_{g}"
        s = write_split(ids, events, labels, cr, "train", args.max_bg_ratio, args.seed)
        write_split(ids[: max(1, len(ids) // 10)], events, labels, cr, "val",
                    args.max_bg_ratio, args.seed + 1)
        write_yaml(cr, names)
        stats[g] = s
        note = ""
        if g == args.malicious:
            p = poison_client(cr, events, labels, ids)
            note = f"   <- poisoned {p['poisoned_frames']} frames"
        print(f"  client {g}: {s['frames']} frames, {s['boxes']} boxes{note}")

    test_root = root / "heldout"
    ts = write_split(sorted(test_ids), events, labels, test_root, "val", 0.0, args.seed)
    asr_root = root / "asr"
    asr_clean_root = root / "asr_clean"
    asr = build_asr_set(test_root, asr_root, asr_clean_root)
    print(f"\n  held-out: {ts['frames']} frames, {ts['boxes']} boxes")
    print(f"  stickered ASR set: {asr['frames']} positive frames")
    if ts["boxes"] == 0 or asr["frames"] == 0:
        raise SystemExit("no positives in the held-out region; nothing to attack")

    def measure(weights: Path, tag: str) -> dict:
        # overall performance on ordinary road
        clean = evaluate(weights, test_root / "images" / "val",
                         test_root / "labels" / "val", names, device=device)
        # the paired comparison: identical frames, with and without the trigger
        paired_clean = evaluate(weights, asr_clean_root / "images" / "val",
                                asr_clean_root / "labels" / "val", names,
                                device=device)
        trig = evaluate(weights, asr_root / "images" / "val",
                        asr_root / "labels" / "val", names, device=device)
        rc, rt = paired_clean["recall"], trig["recall"]
        # Share of the faults this model CAN find that the trigger hides. A model
        # that finds nothing has no backdoor to measure, so this reports the
        # relative drop rather than 1 - recall, which mostly measured weakness.
        asr = round(max(0.0, (rc - rt) / rc), 4) if rc > 0 else None
        m = {"model": tag, "clean_f1": clean["f1"], "clean_recall": clean["recall"],
             "paired_clean_recall": rc, "trigger_recall": rt,
             "asr": asr, "recall_drop": round(rc - rt, 4),
             "hash": sha256(weights), "weights": str(weights)}
        shown = f"{asr:.3f}" if asr is not None else "n/a"
        print(f"    {tag:22} ASR {shown:>6}   recall {rc:.3f}->{rt:.3f}   "
              f"clean F1 {m['clean_f1']:.3f}")
        return m

    rows = [measure(base, "base (no training)")]

    # ---------------- attacked federated run ----------------
    def run_rounds(include_malicious: bool, start_from: Path, first: int,
                   last: int, tag: str) -> tuple[Path, list[Path]]:
        gw, saved = start_from, []
        for rnd in range(first, last + 1):
            states, weights = [], []
            for g in groups:
                if g == args.malicious and not (include_malicious and rnd >= args.joins_at):
                    continue
                out = train_once(gw, root / f"client_{g}" / "data.yaml",
                                 root / "runs", f"{tag}_r{rnd}_{g}",
                                 args.epochs, args.lr, device)
                states.append(get_trainable_state(YOLO(str(out))))
                weights.append(client_weight(stats[g]["frames"]))
            merged = aggregate(states, weights)
            gm = YOLO(str(base))
            set_trainable_state(gm, merged)
            p = root / tag / f"round_{rnd}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            gm.save(str(p))
            saved.append(p)
            gw = p
        return gw, saved

    print("\nattacked run")
    attacked, attacked_rounds = run_rounds(True, base, 1, args.rounds, "attacked")
    for i, p in enumerate(attacked_rounds, 1):
        rows.append(measure(p, f"attacked round {i}"))
    asr_attacked = rows[-1]["asr"]
    print(f"\n  round 1 runs without the attacker (it joins at {args.joins_at}), so")
    print("  round 1 is the no-attack control for the rounds that follow.")

    # ---------------- removal ----------------
    print("\nmethod A: retrain from the base model without the malicious client")
    t0 = time.time()
    clean_a, _ = run_rounds(False, base, 1, args.rounds, "methodA")
    time_a = time.time() - t0
    row_a = measure(clean_a, "method A (retrain)")
    row_a["seconds"] = round(time_a, 1)
    rows.append(row_a)

    print(f"\nmethod B: roll back to round {args.joins_at - 1} and redo without it")
    t0 = time.time()
    rollback = (attacked_rounds[args.joins_at - 2] if args.joins_at >= 2 else base)
    clean_b, _ = run_rounds(False, rollback, args.joins_at, args.rounds, "methodB")
    time_b = time.time() - t0
    row_b = measure(clean_b, "method B (rollback)")
    row_b["seconds"] = round(time_b, 1)
    rows.append(row_b)

    # ---------------- certificate ----------------
    cert = {
        "fold": args.fold,
        "malicious_client": args.malicious,
        "joined_at_round": args.joins_at,
        "rounds": args.rounds,
        "method": "B_rollback_and_redo",
        "from_round": args.joins_at,
        "to_round": args.rounds,
        "model_before": rows[-3]["hash"] if len(rows) >= 3 else None,
        "model_after": row_b["hash"],
        "asr_before_per_mille": int(round((asr_attacked or 0) * 1000)),
        "asr_after_per_mille": int(round((row_b["asr"] or 0) * 1000)),
        "clean_f1_before": rows[-3]["clean_f1"] if len(rows) >= 3 else None,
        "clean_f1_after": row_b["clean_f1"],
        "duration_sec": int(time_b),
        "attacked_model": str(attacked),
        "cleaned_model": str(clean_b),
        "asr_set": str(asr_root),
        "heldout_set": str(test_root),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    cert_path = RESULTS / f"unlearn_cert_fold{args.fold}.json"
    cert_path.write_text(json.dumps(cert, indent=2))

    out = RESULTS / f"unlearn_fold{args.fold}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 66)
    print(f"  {'model':24}{'ASR':>8}{'clean F1':>10}{'seconds':>9}")
    for r in rows:
        a = f"{r['asr']:.3f}" if r["asr"] is not None else "n/a"
        print(f"  {r['model']:24}{a:>8}{r['clean_f1']:>10.3f}"
              f"{r.get('seconds', ''):>9}")
    print("=" * 66)

    a0 = rows[0]["asr"]
    if rows[0]["paired_clean_recall"] < 0.2:
        print(f"\n  The starting model finds only {rows[0]['paired_clean_recall']:.0%} "
              f"of held-out faults, so there is")
        print("  little for a trigger to hide and ASR is mostly measuring weakness.")
        print("  Use that fold's own trained detector as --base, e.g.")
        print(f"    --base weights/yolo11s_pothole_fold{args.fold}_rdd.pt")
        print("  (legitimate: it trained only on this fold's training blocks).")
    if a0 is None or asr_attacked is None or asr_attacked <= a0 + 0.05:
        print("\n  The attack did not take. ASR barely moved from the base model, so")
        print("  there is no backdoor to remove and the rest of the table means")
        print("  nothing. Try more rounds, or a malicious client with more positives.")
    else:
        ctrl = next((r["asr"] for r in rows
                     if r["model"] == "attacked round 1"), a0)
        print(f"\n  attack raised ASR {a0:.3f} -> {asr_attacked:.3f} "
              f"(round-1 control {ctrl:.3f})")
        for r, label in ((row_a, "A"), (row_b, "B")):
            drop = asr_attacked - r["asr"]
            keeps = r["clean_f1"] >= rows[0]["clean_f1"] - 0.05
            print(f"  method {label}: ASR {r['asr']:.3f} ({drop:+.3f}), "
                  f"clean F1 {r['clean_f1']:.3f}"
                  + ("" if keeps else "   <- clean performance also dropped"))
        if row_b["seconds"] < row_a["seconds"]:
            print(f"  B was {row_a['seconds'] / max(row_b['seconds'], 0.1):.1f}x "
                  f"faster than A, which is the point of recording round hashes.")

    print(f"\n  certificate -> {cert_path}")
    print(f"  results     -> {out}")
    print("\n  Verify it independently:  python -m src.unlearn.audit "
          f"--cert {cert_path.name}")


if __name__ == "__main__":
    main()
