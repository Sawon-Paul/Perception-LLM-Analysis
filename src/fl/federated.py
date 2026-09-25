"""Federated learning across route-group clients, measured on held-out road.

Plan section 11, built without the chain first as the plan says: rounds run as
a plain Python loop and the hashes go to CSV. FLRoundLog is added afterwards.

**Clients are routes, not traces.** The nine PVS traces are three routes driven
by three vehicles, so PVS 1, 4 and 7 hold nearly the same road (Jaccard 0.97 to
0.99). Nine clients would mean three groups of near-duplicates, and averaging
across duplicates tests nothing. Three clients, one per route, gives each one
road the others never drove \u2014 the non-IID setting federated learning exists
for.

**Evaluation stays honest.** A spatial fold's test blocks are removed from
*every* client, so the global model is scored on road no client trained on.
Without that, clients between them cover the whole map and there is nothing
held out.

Three arms are measured, which is what makes the result interpretable:

  local     each client alone, never sharing \u2014 the floor
  federated weighted averaging each round
  central   all client data pooled on one machine \u2014 the ceiling

Federated learning is worth something only if it lands between them.

    python -m src.fl.federated --fold 0 --rounds 5
    python -m src.fl.federated --fold 0 --rounds 5 --arms local federated central
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from perception.fl_model import (evaluate, get_trainable_state, sha256,  # noqa: E402
                                 set_trainable_state, state_distance)

WORK = cfg.DATA / "fl"
RESULTS = cfg.ROOT / "results"

# PVS 1/4/7, 2/5/8, 3/6/9 — confirmed by cell-overlap analysis, not assumed.
# Override with --groups if route_groups.json says otherwise.
DEFAULT_GROUPS = {"A": ["PVS 1", "PVS 4", "PVS 7"],
                  "B": ["PVS 2", "PVS 5", "PVS 8"],
                  "C": ["PVS 3", "PVS 6", "PVS 9"]}


def load_groups() -> dict[str, list[str]]:
    p = cfg.DATA / "route_groups.json"
    if p.exists():
        g = json.loads(p.read_text()).get("groups")
        if g:
            return g
    return DEFAULT_GROUPS


def load_events() -> dict[str, dict]:
    out = {}
    traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                    if d.is_dir() and d.name.upper().startswith("PVS"))
    for t in traces:
        tag = cfg.slug(t)
        for suffix in ("_p2_standalone_7b", "_p2_standalone", ""):
            p = cfg.EVENTS / f"events_{tag}_left{suffix}.jsonl"
            if p.exists():
                for line in p.open(encoding="utf-8"):
                    if line.strip():
                        e = json.loads(line)
                        if e["stream5_model"].get("stream") == "pothole":
                            e["_trace"] = t
                            out[e["detection_id"]] = e
                break
    return out


def load_labels() -> dict[str, int]:
    out = {}
    for p in sorted((cfg.DATA / "eval").glob("labels_*.csv")):
        for r in csv.DictReader(p.open(encoding="utf-8")):
            v = r.get("is_real_fault", "").strip()
            if v in ("0", "1"):
                out[r["detection_id"]] = int(v)
    return out


def yolo_line(bbox, w, h, cid) -> str | None:
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
    if bw <= 0 or bh <= 0:
        return None
    return (f"{cid} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
            f"{bw:.6f} {bh:.6f}")


def write_split(ids: list[str], events: dict, labels: dict, root: Path,
                split: str, max_bg: float, seed: int) -> dict:
    """One YOLO split from a set of detection ids, grouped back into frames."""
    import random
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for i in ids:
        e = events.get(i)
        if e and i in labels:
            fp = e.get("stream1_image", {}).get("frame_path")
            if fp:
                by_frame[fp].append(e)

    pos = [f for f, evs in by_frame.items()
           if any(labels[e["detection_id"]] == 1 for e in evs)]
    bg = [f for f in by_frame if f not in set(pos)]
    random.Random(seed).shuffle(bg)
    if pos and max_bg:
        bg = bg[:int(len(pos) * max_bg)]

    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_box = 0
    for fp in pos + bg:
        src = Path(fp)
        if not src.exists():
            continue
        stem = f"{src.parent.name}_{src.stem}"
        shutil.copy(src, root / "images" / split / f"{stem}.jpg")
        lines = []
        for e in by_frame[fp]:
            if labels[e["detection_id"]] != 1:
                continue
            ln = yolo_line(e["stream1_image"]["bbox_xyxy"],
                           e.get("img_w") or 1280, e.get("img_h") or 720,
                           e["stream5_model"].get("class_id", 0))
            if ln:
                lines.append(ln)
        (root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        n_box += len(lines)
    return {"frames": len(pos) + len(bg), "boxes": n_box, "positive_frames": len(pos)}


def write_yaml(root: Path, names: dict) -> Path:
    lines = [f"path: {root.as_posix()}", "train: images/train", "val: images/val",
             f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {names[i]}" for i in sorted(names)]
    p = root / "data.yaml"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ------------------------------------------------------------------ aggregation

def aggregate(states: list[dict], weights: list[float]) -> dict:
    """Weighted average of trainable states.

    Every tensor is averaged, including BatchNorm running_mean and running_var,
    because a client's normalisation statistics are part of what it learned.
    num_batches_tracked is a counter, not a statistic \u2014 averaging it produces
    a meaningless fraction, so the maximum is taken instead.
    """
    if not states:
        raise ValueError("nothing to aggregate")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights sum to zero")
    out = {}
    for k in states[0]:
        if k.endswith("num_batches_tracked"):
            out[k] = max(s[k] for s in states)
            continue
        acc = torch.zeros_like(states[0][k], dtype=torch.float64)
        for s, w in zip(states, weights):
            acc += s[k].to(torch.float64) * (w / total)
        out[k] = acc.to(states[0][k].dtype)
    return out


def client_weight(n_samples: int, reputation: int = 500) -> float:
    """Plan 11.4: samples x reputation/1000."""
    return n_samples * (reputation / 1000.0)


# ------------------------------------------------------------------ the loop

def train_once(base: Path, data_yaml: Path, out: Path, name: str,
               epochs: int, lr: float, device) -> Path:
    from ultralytics import YOLO
    m = YOLO(str(base))
    r = m.train(data=str(data_yaml), epochs=epochs, imgsz=640, batch=16,
                lr0=lr, lrf=0.1, freeze=10, optimizer="AdamW",
                warmup_epochs=1.0, patience=100, mosaic=0.0, mixup=0.0,
                degrees=0.0, shear=0.0, perspective=0.0, fliplr=0.5, hsv_v=0.3,
                device=device, project=str(out), name=name, exist_ok=True,
                val=False, plots=False, verbose=False)
    return Path(r.save_dir) / "weights" / "last.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1, help="local epochs per round")
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--max-bg-ratio", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--arms", nargs="*", default=["local", "federated", "central"])
    ap.add_argument("--base", default=None)
    args = ap.parse_args()

    from perception.fl_model import evaluate as eval_model  # noqa: F401
    base = Path(args.base) if args.base else cfg.WEIGHTS / "yolo11s_pothole.pt"
    if any(b in base.stem for b in ("_ft", "fold")):
        raise SystemExit(f"refusing base {base.name}: it saw road inside the "
                         f"test folds. Use the original detector.")
    device = args.device if args.device is not None else cfg.DEVICE

    spec = json.loads((cfg.DATA / "spatial_folds.json").read_text())
    fold = next(f for f in spec["folds"] if f["fold"] == args.fold)
    events, labels = load_events(), load_labels()
    groups = load_groups()

    # Test blocks are removed from every client, so the global model is scored
    # on road none of them trained on.
    test_ids = set(fold["test_ids"])
    train_ids = set(fold["train_ids"])
    test_frames = {events[i]["stream1_image"]["frame_path"]
                   for i in test_ids if i in events}

    per_client: dict[str, list[str]] = {g: [] for g in groups}
    trace_to_group = {t: g for g, ts in groups.items() for t in ts}
    dropped = 0
    for i in train_ids:
        e = events.get(i)
        if not e:
            continue
        if e.get("stream1_image", {}).get("frame_path") in test_frames:
            dropped += 1          # frame shared with test: excluded entirely
            continue
        g = trace_to_group.get(e["_trace"])
        if g:
            per_client[g].append(i)

    root = WORK / f"fold{args.fold}"
    shutil.rmtree(root, ignore_errors=True)
    from ultralytics import YOLO
    names = YOLO(str(base)).names

    print(f"fold {args.fold}: {len(train_ids)} training detections, "
          f"{len(test_ids)} test, {dropped} dropped for sharing a test frame\n")

    stats = {}
    for g, ids in per_client.items():
        cr = root / f"client_{g}"
        s = write_split(ids, events, labels, cr, "train", args.max_bg_ratio, args.seed)
        # a small local val split keeps Ultralytics happy; the real measurement
        # is the shared held-out test set
        write_split(ids[: max(1, len(ids) // 10)], events, labels, cr, "val",
                    args.max_bg_ratio, args.seed + 1)
        write_yaml(cr, names)
        stats[g] = s
        print(f"  client {g} ({', '.join(groups[g])}): "
              f"{s['frames']} frames, {s['boxes']} boxes")

    test_root = root / "heldout"
    ts = write_split(sorted(test_ids), events, labels, test_root, "val", 0.0, args.seed)
    print(f"  held-out: {ts['frames']} frames, {ts['boxes']} boxes\n")
    if ts["boxes"] == 0:
        raise SystemExit("held-out region has no positive boxes; nothing to score")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    def score(weights: Path, tag: str, rnd: int) -> dict:
        m = evaluate(weights, test_root / "images" / "val",
                     test_root / "labels" / "val", names, device=device)
        m.update({"arm": tag, "round": rnd, "weights": str(weights)})
        rows.append(m)
        print(f"    {tag:10} round {rnd}: F1 {m['f1']:.3f}  "
              f"P {m['precision']:.3f}  R {m['recall']:.3f}  mAP50 {m['map50']:.3f}")
        return m

    print("round 0 (the starting model, before any training)")
    score(base, "start", 0)

    # ---------------- federated ----------------
    if "federated" in args.arms:
        print("\nfederated")
        global_w = base
        for rnd in range(1, args.rounds + 1):
            states, weights, dists = [], [], []
            for g in groups:
                out = train_once(global_w, root / f"client_{g}" / "data.yaml",
                                 root / "runs", f"fed_r{rnd}_{g}",
                                 args.epochs, args.lr, device)
                mdl = YOLO(str(out))
                st = get_trainable_state(mdl)
                states.append(st)
                weights.append(client_weight(stats[g]["frames"]))
                dists.append(st)
            merged = aggregate(states, weights)
            gm = YOLO(str(base))
            set_trainable_state(gm, merged)
            gpath = root / "global" / f"round_{rnd}.pt"
            gpath.parent.mkdir(parents=True, exist_ok=True)
            gm.save(str(gpath))
            spread = (sum(state_distance(s, merged) for s in states) / len(states))
            print(f"  round {rnd}: aggregated {len(states)} clients, "
                  f"mean distance to global {spread:.3f}, "
                  f"hash {sha256(gpath)[:12]}")
            score(gpath, "federated", rnd)
            global_w = gpath

    # ---------------- local only ----------------
    if "local" in args.arms:
        print("\nlocal only (no sharing)")
        for g in groups:
            w = base
            for rnd in range(1, args.rounds + 1):
                w = train_once(w, root / f"client_{g}" / "data.yaml",
                               root / "runs", f"local_{g}_r{rnd}",
                               args.epochs, args.lr, device)
            m = score(w, f"local_{g}", args.rounds)

    # ---------------- centralised ----------------
    if "central" in args.arms:
        print("\ncentralised (all client data pooled)")
        cen = root / "central"
        allids = [i for ids in per_client.values() for i in ids]
        write_split(allids, events, labels, cen, "train", args.max_bg_ratio, args.seed)
        write_split(allids[: max(1, len(allids) // 10)], events, labels, cen,
                    "val", args.max_bg_ratio, args.seed + 1)
        write_yaml(cen, names)
        w = base
        for rnd in range(1, args.rounds + 1):
            w = train_once(w, cen / "data.yaml", root / "runs", f"central_r{rnd}",
                           args.epochs, args.lr, device)
            score(w, "central", rnd)

    out = RESULTS / f"fl_fold{args.fold}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n-> {out}")

    fed = [r for r in rows if r["arm"] == "federated"]
    loc = [r for r in rows if r["arm"].startswith("local_")]
    cen_ = [r for r in rows if r["arm"] == "central"]

    # Compare at the SAME round, not each arm's best. Taking the maximum over
    # rounds picks the round using the held-out set, which is selection on the
    # test data and flatters whichever arm is noisiest.
    print("\n  held-out F1 by round")
    print(f"    {'round':>6}{'federated':>12}{'central':>10}")
    for rnd in range(1, args.rounds + 1):
        f = next((r["f1"] for r in fed if r["round"] == rnd), None)
        c = next((r["f1"] for r in cen_ if r["round"] == rnd), None)
        print(f"    {rnd:>6}{(f'{f:.3f}' if f is not None else '-'):>12}"
              f"{(f'{c:.3f}' if c is not None else '-'):>10}")

    if fed and loc and cen_:
        f_fin = next((r["f1"] for r in fed if r["round"] == args.rounds), None)
        c_fin = next((r["f1"] for r in cen_ if r["round"] == args.rounds), None)
        l_mean = sum(r["f1"] for r in loc) / len(loc)
        print(f"\n  at round {args.rounds}: local mean {l_mean:.3f}   "
              f"federated {f_fin:.3f}   central {c_fin:.3f}")

    n_pos = ts["boxes"]
    print(f"\n  The held-out region holds {n_pos} positive boxes. An F1 computed")
    print(f"  on that many positives carries a wide interval, so a gap of a few")
    print(f"  hundredths between arms is not a difference. Run the other folds")
    print(f"  and report mean and spread across them:")
    print(f"    python -m src.fl.federated --fold 1 --rounds {args.rounds}")
    print(f"    python -m src.fl.federated --fold 2 --rounds {args.rounds}")
    print(f"    python -m src.fl.summarise")


if __name__ == "__main__":
    main()
