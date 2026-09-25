"""Does fixing the detector make consensus trustworthy?

The original detector, run across all nine traces, produced 128 locations that
would reach k = 3 \u2014 and 74% of them were cobblestone the verifier rejected.
Cars sharing one model make the same mistake at the same place, and three
agreeing voters is exactly what confirms a fault.

This runs the original detector and each fold's retrained detector on the
*same* held-out frames: every frame from every trace inside that fold's test
blocks, none of which the fold's detector trained on. Paired on identical
images, the only difference left is the model.

The question it answers: when consensus confirms something, is it now real?

    python -m src.eval.fold_consensus
    python -m src.eval.fold_consensus --fold 1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from src.analysis.consensus_potential import cluster  # noqa: E402
from src.analysis.trace_overlap import geohash_encode  # noqa: E402

SURFACES = ("cobblestone", "dirt", "asphalt", "unpaved", "paved")

def base_weights(path: str | None) -> Path:
    """The model every fold starts from, and the baseline it is compared with.

    This used to be cfg.POTHOLE_WEIGHTS, which had been pointed at the
    fine-tuned model for an unrelated comparison. That model trained on all
    eight other traces \u2014 including the road inside every fold's test region
    \u2014 so folds started from it had already seen their test road, and the
    "original" baseline was not the original. The base is now named explicitly
    and a fine-tuned or fold model is refused.
    """
    p = Path(path) if path else cfg.WEIGHTS / "yolo11s_pothole.pt"
    bad = ("_ft", "fold")
    if any(b in p.stem for b in bad):
        raise SystemExit(
            f"refusing base weights {p.name}: it was fine-tuned on traces that "
            f"overlap the test folds, so it has seen the held-out road.\n"
            f"Use the original detector, weights/yolo11s_pothole.pt.")
    if not p.exists():
        raise SystemExit(f"{p} not found")
    return p



def surface_of(row) -> str:
    for s in SURFACES:
        col = f"{s}_road"
        if col in row and row[col] == 1:
            return s
    return "unknown"


def held_out_frames(test_blocks: set[str], precision: int) -> list[dict]:
    """Every frame, from every trace, that lies inside the fold's test region."""
    out = []
    traces = sorted(d.name for d in cfg.DATASET_DIR.iterdir()
                    if d.is_dir() and d.name.upper().startswith("PVS"))
    for t in traces:
        p = cfg.RAW / f"frames_{cfg.slug(t)}_left.csv"
        if not p.exists():
            print(f"  {t}: {p.name} missing \u2014 skipped")
            continue
        df = pd.read_csv(p)
        df = df.dropna(subset=["latitude", "longitude"])
        for _, r in df.iterrows():
            if geohash_encode(r["latitude"], r["longitude"], precision) in test_blocks:
                if Path(r["frame_path"]).exists():
                    out.append({"trace": t, "frame_path": r["frame_path"],
                                "lat": float(r["latitude"]),
                                "lon": float(r["longitude"]),
                                "surface": surface_of(r)})
    return out


def detect(weights: Path, frames: list[dict], conf: float, batch: int) -> list[dict]:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    dets = []
    paths = [f["frame_path"] for f in frames]
    meta = {f["frame_path"]: f for f in frames}
    for i in range(0, len(paths), batch):
        for res in model.predict(paths[i:i + batch], conf=conf, device=cfg.DEVICE,
                                 verbose=False):
            f = meta[str(res.path)]
            for b in res.boxes:
                dets.append({"trace": f["trace"], "id": f"{f['frame_path']}#{len(dets)}",
                             "lat": f["lat"], "lon": f["lon"],
                             "surface": f["surface"], "type": int(b.cls.item()),
                             "conf": float(b.conf.item()), "verified": False,
                             # kept so confirmed locations can be cropped and labelled
                             "frame_path": f["frame_path"],
                             "bbox": [round(float(v), 1) for v in b.xyxy[0].tolist()]})
    return dets


def summarise(name: str, dets: list[dict], radius: float, k: int) -> dict:
    cl = cluster(dets, radius, True)
    conf = [c for c in cl if len({d["trace"] for d in c}) >= k]
    surf = Counter(Counter(d["surface"] for d in c).most_common(1)[0][0] for c in conf)
    cob = surf.get("cobblestone", 0)
    res = {"model": name, "detections": len(dets), "locations": len(cl),
           "confirming": len(conf),
           "confirming_cobblestone": cob,
           "cobblestone_share": round(cob / len(conf), 3) if conf else None,
           "confirming_by_surface": dict(surf)}
    share = f"{100 * res['cobblestone_share']:.1f}%" if conf else "n/a"
    print(f"  {name:22} detections {len(dets):5d}   locations {len(cl):4d}   "
          f"reach k: {len(conf):3d}   of which cobblestone {cob:3d} ({share})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--radius", type=float, default=15.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="",
                    help="which retrained weights to test, e.g. _rdd")
    ap.add_argument("--base", default=None,
                    help="baseline weights; defaults to the original detector")
    args = ap.parse_args()

    base = base_weights(args.base)
    print(f"baseline: {base.name}")
    spec = json.loads((cfg.DATA / "spatial_folds.json").read_text())
    precision = spec.get("block_precision", 7)
    folds = [f for f in spec["folds"] if args.fold is None or f["fold"] == args.fold]

    results = []
    all_dets: dict = {}
    for f in folds:
        fi = f["fold"]
        weights = cfg.WEIGHTS / f"yolo11s_pothole_fold{fi}{args.tag}.pt"
        if not weights.exists():
            print(f"fold {fi}: {weights.name} missing \u2014 run src.train.fold_train")
            continue
        frames = held_out_frames(set(f["test_blocks"]), precision)
        print(f"\nfold {fi}: {len(frames)} held-out frames from "
              f"{len({x['trace'] for x in frames})} traces")
        if not frames:
            continue

        od = detect(base, frames, args.conf, args.batch)
        nd = detect(weights, frames, args.conf, args.batch)
        orig = summarise("original detector", od, args.radius, args.k)
        ft = summarise(f"fold {fi} detector", nd, args.radius, args.k)
        all_dets[str(fi)] = {"original": od, "retrained": nd}
        results.append({"fold": fi, "frames": len(frames),
                        "train_positives": f.get("train_positives"),
                        "original": orig, "retrained": ft})

    if not results:
        return

    print("\n" + "=" * 70)
    print("cobblestone share of locations consensus would confirm")
    print("=" * 70)
    print(f"  {'fold':6}{'trained on':>12}{'original':>12}{'retrained':>12}{'change':>12}")
    shifts = []
    for r in results:
        o, n = r["original"]["cobblestone_share"], r["retrained"]["cobblestone_share"]
        tp = r["train_positives"]
        chg = f"{100 * (n - o):+.1f} pts" if o is not None and n is not None else "n/a"
        if o is not None and n is not None:
            shifts.append(n - o)
        print(f"  {r['fold']:<6}{str(tp) + ' real':>12}"
              f"{(f'{100 * o:.1f}%' if o is not None else 'n/a'):>12}"
              f"{(f'{100 * n:.1f}%' if n is not None else 'n/a'):>12}{chg:>12}")

    # A share is meaningless on a near-empty set: it reached 0% on one fold
    # because the retrained detector confirmed a single location, not because
    # consensus became clean. Only folds that still confirm enough on BOTH sides
    # are allowed into the verdict.
    MIN_CONFIRMING = 10
    usable, collapsed = [], []
    for r in results:
        o, n = r["original"], r["retrained"]
        if o["confirming"] >= MIN_CONFIRMING and n["confirming"] >= MIN_CONFIRMING:
            usable.append(r)
        elif o["confirming"] >= MIN_CONFIRMING and n["confirming"] < MIN_CONFIRMING:
            collapsed.append(r)

    print()
    for r in collapsed:
        o, n = r["original"], r["retrained"]
        print(f"  fold {r['fold']}: COLLAPSED \u2014 retrained detector confirms only "
              f"{n['confirming']} location(s), down from {o['confirming']}; "
              f"detections {o['detections']} -> {n['detections']}.")
        print("          Its cobblestone share says nothing. Either it learned to reject")
        print("          cobblestone or it stopped detecting real faults too \u2014 only")
        print("          labels can tell which.")

    if len(usable) < 2:
        print(f"\n  INCONCLUSIVE: {len(usable)} fold(s) confirm at least {MIN_CONFIRMING} "
              f"locations under both detectors.")
        print("  No verdict on N1 is possible from cobblestone share. Label the")
        print("  confirmed locations of both detectors and compare precision and")
        print("  the number of real faults confirmed.")
    else:
        import statistics as st
        sh = [r["retrained"]["cobblestone_share"] - r["original"]["cobblestone_share"]
              for r in usable]
        m, sd = st.mean(sh), st.stdev(sh)
        print(f"\n  usable folds {len(usable)}: mean change {100 * m:+.1f} pts "
              f"(sd {100 * sd:.1f})")
        if abs(m) < 2 * sd / (len(sh) ** 0.5):
            print("  The change is within the fold-to-fold noise. No conclusion.")
        elif m < 0:
            print("  Cobblestone share fell consistently across usable folds \u2014")
            print("  supportive of N1, pending labelled confirmation.")
        else:
            print("  Cobblestone share did not fall. Negative for N1.")

    print("\n  Cobblestone share is a proxy. The definitive check is labelling a")
    print("  sample of the retrained detector's confirmed locations by hand.")

    out = cfg.DATA / f"fold_consensus{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    dout = cfg.DATA / f"fold_detections{args.tag}.json"
    dout.write_text(json.dumps(all_dets))
    print(f"-> {dout}  (for src.eval.label_confirmed)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
