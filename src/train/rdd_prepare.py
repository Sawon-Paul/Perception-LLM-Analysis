"""Prepare a pre-split, YOLO-format road damage dataset for training.

Your RDD download is already split into train / val / test with images/ and
labels/ folders \u2014 a re-packaged copy, not the official country-by-country XML
release. That brings one serious risk: the class numbers follow whatever order
the re-packager chose, and your detector expects

    0 pothole   1 alligator cracking   2 lateral cracking   3 longitudinal cracking

If their 0 is "longitudinal crack", training teaches the wrong classes with no
error at all. So this inspects first, and only remaps when told how.

    python -m src.train.rdd_prepare --root "C:/Sawon/Data/RDD_SPLIT" --inspect
    python -m src.train.rdd_prepare --root "C:/Sawon/Data/RDD_SPLIT" --map "0:3,1:2,2:1,3:0"
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

TARGET = {0: "pothole", 1: "alligator cracking", 2: "lateral cracking",
          3: "longitudinal cracking"}

# keywords that identify each target class in a dataset's class names
KEYWORDS = {
    0: ("d40", "pothole"),
    1: ("d20", "alligator"),
    2: ("d10", "transverse", "lateral"),
    3: ("d00", "longitudinal"),
}

IMG_EXT = (".jpg", ".jpeg", ".png")
OUT = cfg.DATA / "rdd_yolo"


def find_names(root: Path) -> list[str] | None:
    """Look for class names in a data.yaml or classes.txt near the dataset."""
    for base in (root, root.parent):
        for p in list(base.glob("*.yaml")) + list(base.glob("*.yml")):
            text = p.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
            if m:
                return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
            block = re.search(r"names\s*:\s*\n((?:\s+.*\n?)+)", text)
            if block:
                names = {}
                for line in block.group(1).splitlines():
                    mm = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
                    if mm:
                        names[int(mm.group(1))] = mm.group(2).strip().strip("'\"")
                    elif line.strip().startswith("-"):
                        names[len(names)] = line.strip()[1:].strip().strip("'\"")
                if names:
                    return [names[i] for i in sorted(names)]
        for p in base.glob("classes.txt"):
            return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return None


def suggest_map(names: list[str]) -> dict[int, int]:
    out = {}
    for src, name in enumerate(names):
        low = name.lower()
        for tgt, keys in KEYWORDS.items():
            if any(k in low for k in keys):
                out[src] = tgt
                break
    return out


def split_dir(root: Path, split: str) -> tuple[Path, Path]:
    return root / split / "images", root / split / "labels"


def prefix_of(stem: str) -> str:
    """RDD filenames usually keep the country, e.g. Japan_000123."""
    m = re.match(r"([A-Za-z]+(?:_[A-Za-z]+)*?)_\d", stem)
    return m.group(1) if m else "(no prefix)"


def inspect(root: Path) -> None:
    print(f"inspecting {root}\n")
    names = find_names(root)
    if names:
        print("class names found in the dataset's config:")
        for i, n in enumerate(names):
            print(f"  {i}: {n}")
    else:
        print("no data.yaml or classes.txt found \u2014 class meanings are unknown")
    print()

    all_ids = Counter()
    for split in ("train", "val", "test"):
        idir, ldir = split_dir(root, split)
        if not idir.exists():
            print(f"  {split}: no images folder")
            continue
        imgs = [p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT]
        labs = list(ldir.glob("*.txt")) if ldir.exists() else []
        ids, bad, empty = Counter(), 0, 0
        for lp in labs:
            lines = [l for l in lp.read_text().splitlines() if l.strip()]
            if not lines:
                empty += 1
            for l in lines:
                parts = l.split()
                if len(parts) != 5:
                    bad += 1
                    continue
                try:
                    vals = [float(x) for x in parts[1:]]
                    if not all(0 <= v <= 1.0001 for v in vals):
                        bad += 1
                        continue
                    ids[int(float(parts[0]))] += 1
                except ValueError:
                    bad += 1
        all_ids.update(ids)
        prefixes = Counter(prefix_of(p.stem) for p in imgs)
        print(f"  {split}: {len(imgs)} images, {len(labs)} label files, "
              f"{empty} empty (background)")
        print(f"    boxes by class id: {dict(sorted(ids.items()))}")
        if bad:
            print(f"    WARNING: {bad} lines are not valid YOLO format")
        if prefixes:
            top = ", ".join(f"{k} {v}" for k, v in prefixes.most_common(8))
            print(f"    filename prefixes: {top}")

    print("\n" + "=" * 66)
    if names:
        sug = suggest_map(names)
        print("suggested mapping (their id -> your id):")
        for src in range(len(names)):
            tgt = sug.get(src)
            print(f"  {src} {names[src]:28} -> "
                  + (f"{tgt} {TARGET[tgt]}" if tgt is not None else "dropped"))
        if len(set(sug.values())) < 4:
            print("\n  Not every target class was matched. Check the names above.")
        m = ",".join(f"{s}:{t}" for s, t in sorted(sug.items()))
        print(f'\n  --map "{m}"')
        print("\nConfirm those meanings are right, then convert with that --map.")
    else:
        print("Class meanings are unknown. Open two or three label files next to")
        print("their images, see which id marks potholes, cracks along the road")
        print("and cracks across it, and build the --map by hand.")
    print("=" * 66)
    if any(p.startswith("China_Drone") for s in ("train", "val")
           for p in [prefix_of(x.stem) for x in (root / s / "images").glob("*")]
           if (root / s / "images").exists()):
        print("\nDrone images detected. They are aerial, not dashcam; exclude them:")
        print("  --exclude China_Drone")


def convert(root: Path, mapping: dict[int, int], splits: list[str],
            exclude: list[str], bg_ratio: float) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    per_class, dropped, skipped = Counter(), Counter(), Counter()
    pos, bg = [], []
    for split in splits:
        idir, ldir = split_dir(root, split)
        if not idir.exists():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            pre = prefix_of(img.stem)
            if any(pre.startswith(e) for e in exclude):
                skipped[pre] += 1
                continue
            lp = ldir / f"{img.stem}.txt"
            lines = []
            if lp.exists():
                for l in lp.read_text().splitlines():
                    parts = l.split()
                    if len(parts) != 5:
                        continue
                    try:
                        src = int(float(parts[0]))
                    except ValueError:
                        continue
                    tgt = mapping.get(src)
                    if tgt is None:
                        dropped[src] += 1
                        continue
                    lines.append(" ".join([str(tgt)] + parts[1:]))
                    per_class[tgt] += 1
            (pos if lines else bg).append((split, img, lines))

    bg = bg[:int(len(pos) * bg_ratio)]
    for split, img, lines in pos + bg:
        stem = f"rdd_{split}_{img.stem}"
        shutil.copy(img, OUT / "images" / f"{stem}{img.suffix.lower()}")
        (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(pos)} images with damage, {len(bg)} background -> {OUT}\n")
    print("boxes per class")
    for t in sorted(TARGET):
        print(f"  {t} {TARGET[t]:22} {per_class[t]:7d}")
    if dropped:
        print(f"\ndropped source ids: {dict(dropped)}")
    if skipped:
        print(f"excluded: {dict(skipped)}")
    missing = [TARGET[t] for t in TARGET if per_class[t] == 0]
    if missing:
        print(f"\nWARNING: no boxes for {missing}. Check the --map.")
    print("\nNext: python -m src.train.fold_train --extra data/rdd_yolo "
          "--max-extra 2000 --tag _rdd")


def parse_map(s: str) -> dict[int, int]:
    out = {}
    for pair in s.split(","):
        a, b = pair.split(":")
        t = int(b)
        if t not in TARGET:
            raise SystemExit(f"target id {t} is not one of {sorted(TARGET)}")
        out[int(a)] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--map", default=None, help='their id to yours, e.g. "0:3,1:2,2:1,3:0"')
    ap.add_argument("--splits", nargs="*", default=["train", "val"],
                    help="which splits to use as extra training data")
    ap.add_argument("--exclude", nargs="*", default=["China_Drone"])
    ap.add_argument("--bg-ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} not found")
    if args.inspect or not args.map:
        inspect(root)
        if not args.map:
            return
    convert(root, parse_map(args.map), args.splits, args.exclude, args.bg_ratio)


if __name__ == "__main__":
    main()
