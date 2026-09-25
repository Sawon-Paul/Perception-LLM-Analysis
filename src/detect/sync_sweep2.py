"""Decide the offset by eye, using surface type as the test.

Spotting a pothole in a 200 px thumbnail is hard. Telling dirt from asphalt is
trivial. dataset_labels.csv records the surface at every sample, so:

  pick moments labelled dirt or cobblestone, and moments labelled asphalt,
  pull the frames each candidate offset would pair with them, and look for the
  offset where the labels match what the road actually looks like.

--transitions is stronger still: it finds the moments the surface changes and
shows frames around them. A wrong offset puts the change in the wrong place.

    python -m src.detect.sync_sweep2 --trace "PVS 2"
    python -m src.detect.sync_sweep2 --trace "PVS 2" --transitions
    python -m src.detect.sync_sweep2 --trace "PVS 2" --offsets -40,-20,0,20,40
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

HEAD = """<!doctype html><meta charset="utf-8"><title>sync sweep</title>
<style>
body{font-family:system-ui,sans-serif;margin:20px;background:#101010;color:#eee}
h1{font-size:20px;margin-bottom:4px}
.note{color:#f2b544;font-size:13px;margin:10px 0 22px;max-width:900px;line-height:1.6}
.row{margin-bottom:28px;border-top:1px solid #2a2a2a;padding-top:14px}
.lbl{font-size:16px;font-weight:600;margin-bottom:8px;color:#8ab4f8}
.strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:8px}
.cell{flex:0 0 210px}
.cell img{width:210px;height:118px;object-fit:cover;border-radius:4px;background:#000;
          display:block;border:2px solid transparent}
.cell.rough img{border-color:#e5734f}
.cell.smooth img{border-color:#4f8fe5}
.cap{font-size:11px;color:#999;margin-top:4px;line-height:1.45}
.tag{font-weight:600}
.rough .tag{color:#e5734f}
.smooth .tag{color:#4f8fe5}
</style>
"""

INTRO = """<h1>Sync sweep &mdash; {trace}</h1>
<p class="note">Each row is one candidate offset. Every thumbnail is the video frame
that offset would pair with a labelled sensor moment.<br><br>
<span style="color:#e5734f;font-weight:600">Orange</span> = the labels say dirt,
cobblestone or unpaved here.
<span style="color:#4f8fe5;font-weight:600">Blue</span> = the labels say asphalt.
<br><br>
<b>Find the row where the orange frames really do show unpaved road and the blue ones
show asphalt.</b> That row's offset is correct. If every row is a jumble, the video
and sensors cannot be aligned this way.</p>
"""


def load_labelled(trace: str) -> pd.DataFrame:
    """Sensor timestamps joined to the surface labels (row-index aligned)."""
    pvs = cfg.pvs_dir(trace)
    sens = pd.read_csv(pvs / "dataset_gps_mpu_left.csv",
                       usecols=["timestamp", "acc_z_below_suspension", "speed"])
    labels = pd.read_csv(pvs / "dataset_labels.csv")
    if len(sens) != len(labels):
        raise ValueError(f"row mismatch: sensors={len(sens)} labels={len(labels)}")

    df = pd.concat([sens, labels], axis=1).sort_values("timestamp").reset_index(drop=True)
    z = df["acc_z_below_suspension"]
    df["jolt"] = (z - z.rolling(100, center=True, min_periods=1).mean()).abs()

    def surface(r) -> str:
        for c in ("cobblestone_road", "dirt_road", "asphalt_road"):
            if r.get(c) == 1:
                return c.replace("_road", "")
        if r.get("unpaved_road") == 1:
            return "unpaved"
        if r.get("paved_road") == 1:
            return "paved"
        return "unknown"

    df["surface"] = df.apply(surface, axis=1)
    df["rough"] = df["surface"].isin(["dirt", "cobblestone", "unpaved"])
    return df


def pick_contrast(df: pd.DataFrame, n_each: int, min_gap_s: float = 15.0) -> pd.DataFrame:
    """n_each rough moments and n_each asphalt moments, well spread in time."""
    picked = []
    for rough in (True, False):
        sub = df[(df["rough"] == rough) & (df["speed"] > 2.0)]
        if sub.empty:
            continue
        sub = sub.nlargest(len(sub) // 2 + 1, "jolt") if rough else sub.sample(
            min(len(sub), 4000), random_state=0)
        used = []
        for _, r in sub.iterrows():
            ts = float(r["timestamp"])
            if all(abs(ts - u) > min_gap_s for u in used):
                used.append(ts)
                picked.append(r)
            if len(used) >= n_each:
                break
    return pd.DataFrame(picked).sort_values("timestamp")


def pick_transitions(df: pd.DataFrame, n: int, pad_s: float = 6.0) -> pd.DataFrame:
    """Moments where the surface label changes, plus a frame either side."""
    change = df.index[df["surface"].ne(df["surface"].shift())].tolist()[1:]
    out, used = [], []
    for i in change:
        ts = float(df.loc[i, "timestamp"])
        if any(abs(ts - u) < 30 for u in used):
            continue
        used.append(ts)
        for d in (-pad_s, pad_s):
            near = df.iloc[(df["timestamp"] - (ts + d)).abs().argsort()[:1]]
            if not near.empty:
                out.append(near.iloc[0])
        if len(used) >= n:
            break
    return pd.DataFrame(out)


def grab(video: Path, t: float, dst: Path) -> bool:
    if t < 0:
        return False
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                    "-frames:v", "1", "-q:v", "3", "-vf", "scale=420:-1", str(dst)],
                   check=False)
    return dst.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--offsets", default="-60,-30,-15,0,15,30,60")
    ap.add_argument("--transitions", action="store_true",
                    help="use surface-change moments instead of contrasting samples")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    offsets = [float(x) for x in args.offsets.split(",")]
    pvs = cfg.pvs_dir(args.trace)
    video = pvs / "video_environment.mp4"

    df = load_labelled(args.trace)
    counts = df["surface"].value_counts().to_dict()
    print(f"surface distribution: {counts}")
    if len(counts) < 2:
        print("\nOnly one surface type in this trace — the contrast test cannot work "
              "here. Try another trace, or use --transitions on one that varies.")
        return

    ev = (pick_transitions(df, args.n) if args.transitions
          else pick_contrast(df, args.n))
    if ev.empty:
        print("no usable events found")
        return
    print(f"{len(ev)} events, {len(offsets)} offsets = {len(ev) * len(offsets)} frames")

    t0 = float(df["timestamp"].min())
    out_dir = cfg.DATA / "sync_sweep" / cfg.slug(args.trace)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    rows = []
    for oi, off in enumerate(offsets):
        cells = []
        for ei, (_, r) in enumerate(ev.iterrows()):
            vt = float(r["timestamp"]) - t0 - off
            img = out_dir / f"o{oi:02d}_e{ei:02d}.jpg"
            if not grab(video, vt, img):
                continue
            cls = "rough" if r["rough"] else "smooth"
            cells.append(
                f'<div class="cell {cls}"><img src="{img.resolve().as_uri()}">'
                f'<div class="cap"><span class="tag">{r["surface"]}</span> '
                f'&middot; t={vt:.0f}s<br>jolt {r["jolt"]:.1f} '
                f'&middot; {r["speed"] * 3.6:.0f} km/h</div></div>')
        rows.append(f'<div class="row"><div class="lbl">offset {off:+g} s</div>'
                    f'<div class="strip">{"".join(cells)}</div></div>')
        print(f"  offset {off:+g}s: {len(cells)} frames", flush=True)

    sheet = out_dir / "sweep.html"
    sheet.write_text(HEAD + INTRO.replace("{trace}", args.trace) + "\n".join(rows),
                     encoding="utf-8")
    print(f"\n-> {sheet}")
    print("Orange border = labels say unpaved. Blue = labels say asphalt.")
    print("Find the row where the pictures agree with the labels.")


if __name__ == "__main__":
    main()
