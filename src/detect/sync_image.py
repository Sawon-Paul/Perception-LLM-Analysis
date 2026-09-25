"""Produces a single JPG you can upload, and probes the authors' own alignment.

Two jobs:

1. timeline image — frames sampled across the whole video in a strip, with one
   colour bar per candidate offset underneath showing what the labels claim the
   surface was. Written as ONE self-contained .jpg, so it can be shared or
   uploaded without the images going missing.

2. composite probe — video_environment_dataset_left.mp4 shows the dashcam beside
   the sensor plots, so its makers already had an alignment. This locates the
   dashcam panel inside the composite, matches it against video_environment.mp4,
   and reports the time shift between them.

    python -m src.detect.sync_image --trace "PVS 2"
    python -m src.detect.sync_image --trace "PVS 2" --composite
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402

RGB = {"asphalt": (59, 111, 212), "paved": (90, 134, 221),
       "dirt": (138, 90, 43), "cobblestone": (212, 130, 59),
       "unpaved": (160, 106, 53), "unknown": (68, 68, 68)}

CELL_W, CELL_H, BAR_H, LABEL_W = 150, 84, 26, 78


# ------------------------------------------------------------------ shared

def surface_series(trace: str) -> tuple[pd.DataFrame, float]:
    pvs = cfg.pvs_dir(trace)
    sens = pd.read_csv(pvs / "dataset_gps_mpu_left.csv", usecols=["timestamp"])
    labels = pd.read_csv(pvs / "dataset_labels.csv")
    if len(sens) != len(labels):
        raise ValueError(f"row mismatch: sensors={len(sens)} labels={len(labels)}")
    df = pd.concat([sens, labels], axis=1).sort_values("timestamp").reset_index(drop=True)

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
    return df[["timestamp", "surface"]], float(df["timestamp"].min())


def duration(video: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                   "format=duration", "-of", "csv=p=0", str(video)])
    return float(out.decode().strip())


def frame_at(video: Path, t: float, width: int = 300) -> Image.Image | None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:-1", "-pix_fmt", "yuvj420p",
         "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    import io
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


# ------------------------------------------------------------- timeline jpg

def build_timeline(trace: str, offsets: list[float], n: int, per_row: int,
                   stretch: bool = True) -> Path:
    pvs = cfg.pvs_dir(trace)
    video = pvs / "video_environment.mp4"
    lab, t0 = surface_series(trace)
    vdur = duration(video)

    print(f"video {vdur:.1f}s | surfaces {lab['surface'].value_counts().to_dict()}")

    frames = []
    for i in range(n):
        vt = vdur * i / (n - 1) * 0.995
        im = frame_at(video, vt, CELL_W * 2)
        if im is not None:
            frames.append((vt, im.resize((CELL_W, CELL_H))))
        print(f"  frame {i + 1}/{n}", end="\r", flush=True)
    print(f"\n{len(frames)} frames")

    ts = lab["timestamp"].to_numpy()
    surf = lab["surface"].to_numpy()

    def surface_at(want: float) -> str:
        if want < ts[0] or want > ts[-1]:
            return "unknown"
        return surf[min(int(ts.searchsorted(want)), len(surf) - 1)]

    span_s = float(ts[-1] - ts[0])

    def surface_stretched(vt: float) -> str:
        """Linear map: video start -> sensor start, video end -> sensor end."""
        return surface_at(t0 + span_s * (vt / vdur))

    rows: list[tuple[str, object]] = []
    if stretch:
        rows.append(("STRETCH", surface_stretched))
    for off in offsets:
        rows.append((f"{off:+g}s", (lambda o: lambda vt: surface_at(t0 + o + vt))(off)))

    chunks = [frames[i:i + per_row] for i in range(0, len(frames), per_row)]
    block_h = CELL_H + 14 + len(rows) * BAR_H + 22
    W = LABEL_W + per_row * CELL_W
    H = 54 + len(chunks) * block_h

    img = Image.new("RGB", (W, H), (14, 14, 14))
    d = ImageDraw.Draw(img)
    d.text((10, 12), f"{trace} — STRETCH row (green) is the candidate answer",
           fill=(240, 240, 240))
    x = 10
    for name in ("asphalt", "dirt", "cobblestone", "unknown"):
        d.rectangle([x, 30, x + 12, 42], fill=RGB[name])
        d.text((x + 17, 31), name, fill=(170, 170, 170))
        x += 22 + len(name) * 7

    y = 54
    for chunk in chunks:
        for ci, (vt, im) in enumerate(chunk):
            img.paste(im, (LABEL_W + ci * CELL_W, y))
            d.text((LABEL_W + ci * CELL_W + 4, y + CELL_H + 2),
                   f"{vt:.0f}s", fill=(130, 130, 130))
        d.text((6, y + CELL_H // 2 - 5), "video", fill=(138, 180, 248))

        by = y + CELL_H + 14
        for name, fn in rows:
            hot = name == "STRETCH"
            d.text((6, by + 7), name, fill=(120, 230, 140) if hot else (138, 180, 248))
            for ci, (vt, _) in enumerate(chunk):
                s = fn(vt)
                d.rectangle([LABEL_W + ci * CELL_W, by,
                             LABEL_W + (ci + 1) * CELL_W - 2, by + BAR_H - 2],
                            fill=RGB.get(s, RGB["unknown"]))
            if hot:
                d.rectangle([LABEL_W - 2, by - 2, W - 2, by + BAR_H - 1],
                            outline=(120, 230, 140), width=2)
            by += BAR_H
        y += block_h

    out = cfg.DATA / "sync_timeline" / cfg.slug(trace)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "timeline.jpg"
    img.save(path, quality=88)
    print(f"\n-> {path}")
    print("Upload that .jpg — it is one self-contained image.")
    return path


# --------------------------------------------------------- composite probe

def probe_composite(trace: str, samples: int = 6, search_s: float = 200.0) -> None:
    """Find the time shift between the composite's dashcam panel and the raw video."""
    pvs = cfg.pvs_dir(trace)
    env = pvs / "video_environment.mp4"
    comp = pvs / "video_environment_dataset_left.mp4"
    if not comp.exists():
        print(f"{comp.name} not found")
        return

    d_env, d_comp = duration(env), duration(comp)
    lab, _ = surface_series(trace)
    span = float(lab["timestamp"].max() - lab["timestamp"].min())
    print(f"\nenvironment video : {d_env:.2f} s")
    print(f"composite video   : {d_comp:.2f} s")
    print(f"sensor span       : {span:.2f} s")
    print(f"composite - env   : {d_comp - d_env:+.2f} s")
    print(f"composite - sensor: {d_comp - span:+.2f} s")

    if abs(d_comp - d_env) < 2:
        print("\n  Composite matches the environment video length. The plots were "
              "most likely stretched onto the video, not the other way round.")
    elif abs(d_comp - span) < 2:
        print("\n  Composite matches the SENSOR span. The dashcam was stretched or "
              "padded to cover the whole log — worth confirming below.")

    # locate the dashcam panel: composites usually place it on the left half
    probe = frame_at(comp, d_comp * 0.5, 640)
    if probe is None:
        print("could not read the composite")
        return
    w, h = probe.size
    print(f"\ncomposite frame is {w}x{h}; assuming the dashcam panel is the left half")

    import cv2

    def gray(im: Image.Image, box=None) -> np.ndarray:
        if box:
            im = im.crop(box)
        return np.asarray(im.convert("L").resize((96, 54)), dtype=np.float32)

    shifts = []
    for k in range(1, samples + 1):
        ct = d_comp * k / (samples + 1)
        cf = frame_at(comp, ct, 640)
        if cf is None:
            continue
        panel = gray(cf, (0, 0, w // 2, h))

        best, best_t = -2.0, None
        for et in np.arange(max(0, ct - search_s), min(d_env, ct + search_s), 2.0):
            ef = frame_at(env, float(et), 192)
            if ef is None:
                continue
            score = float(cv2.matchTemplate(gray(ef), panel, cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best:
                best, best_t = score, float(et)
        if best_t is not None:
            shifts.append((ct, best_t, best))
            print(f"  composite t={ct:7.1f}s  ->  environment t={best_t:7.1f}s  "
                  f"(match {best:.3f}, shift {best_t - ct:+.1f}s)")

    if len(shifts) >= 3:
        good = [s for s in shifts if s[2] > 0.5]
        if good:
            deltas = [b - c for c, b, _ in good]
            print(f"\n  {len(good)} confident matches, shift "
                  f"{np.mean(deltas):+.1f} s (sd {np.std(deltas):.1f})")
            if np.std(deltas) < 3:
                print("  Consistent shift — the composite and the raw video are "
                      "related by a constant offset.")
            else:
                print("  Inconsistent shift — likely time-scaled, not offset.")
        else:
            print("\n  No confident matches. The panel may not be the left half; "
                  "open the composite and check the layout.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=cfg.DEFAULT_TRACE)
    ap.add_argument("--offsets", default="-90,-60,-30,-15,0,15,30,60,90")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--per-row", type=int, default=16)
    ap.add_argument("--no-stretch", action="store_true",
                    help="omit the linear stretch row")
    ap.add_argument("--composite", action="store_true",
                    help="also probe the composite video (slow, a few minutes)")
    args = ap.parse_args()

    build_timeline(args.trace, [float(x) for x in args.offsets.split(",")],
                   args.n, args.per_row, stretch=not args.no_stretch)
    if args.composite:
        probe_composite(args.trace)


if __name__ == "__main__":
    main()
