# VERMA-FL-UL — LLM verification pipeline

Paired dashcam video + IMU + GPS from the PVS dataset, two YOLO detectors, and a
two-phase LLM verifier. Everything below assumes you are starting from nothing.

---

## 1. Prerequisites

| Thing | Why | Check |
|---|---|---|
| Python 3.11 | bitsandbytes and ultralytics wheels | `python3.11 --version` |
| NVIDIA driver ≥ 570 | required by the CUDA 12.8 runtime | `nvidia-smi` |
| ffmpeg + ffprobe | frame extraction | `ffmpeg -version` |
| Git | version control | `git --version` |

Install ffmpeg if missing:

```bash
# Ubuntu / WSL
sudo apt update && sudo apt install -y ffmpeg
# Windows
winget install Gyan.FFmpeg
```

`nvidia-smi` prints the highest CUDA version your **driver** supports. It can be
higher than 12.8 — that is fine. Lower means update the driver.

---

## 2. Virtual environment

```bash
cd verma
python3.11 -m venv venv

source venv/bin/activate        # Linux / WSL / macOS
# venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
```

Install torch **first and separately**. If you skip this, `ultralytics` pulls the
CPU build of torch and CUDA silently never runs:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify before going further:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected: a version string, `12.8`, and `True`. If it prints `False`, your driver
is older than the runtime — update the driver, do not downgrade torch.

---

## 3. Secrets

```bash
cp .env.example .env
```

Edit `.env`:

```
DASHSCOPE_API_KEY=sk-...           # from dashscope.console.aliyun.com
OVERPASS_CONTACT=you@example.com   # Overpass asks for a contact in the User-Agent
PVS_DIR=/absolute/path/to/PVS 2
```

`.env` is gitignored. If a key has ever been committed anywhere, rotate it now —
deleting the line does not help, git history keeps it forever.

---

## 4. Put your files in place

```
weights/yolo11s_pothole.pt      your trained pothole model
weights/yolo11s_sign.pt         your trained sign model
```

Leave the PVS folder wherever it is and point `PVS_DIR` at it.

**Only `video_environment.mp4` is road footage.** `video_dataset_left.mp4` and
`video_dataset_right.mp4` are animated sensor plots; the `video_environment_dataset_*`
files are composites of the two. YOLO on a plot render produces nothing.

---

## 5. Smoke test

Never run the full pipeline first. Check the plumbing with a tiny run:

```bash
python -m src.pipeline.run --side left --smoke
```

This extracts at 1 fps and caps each LLM stage at 50 events. Takes minutes, not hours.

---

## 6. The one thing you must verify

Stage 0 prints a sync check:

```
  video duration : 1234.56 s
  sensor span    : 1234.10 s
  difference     : 0.46 s
  OK: durations agree within 2 s.
```

The pipeline assumes **video frame 0 == the first sensor timestamp**. That is an
assumption, not a fact. If the difference exceeds 2 s, every downstream join is
misaligned and every result is meaningless.

Manual anchor when it fails:

1. Open `dataset_labels.csv`, find a row where `speed_bump_asphalt == 1`.
2. Its offset from `t0` = that row index ÷ 100 (sensors run at 100 Hz).
3. Scrub `video_environment.mp4` to that second. Do you see a speed bump?
4. If the bump appears N seconds later, rerun with `--t0-offset N`.

Do this before extracting thousands of frames.

---

## 7. Full run

```bash
python -m src.pipeline.run --side left
```

Stages, in order:

| # | Stage | Output |
|---|---|---|
| 0 | extract frames, join sensors + labels | `data/raw/frames_left.csv` |
| 1 | both YOLO models | `data/detections/dets_left.jsonl` |
| 2 | 5-stream context vector + crops | `data/events/events_left.jsonl` |
| 3 | data-derived thresholds + rule baseline | `data/thresholds.json` |
| 4 | Phase 1 text-only prior | `data/events/events_left_p1.jsonl` |
| 5 | Phase 2 multimodal verify + fusion | `data/events/events_left_p2.jsonl` |

A failed stage prints the exact resume command. Nothing is recomputed: Overpass
responses and Phase 1 answers are cached to `data/cache/`, so reruns are free.

---

## 8. Evaluation — do not skip this

The PVS labels describe road *quality over a stretch*. They never say whether one
specific YOLO box is a real defect. Per-detection precision and recall need labels
only you can make.

```bash
python -m src.eval.make_eval_set --side left --n 300
```

Open `data/eval/sheet.html`, scroll the crops, and fill `is_real_fault` (1 or 0) in
`data/eval/labels.csv`. Label before looking at any model score — otherwise the
evaluation is contaminated.

```bash
python -m src.eval.make_eval_set --side left --check    # validate
python -m src.eval.metrics --side left                  # the results table
```

You get precision, recall, F1 and PR-AUC for five methods:

- `yolo_only` — detector confidence alone
- `rule` — the free sensor + confidence baseline
- `phase1_only` — text prior, no image
- `phase2_only` — multimodal, no prior
- `fused` — the full pipeline

**The two comparisons that decide your architecture:**

`fused` vs `rule`. A rule costing nothing beating your LLM pipeline is a real
result — report it, then reconsider the architecture.

`fused` vs `phase2_only`. This is the ablation that justifies having two phases at
all. Run it with:

```bash
python -m src.llm.phase2_verify --side left --no-gate
python -m src.eval.metrics --side left --events data/events/events_left_p2_nogate.jsonl
```

If Phase 1 adds no accuracy, its justification has to be cost — so measure the cost:
`phase1_prior` prints how much Phase 2 work the gate removed.

---

## 9. Known placeholders you must fix before writing up

- **Fusion weights** in `src/llm/phase2_verify.py:fuse()` are `0.25 / 0.25 / 0.50`.
  These are arbitrary. Fit them on your labelled set (logistic regression on the
  three scores) and hardcode the fitted values. Never defend an arbitrary weight.
- **`PHASE1_GATE = 0.30`** in `configs/config.py` is a guess. Pick it from the
  precision-recall curve of `phase1_only` at the recall you are willing to lose.
- **`EXTRACT_FPS = 4.0`** is a guess. At 40 km/h a pothole is visible about 1 s,
  so 4 frames. Raise it if detections get missed; do not use the native 30 fps.

---

## 10. Layout

```
configs/config.py                 paths, thresholds, model names
src/detect/extract_frames.py      video -> timestamped frames + sensor join
src/detect/dump_detections.py     both YOLO models -> JSONL
src/context/overpass_client.py    road class, maxspeed, junction (replaces Nominatim)
src/context/builder.py            the 5-stream context vector
src/llm/phase1_prior.py           Qwen-plus text-only prior
src/llm/phase2_verify.py          Qwen2.5-VL-3B 4-bit verify + fusion
src/eval/baseline.py              data-derived thresholds + rule baseline
src/eval/make_eval_set.py         manual labelling helper
src/eval/metrics.py               the results table
src/pipeline/run.py               runs everything
src/utils/                        cache, JSONL io
```

---

## 11. Troubleshooting

| Symptom | Cause |
|---|---|
| `torch.cuda.is_available()` is False | CPU torch installed, or driver too old |
| `ffprobe: not found` | ffmpeg not installed or not on PATH |
| `row count mismatch: sensors=N labels=M` | wrong side, or a truncated CSV |
| `<95% matched` after stage 0 | wrong `t0`, or sync assumption broken |
| Overpass returns nothing anywhere | rural trace with sparse OSM coverage — check one point in a browser first |
| `DASHSCOPE_API_KEY not set` | `.env` missing or in the wrong directory |
| CUDA OOM in Phase 2 | lower `max_pixels` in `phase2_verify.py` |

---

## 12. Order of work

1. venv + torch check
2. sync verification ← everything depends on this
3. smoke run
4. **200 GPS points through Overpass — count how many tags come back non-null.**
   Rural Santa Catarina may have almost no `maxspeed` or `lanes`. If stream 3 is
   mostly empty, the road-context stream is not a real contribution and you should
   find that out now, not in month three.
5. full run
6. label 300 crops
7. metrics + ablation
8. only then: fix weights, write results
