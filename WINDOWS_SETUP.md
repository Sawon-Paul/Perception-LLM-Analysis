# Windows setup — Command Prompt (cmd.exe)

Every command here is for **cmd**, not PowerShell. If your prompt looks like
`C:\Users\You>` you are in cmd. If it looks like `PS C:\Users\You>` you are in
PowerShell — type `cmd` and press Enter to switch.

---

## 1. Install the four prerequisites

### Python 3.11

Download from python.org → Windows installer (64-bit), version 3.11.x.

**During install, tick "Add python.exe to PATH".** Miss this and nothing below works.

Do not use 3.12 or 3.13. `bitsandbytes` wheels for Windows lag behind.

Check:

```cmd
py -3.11 --version
```

Want `Python 3.11.x`. If you get `'py' is not recognized`, reinstall with the PATH box ticked.

### ffmpeg

```cmd
winget install Gyan.FFmpeg
```

**Close the cmd window and open a new one.** PATH changes only apply to new windows.

```cmd
ffmpeg -version
ffprobe -version
```

Both must print something. If `winget` is missing, download from
`https://www.gyan.dev/ffmpeg/builds/`, extract to `C:\ffmpeg`, then add
`C:\ffmpeg\bin` to PATH via System Properties → Environment Variables.

### NVIDIA driver

```cmd
nvidia-smi
```

Top-right shows a CUDA version. It must be **12.8 or higher**. Lower means update
your driver from nvidia.com. Higher is fine — the driver is backward compatible.

If `nvidia-smi` is not recognized, you have no NVIDIA GPU or no driver. Phase 2
will not run.

### 7-Zip or built-in extractor

Windows can extract .zip natively. No install needed.

---

## 2. Unpack the project

1. Download `LLM.zip`.
2. Right-click it → **Extract All...**
3. In the destination box type exactly `C:\Sawon` and click Extract.

Windows sometimes appends the zip name to the destination, giving you
`C:\Sawon\LLM\LLM\`. Check for that and move the inner folder up if it happened.

You should end with `C:\Sawon\LLM\` containing `README.md`, `setup.bat`, `src`,
`configs`, `weights`.

Open cmd and go there:

```cmd
cd /d C:\Sawon\LLM
dir
```

`/d` is needed to change drive letters. `dir` should list `README.md`,
`requirements.txt`, `configs`, `src`, `weights`.

**Every command below runs from this folder.** If you close cmd, `cd /d C:\Sawon\LLM` again.

---

## 3. Virtual environment

```cmd
py -3.11 -m venv venv
venv\Scripts\activate.bat
```

Your prompt now starts with `(venv)`:

```
(venv) C:\Sawon\LLM>
```

**No `(venv)` means it did not activate.** Everything after this will install into the
wrong Python. Fix it before continuing.

Upgrade pip:

```cmd
python -m pip install --upgrade pip
```

### Deactivating and coming back

```cmd
deactivate
```

Next session:

```cmd
cd /d C:\Sawon\LLM
venv\Scripts\activate.bat
```

---

## 4. Install torch FIRST

Order matters. Install torch on its own, before requirements.txt:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

This downloads ~2.5 GB. Takes a while.

If you install `requirements.txt` first, `ultralytics` pulls the CPU build of torch
and your GPU silently never runs. You would not get an error — just very slow
inference and `False` below.

Verify:

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output:

```
2.7.0+cu128 12.8 True
```

`False` means the CPU build got installed or your driver is too old. Fix:

```cmd
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 5. Install everything else

```cmd
pip install -r requirements.txt
```

Then confirm torch survived:

```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

Still `True`. If it flipped to `False`, something pulled CPU torch — redo step 4.

### bitsandbytes on Windows

```cmd
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

If this errors, install the Windows build explicitly:

```cmd
pip install bitsandbytes --upgrade --force-reinstall
```

bitsandbytes is only needed for Phase 2 (4-bit quantization). Steps 0-4 of the
pipeline run without it.

---

## 6. Configure secrets

```cmd
copy .env.example .env
notepad .env
```

Fill it in:

```
DASHSCOPE_API_KEY=sk-your-real-key-here
OVERPASS_URL=https://overpass-api.de/api/interpreter
OVERPASS_CONTACT=your.email@example.com
DATASET_DIR=C:/Sawon/LLM/Dataset
DEFAULT_TRACE=PVS 2
```

**Use forward slashes in `DATASET_DIR`.** Backslashes get read as escape characters
and your path will break. `C:/Sawon/...` works fine on Windows.

`DATASET_DIR` is the folder containing `PVS 1` ... `PVS 9`, not one trace folder.

Do not put quotes around the values. Do not put spaces around the `=`.

Save and close Notepad.

Check it loads:

```cmd
python -c "from configs import config as c; print(c.DATASET_DIR, c.DATASET_DIR.exists()); print(c.pvs_dir(c.DEFAULT_TRACE).exists())"
```

Must print your path and `True`. `False` means the path is wrong — check for a
typo or a missing folder level.

---

## 7. Put your YOLO weights in place

Copy your two trained models into `C:\Sawon\LLM\weights\` and rename them:

```
weights\yolo11s_pothole.pt
weights\yolo11s_sign.pt
```

```cmd
dir weights
```

Both must be listed. Exact names matter — the code looks for these.

---

## 8. Smoke test

Never run the full pipeline first.

```cmd
python -m src.pipeline.run --side left --smoke
```

Extracts at 1 fps, caps each LLM stage at 50 events. Minutes, not hours.

---

## 9. Read the sync check

Stage 0 prints:

```
  video duration : 1234.56 s
  sensor span    : 1234.10 s
  difference     : 0.46 s
  OK: durations agree within 2 s.
```

The pipeline assumes video frame 0 lines up with the first sensor timestamp. That
is an assumption. If the difference exceeds 2 s, every join downstream is shifted
and every number in your results chapter is wrong.

To check it by hand:

1. Open `dataset_labels.csv` in Excel.
2. Find a row where `speed_bump_asphalt` is `1`. Note the row number.
3. Sensors run at 100 Hz, so that moment is `row_number / 100` seconds into the trace.
4. Open `video_environment.mp4`, scrub to that second. Do you see a speed bump?
5. If the bump shows up N seconds later, rerun with:

```cmd
python -m src.pipeline.run --side left --smoke --from-stage 0
```

after editing the offset, or run stage 0 directly:

```cmd
python -m src.detect.extract_frames --side left --t0-offset 8
```

Do this before extracting thousands of frames.

---

## 10. Full run

```cmd
python -m src.pipeline.run --side left
```

| Stage | What | Output |
|---|---|---|
| 0 | frames + sensor join | `data\raw\frames_left.csv` |
| 1 | both YOLO models | `data\detections\dets_left.jsonl` |
| 2 | context vector + crops | `data\events\events_left.jsonl` |
| 3 | thresholds + baseline | `data\thresholds.json` |
| 4 | Phase 1 prior | `data\events\events_left_p1.jsonl` |
| 5 | Phase 2 verify | `data\events\events_left_p2.jsonl` |

A failed stage prints its own resume command. Overpass and Phase 1 are cached, so
reruns cost nothing and do not re-bill the API.

Running one stage alone:

```cmd
python -m src.detect.extract_frames --side left
python -m src.detect.dump_detections --side left
python -m src.context.builder --side left
python -m src.eval.baseline --side left
python -m src.llm.phase1_prior --side left
python -m src.llm.phase2_verify --side left
```

---

## 11. Evaluation

```cmd
python -m src.eval.make_eval_set --side left --n 300
start data\eval\sheet.html
notepad data\eval\labels.csv
```

`start` opens the contact sheet in your browser. Scroll the crops. In
`labels.csv`, put `1` for a real fault or `0` for a false positive next to each
detection_id.

Label before looking at any model score. Otherwise the evaluation is contaminated
and the numbers mean nothing.

```cmd
python -m src.eval.make_eval_set --side left --check
python -m src.eval.metrics --side left
```

The ablation that justifies having two phases:

```cmd
python -m src.llm.phase2_verify --side left --no-gate
python -m src.eval.metrics --side left --events data\events\events_left_p2_nogate.jsonl
```

---

## 12. Windows-specific problems

| Error | Fix |
|---|---|
| `'py' is not recognized` | Python installed without PATH — reinstall, tick the box |
| `'ffprobe' is not recognized` | Open a **new** cmd window after installing ffmpeg |
| No `(venv)` in prompt | `venv\Scripts\activate.bat` did not run — check you are in `C:\Sawon\LLM` |
| `torch.cuda.is_available()` is False | CPU torch installed — uninstall and redo step 4 |
| `FileNotFoundError` on PVS path | backslashes in `.env` — switch to forward slashes |
| Outputs from one trace overwritten | you are on an old build — filenames now include the trace tag |
| `ImportError: bitsandbytes` | Phase 2 only; run stages 0-4 meanwhile |
| `CUDA out of memory` | lower `max_pixels` in `src\llm\phase2_verify.py` |
| Path too long errors | move the project to `C:\Sawon`, not a deep Downloads folder |
| `ModuleNotFoundError: configs` | you are not in `C:\Sawon\LLM` — `cd /d C:\Sawon\LLM` |

---

## 13. Session checklist

Every time you sit down:

```cmd
cd /d C:\Sawon\LLM
venv\Scripts\activate.bat
python -c "import torch; print(torch.cuda.is_available())"
```

`(venv)` in the prompt and `True` from that command. Then work.
