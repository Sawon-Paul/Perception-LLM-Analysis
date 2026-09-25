# START HERE

Project location: **`C:\Sawon\LLM`**

Everything below assumes the files sit there. If they are somewhere else, move them
now — a short path with no spaces avoids Windows path-length problems later.

---

## The five clicks

| # | Do this | What it does |
|---|---|---|
| 1 | Double-click **`setup.bat`** | Installs Python venv, torch for CUDA 12.8, all packages |
| 2 | Edit **`.env`** in Notepad | Your API key and the path to your PVS folder |
| 3 | Already done — your models are in **`weights\`** | `yolo11s_pothole.pt`, `yolo11s_sign.pt` |
| 4 | Double-click **`check.bat`** | Tells you exactly what is still missing |
| 5 | Double-click **`smoke.bat`** | Tiny test run of the whole pipeline |

Only after all five are green: **`run_full.bat`**.

---

## Step 2 in detail — the `.env` file

`setup.bat` creates it. Open it:

```cmd
notepad C:\Sawon\LLM\.env
```

Fill in these lines:

```
DASHSCOPE_API_KEY=sk-your-real-key-here
OVERPASS_CONTACT=your.email@example.com
DATASET_DIR=C:/Sawon/LLM/Dataset
DEFAULT_TRACE=PVS 2
```

**Forward slashes in `DATASET_DIR`.** Backslashes are read as escape characters and
the path silently breaks. `C:/Sawon/...` works perfectly on Windows.

No quotes around values. No spaces around the `=`.

`DATASET_DIR` points at the folder holding all nine `PVS n` folders, not at one of
them. `DEFAULT_TRACE` picks which one runs when you do not pass `--trace`.

---

## Working with nine traces

`C:\Sawon\LLM\Dataset` holds `PVS 1` through `PVS 9`. Each is a separate drive:
different road, different conditions, its own video and sensor log.

Every output filename carries the trace name, so traces never overwrite each other:

```
data\raw\frames_PVS2_left.csv
data\events\events_PVS2_left_p2.jsonl
data\thresholds_PVS2.json
```

One trace:

```cmd
run_full.bat "PVS 2"
run_full.bat "PVS 5" right
```

All nine, back to back (only after one has run clean):

```cmd
run_all_traces.bat
```

Traces are not interchangeable. Sync must be verified per trace — each video has
its own start offset.

### Why nine traces matters for your thesis

Your FL design currently says "simulated vehicle clients". Nine independent drives
give you a real non-IID partition instead: each trace is one client, with genuinely
different road surface distributions. That is a stronger claim than simulation, and
it costs you nothing extra — the data is already sitting there.

---

## Working in cmd

Double-click **`shell.bat`**. It opens cmd already in `C:\Sawon\LLM` with the venv
active and prints the common commands.

Doing it by hand instead:

```cmd
cd /d C:\Sawon\LLM
venv\Scripts\activate.bat
```

Your prompt must show `(venv)`:

```
(venv) C:\Sawon\LLM>
```

**No `(venv)` means the venv is not active.** Commands will run against the wrong
Python and fail in confusing ways. This is the single most common problem.

---

## The one thing you must not skip

`smoke.bat` prints a sync check in stage 0:

```
  video duration : 1234.56 s
  sensor span    : 1234.10 s
  difference     : 0.46 s
  OK: durations agree within 2 s.
```

The pipeline assumes video frame 0 lines up with the first sensor reading. That is
an assumption, not a fact. If the difference is over 2 seconds, every join is
shifted and every number in your results chapter is meaningless.

`WINDOWS_SETUP.md` section 9 has the manual fix.

---

## Full command reference

Run these from `shell.bat` or after activating the venv.

```cmd
python -m src.utils.check_setup                    preflight
python -m src.pipeline.run --side left --smoke     tiny run
python -m src.pipeline.run --side left             full run
python -m src.pipeline.run --side left --from-stage 3    resume partway

python -m src.detect.extract_frames --side left    stage 0 alone
python -m src.detect.dump_detections --side left   stage 1 alone
python -m src.context.builder --side left          stage 2 alone
python -m src.eval.baseline --side left            stage 3 alone
python -m src.llm.phase1_prior --side left         stage 4 alone
python -m src.llm.phase2_verify --side left        stage 5 alone

python -m src.eval.make_eval_set --side left --n 300      build label sheet
python -m src.eval.make_eval_set --side left --check      validate labels
python -m src.eval.metrics --side left                    results table
```

---

## Where things land

```
C:\Sawon\LLM\
  data\raw\frames\            extracted video frames
  data\raw\frames_left.csv    frames joined to sensors + labels
  data\raw\crops\             bbox crops for Phase 2
  data\detections\            YOLO output
  data\events\                context vectors, Phase 1, Phase 2
  data\cache\                 Overpass + Phase 1 cache (reruns are free)
  data\eval\                  your label sheet and verdicts
  data\thresholds.json        thresholds derived from your data
```

`data\` is gitignored. It gets large — the extracted frames are the bulk of it.

---

## Read next

- `WINDOWS_SETUP.md` — full Windows walkthrough, every error and its fix
- `README.md` — what each stage does and why, plus the three placeholder values
  you must calibrate before writing up
