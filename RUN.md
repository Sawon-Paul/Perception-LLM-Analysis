# Run it once

```cmd
cd /d C:\Sawon\LLM
venv\Scripts\activate.bat

python -m src.selftest
python -m src.pipeline.run_all --traces "PVS 2"
python -m src.eval.metrics --trace "PVS 2"
python -m src.eval.window_eval --trace "PVS 2"
```

If the self test passes, any later failure is data or environment, not code.
It runs 13 checks on synthetic data and touches nothing of yours.

All nine traces, several hours, safe to leave overnight:

```cmd
python -m src.pipeline.run_all --skip-done
python -m src.eval.window_eval --all
```

Larger verifier, ~11 GB at 4-bit, fits your 16 GB card. Writes to its own files
so 3B and 7B can be compared:

```cmd
python -m src.llm.phase2_verify --trace "PVS 2" --model 7b
```

---

# What this build does that the earlier ones did not

**Time to impact.** The camera sees a defect while it is still ahead; the wheels
arrive later. A box whose bottom edge sits at image row y lies at
`d = K / (y - y_horizon)`, so the wheels reach it after `tau = d / v`. The
question is no longer "was it rough soon" but "did a jolt occur at t + tau".
Real damage answers yes; constant cobblestone rumble is rough at every offset
and shows no excess at that specific moment. Verified on synthetic data: a real
defect scores +49 percentile points, constant rumble scores 0.

**Camera parameters fitted, and rejected when unsupported.** K and y_horizon are
grid-searched to maximise how much jolts cluster at predicted impact times
versus control times seconds away. On synthetic data with a known camera the fit
recovers tau to 0.10 s. When detections are mostly false positives there is
nothing to align to, and the fit is rejected rather than dressed up — it checks
both the alignment score and whether the implied mounting height is physically
plausible.

**A self test that catches integration breakage.** Two earlier patches crashed
because a call site changed without its signature, and because of a missing
import. Both are now caught before you run anything.

---

# What this build cannot fix

Three measured facts about your data. None is a code problem.

**1. Five positives.** Of 148 hand-labelled crops, 5 show real damage. Every rate
computed on that set moves by a third if one label changes. No difference
between methods can be proven from it.

**2. The dataset's road-quality labels track surface, not damage.**

```
asphalt        0.0% labelled bad   (n=59,329)
cobblestone   23.8% labelled bad   (n=20,737)
dirt          62.6% labelled bad   (n=44,618)
```

Zero bad samples in 59,000 asphalt readings. So window-level scoring against
`bad_road` measures "is this an unpaved road", not "is this road damaged". This
was my proposed fix for problem 1 and it does not work on this data. The code
prints this table and warns you.

**3. Camera-sensor alignment could not be fitted on PVS 2.** Best alignment
+1.64 percentile points against a threshold of 5. The cause is measurable:
calibration assumes detections correspond to something real, and roughly 97% of
yours do not, so the signal is diluted about thirty-fold.

---

# What to report

**The result:**

| method | precision | recall | TP | FP |
|---|---|---|---|---|
| YOLO alone | 2.1% | 40% | 2 | 92 |
| + LLM verifier | 8.7% | 80% | 4 | 42 |

Precision roughly quadrupled and recall doubled at the same time, on 148 crops
with 5 positives. State the counts and the width alongside the rates.

**Three negative results, each measured, each worth reporting:**

- Temporal camera-sensor corroboration could not be calibrated, because the
  detector's false-positive rate leaves no signal to align to (+1.64 against a
  threshold of 5).
- Absolute vibration thresholds do not transfer across surfaces: on cobblestone
  a large jolt reflects the design of the surface, not damage. A raw look-ahead
  jolt made results worse, precision 8.7% to 6.4% and PR-AUC 0.274 to 0.102.
- The dataset's road-quality labels are confounded with surface type and cannot
  serve as ground truth for damage detection.

**The underlying cause:** your detector was trained on asphalt distress; this
route is largely cobblestone, whose joints look like alligator cracking. That is
a domain mismatch, and it is exactly what the LLM verifier is correcting.

Reported straight, with the numbers, this is a defensible thesis. Numbers tuned
against five images are not, and that is what gets found in a viva.

---

# Constants still to justify

| constant | where | status |
|---|---|---|
| fusion weights 0.25 / 0.25 / 0.50 | `phase2_verify.fuse` | arbitrary — fit by logistic regression once you have more labels |
| `PHASE1_GATE` 0.30 | `config.py` | unused until Phase 1 runs |
| `EXTRACT_FPS` 4.0 | `config.py` | reasonable, untested |
| window length 5 s | `window_eval.py` | try 3 s and 10 s |
| camera K, y_horizon | fitted per trace | reported with a fit score, rejected when weak |
