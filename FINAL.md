# Final build — what changed and why

## The three problems this build fixes

**1. The sensor stream was useless, then harmful.**

The camera sees a defect while it is still ahead; the wheels reach it later.
Two earlier attempts got this wrong:

| attempt | what it did | precision | PR-AUC |
|---|---|---|---|
| jolt at the frame's own time | described road already crossed | 8.7% | 0.274 |
| max jolt over the next 3 s | true everywhere on cobblestone | 6.4% | 0.102 |
| **this build** | **jolt at the predicted moment of impact** | to be measured | |

Now the geometry predicts *when*: a box whose bottom edge sits at image row y
lies at distance `d = K / (y - y_horizon)`, so the wheels arrive after
`tau = d / v`. The question becomes "did a jolt occur at t + tau", which real
damage answers yes and constant cobblestone rumble answers no.

`K` and `y_horizon` are **fitted from your data** by maximising how much jolts
cluster at predicted impact times versus control times a few seconds away. On a
synthetic trace with a known camera the fit recovered K=756 against a true 780
and the horizon row to within 1 px, with a median tau error of 0.10 s. If the
fit score is weak on real data the code says so and marks the parameters
unfitted, so nothing downstream pretends they were measured.

**2. Five positives cannot support any claim.**

148 hand-labelled crops with 5 positives means one label changing moves
precision by a third. `window_eval.py` uses ground truth you already own:
`dataset_labels.csv` records good/regular/bad road at every sample. Cut each
drive into 5 s windows and you get thousands of windows and hundreds of
positives across nine traces.

It also prints the share of BAD samples per surface. If bad-road labels sit
almost entirely on one surface, window scores measure surface recognition
rather than damage detection — the code warns you, and that belongs in the
write-up either way.

**3. The rule baseline was reading nothing.**

`baseline.py` wrote to its own file; `metrics.py` read a different one. The
`rule` row scored a flat zero because the join never happened, not because the
rule failed. `metrics.py` now merges it and says how many events matched.

## Running it

```cmd
python -m src.pipeline.run_all --skip-done
python -m src.eval.window_eval --all
python -m src.eval.metrics --trace "PVS 2"
```

Larger verifier (~11 GB at 4-bit, fits your 16 GB card):

```cmd
python -m src.llm.phase2_verify --trace "PVS 2" --model 7b
```

3B and 7B write to separate files so they can be compared.

## What is verified and what is not

Verified on synthetic data with known answers: camera fitting, tau accuracy,
jolt lookup, window scoring, surface-relative ranking, prompt wording, the
metrics join.

Not verified: whether any of it improves your real numbers. That is what the
next run tells you. If the time-to-impact evidence does not help, drop the
sensor stream and report that honestly — a negative result reported straight is
worth more than a third round of tuning against five images.

## Constants still to justify

| constant | where | status |
|---|---|---|
| fusion weights 0.25/0.25/0.50 | `phase2_verify.fuse` | arbitrary — fit by logistic regression on labelled windows |
| `PHASE1_GATE` 0.30 | `config.py` | unused until Phase 1 runs |
| `EXTRACT_FPS` 4.0 | `config.py` | reasonable, untested |
| window length 5 s | `window_eval.py` | test 3 s and 10 s |
| camera K, y_horizon | fitted per trace | reported with a fit score |
