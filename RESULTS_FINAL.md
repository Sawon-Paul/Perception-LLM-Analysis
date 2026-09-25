# Results — PVS 2

Every method is scored only on the labelled detections it actually produced, so the counts differ between rows — a retrained detector emits different detections from the original. The n and positive columns make that explicit.

Rates carry Wilson 95% intervals. A false-positive rate measured on hundreds of negatives is reasonably tight; a recall measured on a handful of positives is not.


## Main comparison

| method | n | positives | false positives | FP rate [95% CI] | recall [95% CI] | precision | F1 | PR-AUC |
|---|---|---|---|---|---|---|---|---|
Detector rows are taken from the run with the most labelled detections (Qwen2.5-VL-7B, 25 of them).

| Detector alone, every detection | 25 | 23 | 2 / 2 | 100.0% [34.2–100.0] | 100.0% [85.7–100.0] | 0.920 | 0.958 | 0.928 |
| Detector alone, conf>=0.5 | 25 | 23 | 1 / 2 | 50.0% [9.5–90.5] | 43.5% [25.6–63.2] | 0.909 | 0.588 | 0.928 |
| Sensor rule baseline | 25 | 23 | 0 / 2 | 0.0% [0.0–65.8] | 0.0% [0.0–14.3] | 0.000 | 0.000 | 0.903 |
| + Qwen2.5-VL-3B | 6 | 6 | 0 / 0 | n/a | 50.0% [18.8–81.2] | 1.000 | 0.667 | 1.000 |
| + Qwen2.5-VL-7B | 25 | 23 | 1 / 2 | 50.0% [9.5–90.5] | 65.2% [44.9–81.2] | 0.938 | 0.769 | 0.974 |

## Output validity

| model | self-inconsistent JSON | parse failures |
|---|---|---|
| Qwen2.5-VL-3B | 67 / 413 (16.2%) | 0 |
| Qwen2.5-VL-7B | 5 / 74 (6.8%) | 0 |

Self-inconsistent means the model contradicted its own severity or fault type. Those are corrected in code, so the rate measures output reliability rather than accuracy.


## Operating point

0.5 was a default, not a choice. No threshold is fitted here: the positive counts are too small for a fit to be anything but memorisation.


**Qwen2.5-VL-3B** (n=6, 6 positive)

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 3 | 0 | 3 | 1.000 | 0.500 |
| 0.2 | 3 | 0 | 3 | 1.000 | 0.500 |
| 0.4 | 3 | 0 | 3 | 1.000 | 0.500 |
| 0.5 | 3 | 0 | 3 | 1.000 | 0.500 |
| 0.6 | 3 | 0 | 3 | 1.000 | 0.500 |
| 0.7 | 3 | 0 | 3 | 1.000 | 0.500 |
| 0.8 | 3 | 0 | 3 | 1.000 | 0.500 |
| 0.9 | 1 | 0 | 5 | 1.000 | 0.167 |

**Qwen2.5-VL-7B** (n=25, 23 positive)

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 15 | 1 | 8 | 0.938 | 0.652 |
| 0.2 | 15 | 1 | 8 | 0.938 | 0.652 |
| 0.4 | 15 | 1 | 8 | 0.938 | 0.652 |
| 0.5 | 15 | 1 | 8 | 0.938 | 0.652 |
| 0.6 | 15 | 1 | 8 | 0.938 | 0.652 |
| 0.7 | 14 | 0 | 9 | 1.000 | 0.609 |
| 0.8 | 3 | 0 | 20 | 1.000 | 0.130 |
| 0.9 | 1 | 0 | 22 | 1.000 | 0.043 |

## Limitations

- Qwen2.5-VL-3B was scored on 6 labelled detections with 6 true faults.

- Qwen2.5-VL-7B was scored on 25 labelled detections with 23 true faults.

- Labels were produced with AI assistance under a written rubric (cobblestone joints, shadows, sealant lines and completed repairs count as negatives) and spot-checked by hand. State this.

- Fusion weights and the Phase 1 gate are unfitted defaults.
