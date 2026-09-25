# Results — PVS 2

Hand-labelled evaluation set: **148 detections, 5 true faults, 143 false positives**.

Rates carry Wilson 95% intervals. The two halves of this set are not equally reliable: the false-positive rate rests on 143 negatives and is reasonably tight, while recall rests on 5 positives and is not. Read them accordingly.


## Main comparison

| method | false positives | FP rate [95% CI] | recall [95% CI] | precision | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| YOLO alone (conf>=0.5) | 92 / 143 | 64.3% [56.2–71.7] | 40.0% [11.8–76.9] | 0.021 | 0.040 | 0.224 |
| Sensor rule baseline | 7 / 143 | 4.9% [2.4–9.8] | 0.0% [0.0–43.4] | 0.000 | 0.000 | 0.033 |
| + Qwen2.5-VL-3B | 21 / 143 | 14.7% [9.8–21.4] | 60.0% [23.1–88.2] | 0.125 | 0.207 | 0.370 |
| + Qwen2.5-VL-7B | 2 / 143 | 1.4% [0.4–5.0] | 60.0% [23.1–88.2] | 0.600 | 0.600 | 0.427 |

## Output validity

| model | self-inconsistent JSON | parse failures |
|---|---|---|
| Qwen2.5-VL-3B | 67 / 413 (16.2%) | 0 |
| Qwen2.5-VL-7B | 14 / 413 (3.4%) | 0 |

Self-inconsistent means the model returned a verdict that contradicted its own severity or fault type. Those are corrected in code rather than trusted, so the rate is a measure of output reliability, not of accuracy.


## Operating point

The 0.5 threshold was a default, not a choice. The sweep below shows what was available. No threshold is fitted on this set: with 5 positives, any fit would simply memorise them.


**Qwen2.5-VL-3B**

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.1 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.2 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.3 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.4 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.5 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.6 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.7 | 3 | 21 | 2 | 0.125 | 0.600 |
| 0.8 | 3 | 6 | 2 | 0.333 | 0.600 |
| 0.9 | 2 | 1 | 3 | 0.667 | 0.400 |

**Qwen2.5-VL-7B**

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.1 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.2 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.3 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.4 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.5 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.6 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.7 | 3 | 2 | 2 | 0.600 | 0.600 |
| 0.8 | 2 | 1 | 3 | 0.667 | 0.400 |
| 0.9 | 0 | 0 | 5 | 0.000 | 0.000 |

## Limitations

- Only 5 true faults in the labelled set. Recall and precision have wide intervals; the false-positive rate is the well-measured quantity.

- Labels for this set were produced with AI assistance under a written rubric (cobblestone joints, shadows, sealant lines and completed repairs count as negatives). State this in the write-up.

- Fusion weights and the Phase 1 gate are unfitted defaults, reported as such rather than presented as tuned.
