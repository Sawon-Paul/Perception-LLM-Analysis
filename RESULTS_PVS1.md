# Results — PVS 1

Hand-labelled evaluation set: **958 detections, 59 true faults, 899 false positives**.

Rates carry Wilson 95% intervals. The two halves of this set are not equally reliable: the false-positive rate rests on 899 negatives and is reasonably tight, while recall rests on 59 positives and is not. Read them accordingly.


## Main comparison

| method | false positives | FP rate [95% CI] | recall [95% CI] | precision | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| YOLO alone (conf>=0.5) | 543 / 899 | 60.4% [57.2–63.5] | 49.2% [36.8–61.6] | 0.051 | 0.092 | 0.062 |
| Sensor rule baseline | 35 / 899 | 3.9% [2.8–5.4] | 5.1% [1.7–13.9] | 0.079 | 0.062 | 0.147 |
| + Qwen2.5-VL-3B | 141 / 899 | 15.7% [13.5–18.2] | 45.8% [33.7–58.3] | 0.161 | 0.238 | 0.222 |
| + Qwen2.5-VL-7B | 6 / 899 | 0.7% [0.3–1.4] | 40.7% [29.1–53.4] | 0.800 | 0.539 | 0.548 |

## Output validity

| model | self-inconsistent JSON | parse failures |
|---|---|---|
| Qwen2.5-VL-3B | 139 / 977 (14.2%) | 0 |
| Qwen2.5-VL-7B | 15 / 977 (1.5%) | 0 |

Self-inconsistent means the model returned a verdict that contradicted its own severity or fault type. Those are corrected in code rather than trusted, so the rate is a measure of output reliability, not of accuracy.


## Operating point

The 0.5 threshold was a default, not a choice. The sweep below shows what was available. No threshold is fitted on this set: with 59 positives, any fit would simply memorise them.


**Qwen2.5-VL-3B**

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.1 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.2 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.3 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.4 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.5 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.6 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.7 | 27 | 141 | 32 | 0.161 | 0.458 |
| 0.8 | 24 | 68 | 35 | 0.261 | 0.407 |
| 0.9 | 8 | 17 | 51 | 0.320 | 0.136 |

**Qwen2.5-VL-7B**

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 24 | 6 | 35 | 0.800 | 0.407 |
| 0.1 | 24 | 6 | 35 | 0.800 | 0.407 |
| 0.2 | 24 | 6 | 35 | 0.800 | 0.407 |
| 0.3 | 24 | 6 | 35 | 0.800 | 0.407 |
| 0.4 | 24 | 6 | 35 | 0.800 | 0.407 |
| 0.5 | 24 | 6 | 35 | 0.800 | 0.407 |
| 0.6 | 24 | 6 | 35 | 0.800 | 0.407 |
| 0.7 | 21 | 5 | 38 | 0.808 | 0.356 |
| 0.8 | 17 | 2 | 42 | 0.895 | 0.288 |
| 0.9 | 1 | 2 | 58 | 0.333 | 0.017 |

## Limitations

- Only 59 true faults in the labelled set. Recall and precision have wide intervals; the false-positive rate is the well-measured quantity.

- Labels for this set were produced with AI assistance under a written rubric (cobblestone joints, shadows, sealant lines and completed repairs count as negatives). State this in the write-up.

- Fusion weights and the Phase 1 gate are unfitted defaults, reported as such rather than presented as tuned.
