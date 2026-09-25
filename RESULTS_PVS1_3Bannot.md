# Results — PVS 1

Hand-labelled evaluation set: **958 detections, 162 true faults, 796 false positives**.

Rates carry Wilson 95% intervals. The two halves of this set are not equally reliable: the false-positive rate rests on 796 negatives and is reasonably tight, while recall rests on 162 positives and is not. Read them accordingly.


## Main comparison

| method | false positives | FP rate [95% CI] | recall [95% CI] | precision | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| YOLO alone (conf>=0.5) | 494 / 796 | 62.1% [58.6–65.4] | 48.1% [40.6–55.8] | 0.136 | 0.212 | 0.142 |
| Sensor rule baseline | 34 / 796 | 4.3% [3.1–5.9] | 2.5% [1.0–6.2] | 0.105 | 0.040 | 0.290 |
| + Qwen2.5-VL-3B | 102 / 796 | 12.8% [10.7–15.3] | 40.7% [33.5–48.4] | 0.393 | 0.400 | 0.433 |
| + Qwen2.5-VL-7B | 3 / 796 | 0.4% [0.1–1.1] | 16.7% [11.7–23.2] | 0.900 | 0.281 | 0.536 |

## Output validity

| model | self-inconsistent JSON | parse failures |
|---|---|---|
| Qwen2.5-VL-3B | 139 / 977 (14.2%) | 0 |
| Qwen2.5-VL-7B | 15 / 977 (1.5%) | 0 |

Self-inconsistent means the model returned a verdict that contradicted its own severity or fault type. Those are corrected in code rather than trusted, so the rate is a measure of output reliability, not of accuracy.


## Operating point

The 0.5 threshold was a default, not a choice. The sweep below shows what was available. No threshold is fitted on this set: with 162 positives, any fit would simply memorise them.


**Qwen2.5-VL-3B**

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.1 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.2 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.3 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.4 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.5 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.6 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.7 | 66 | 102 | 96 | 0.393 | 0.407 |
| 0.8 | 51 | 41 | 111 | 0.554 | 0.315 |
| 0.9 | 16 | 9 | 146 | 0.640 | 0.099 |

**Qwen2.5-VL-7B**

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.0 | 27 | 3 | 135 | 0.900 | 0.167 |
| 0.1 | 27 | 3 | 135 | 0.900 | 0.167 |
| 0.2 | 27 | 3 | 135 | 0.900 | 0.167 |
| 0.3 | 27 | 3 | 135 | 0.900 | 0.167 |
| 0.4 | 27 | 3 | 135 | 0.900 | 0.167 |
| 0.5 | 27 | 3 | 135 | 0.900 | 0.167 |
| 0.6 | 27 | 3 | 135 | 0.900 | 0.167 |
| 0.7 | 24 | 2 | 138 | 0.923 | 0.148 |
| 0.8 | 18 | 1 | 144 | 0.947 | 0.111 |
| 0.9 | 2 | 1 | 160 | 0.667 | 0.012 |

## Limitations

- Only 162 true faults in the labelled set. Recall and precision have wide intervals; the false-positive rate is the well-measured quantity.

- Labels for this set were produced with AI assistance under a written rubric (cobblestone joints, shadows, sealant lines and completed repairs count as negatives). State this in the write-up.

- Fusion weights and the Phase 1 gate are unfitted defaults, reported as such rather than presented as tuned.
