# Rebuild the PVS 2 labels, then re-report

The label file mixed two detector generations: the original model's 413
detections and the retrained model's 25. Rows numbered 1-25 in that mixed file
are not the 25 cards in the sheet, so the pasted digits landed on the wrong rows.

## 1. Set the mixed file aside

```cmd
cd /d C:\Sawon\LLM
move data\eval\labels_PVS2.csv data\eval\labels_PVS2_mixed.csv
```

Nothing is lost: `data\eval\labels.csv.bak` still holds the original 148 human
labels, and `restore_pvs2_labels.py` can rebuild them.

## 2. Fresh label file for the retrained detector's 25 detections

```cmd
python -m src.eval.make_eval_set --trace "PVS 2" --n 30
```

## 3. Paste the verdicts (card order, 25 digits)

```cmd
python -m src.eval.paste_labels --trace "PVS 2" --start 1 --digits 1111111101111111111011111
```

## 4. Check

```cmd
python -m src.eval.make_eval_set --trace "PVS 2" --check
```

Expect `25/25 labelled; 23 positive, 2 negative`. If it says anything else, the
sheet was regenerated in a different order — send the output before continuing.

## 5. Report

```cmd
python -m src.eval.metrics --trace "PVS 2" --events data\events\events_PVS2_left_p2_standalone_7b.jsonl
python -m src.eval.report --trace "PVS 2" --out RESULTS_FINAL.md
```

The new report scores each method on its own detections and prints n and the
positive count per row, so the two detector generations can no longer be mixed
into one denominator.

## Keeping the before/after comparison

The original detector's numbers are already written up in the earlier
`RESULTS.md`: 148 human labels, 5 positives, FP 64.3% down to 1.4% with the 7B
verifier. Keep that file. The two comparisons answer different questions and
belong in different tables:

- **before fine-tuning** — does an LLM verifier fix a detector that fires on
  everything? Yes: FP 64.3% to 1.4%.
- **after fine-tuning** — is the verifier still needed once the detector is
  fixed? On these 25 detections, no: the detector alone reaches precision 0.926
  and recall 1.000, while adding the verifier costs 9 of 25 real faults to
  remove 1 false positive.
