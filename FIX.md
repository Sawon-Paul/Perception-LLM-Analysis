# Two fixes

## 1. auto_label.py — replaced

**Circular validation.** It validated the annotator against a label file the
annotator had written, which returned kappa 0.964. That measured self-consistency,
not accuracy. It now refuses unless most of the reference labels are human.

The honest figure for this annotator is **kappa 0.826 against 148 human labels**.
Use that one in the write-up. Discard 0.964.

**Label loss.** Applying to a trace rewrote its file with only the crops in that
run, so 148 hand judgements were lost when the retrained detector produced
different detections. Human labels outside the current run are now carried over.

## 2. report.py — one function

`wilson()` crashes when the positive count exceeds the sample size, which happens
when labels and events come from different detector runs. Replace the start of
its body with:

```python
    if n <= 0 or k < 0:
        return (0.0, 0.0)
    k = min(k, n)          # k > n means the counts came from different sets
    p = k / n
```
