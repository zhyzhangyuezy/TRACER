# Public Mechanism Analysis

## Gate Summary

### test
- `TP`: n=1, mean=0.800, median=0.800, q1=0.800, q3=0.800
- `FP`: n=6, mean=0.880, median=0.873, q1=0.801, q3=0.946
- `FN`: n=20, mean=0.895, median=0.944, q1=0.803, q3=0.994
- `TN`: n=483, mean=0.805, median=0.804, q1=0.651, q3=0.999
- `positive`: n=21, mean=0.891, median=0.944, q1=0.800, q3=0.994
- `negative`: n=489, mean=0.806, median=0.804, q1=0.652, q3=0.999

### Analog Change Summary
- `same_top1_rate`: n=3, mean=0.606, median=0.665, q1=0.574, q3=0.668
- `changed_top1_rate`: n=3, mean=0.394, median=0.335, q1=0.332, q3=0.426
- `same_top1_gate_mean`: n=3, mean=0.806, median=0.719, q1=0.710, q3=0.858
- `changed_top1_gate_mean`: n=3, mean=0.816, median=0.796, q1=0.726, q3=0.896
- `same_top1_score_margin_mean`: n=3, mean=0.008, median=0.024, q1=-0.000, q3=0.024
- `changed_top1_score_margin_mean`: n=3, mean=0.020, median=0.035, q1=0.012, q3=0.037

### test_event_disjoint
- `TP`: n=6, mean=0.917, median=0.954, q1=0.841, q3=0.984
- `FP`: n=0
- `FN`: n=12, mean=0.879, median=0.896, q1=0.788, q3=0.994
- `TN`: n=231, mean=0.824, median=0.804, q1=0.693, q3=0.999
- `positive`: n=18, mean=0.892, median=0.944, q1=0.801, q3=0.994
- `negative`: n=231, mean=0.824, median=0.804, q1=0.693, q3=0.999

### Analog Change Summary
- `same_top1_rate`: n=3, mean=0.490, median=0.518, q1=0.458, q3=0.536
- `changed_top1_rate`: n=3, mean=0.510, median=0.482, q1=0.464, q3=0.542
- `same_top1_gate_mean`: n=3, mean=0.837, median=0.794, q1=0.757, q3=0.896
- `changed_top1_gate_mean`: n=3, mean=0.814, median=0.826, q1=0.722, q3=0.912
- `same_top1_score_margin_mean`: n=3, mean=0.015, median=0.014, q1=-0.009, q3=0.038
- `changed_top1_score_margin_mean`: n=3, mean=0.015, median=0.009, q1=0.006, q3=0.020

## Family Slices

| Split | Family | n | Pos | Prefix AUPRC | Campaign AUPRC | Transformer AUPRC | Campaign-Prefix | Campaign-Transformer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| test | atlasv2/m5 | 16 | 1 | 0.556 +- 0.342 | 0.500 +- 0.000 | 0.833 +- 0.236 | -0.056 | -0.333 |
| test | atlasv2/m6 | 23 | 6 | 0.483 +- 0.087 | 0.481 +- 0.096 | 0.528 +- 0.008 | -0.001 | -0.046 |
| test_event_disjoint | atlasv2/s4 | 10 | 6 | 0.841 +- 0.068 | 0.867 +- 0.000 | 0.889 +- 0.000 | +0.026 | -0.022 |
