# LOPO route-threshold sensitivity audit

All processed-window LOPO event folds resolve to sparse_diverse. The audit reports margins to the cold-start, extreme-sparse, and dense-low-diversity gates and counts route flips under simple scalar threshold perturbations.

Regimes: sparse_diverse.
Minimum margin to cold-start rate gate: 0.0072.
Minimum margin to cold-start count gate: 7.
Minimum margin to extreme-sparse rate gate: 0.0222.
Minimum margin to dense rate gate: 0.0065.
Minimum family-count margin away from dense gate: 25.0.

| Threshold scale | Route flips |
|---:|---:|
| 0.8x | 0/7 |
| 0.9x | 0/7 |
| 1.1x | 0/7 |
| 1.2x | 0/7 |
