# ATLASv2 review-budget evidence augmentation audit

This audit uses the train-memory retrieved labels exposed in the label-grounded evidence cache as a retrospective, non-human review-budget proxy. The fusion weight is fixed before evaluation and no threshold is tuned on the held-out windows.

| Method | Recall@5% | Prec@5% | Recall@10% | Prec@10% | Recall@20% | Prec@20% | Lead@10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Score only | 0.667 | 0.800 | 0.778 | 0.519 | 1.000 | 0.353 | 14.2 |
| Retrieved-label analogs only | 0.611 | 0.733 | 0.778 | 0.519 | 1.000 | 0.353 | 14.2 |
| Score + retrieved-label analogs | 0.611 | 0.733 | 0.778 | 0.519 | 1.000 | 0.353 | 14.2 |

## Paired incident-block bootstrap: score+analog minus score-only recall

| Budget | Mean delta | Mean 95% CI |
|---|---:|---:|
| 5pct | -0.015 | [-0.056, +0.000] |
| 10pct | -0.011 | [-0.056, +0.000] |
| 20pct | -0.002 | [-0.056, +0.000] |
