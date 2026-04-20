# ATLASv2 processed-window LOPO family audit

Processed-window leave-one-positive-family-out ATLASv2 audit. Folds are family-disjoint and include one positive test family plus disjoint background families; this supplements but does not replace raw-pipeline multi-fold benchmark construction.

## Policy vs fixed families

| Method | Macro AUPRC | 95% CI | Worst fold | Best/tie | Top-two | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive policy | 0.423 | [0.245, 0.633] | 0.180 | 3/7 | 4/7 | 2.21 |
| Small-Transformer | 0.414 | [0.253, 0.600] | 0.157 | 1/7 | 3/7 | 2.43 |
| Prefix-Only retrieval | 0.418 | [0.251, 0.602] | 0.164 | 1/7 | 4/7 | 2.64 |
| DLinear | 0.390 | [0.209, 0.623] | 0.136 | 3/7 | 3/7 | 2.71 |

## Core mechanism stress

| Method | Macro AUPRC | 95% CI | Worst fold | Delta vs bounded | Delta CI |
|---|---:|---:|---:|---:|---:|
| Core bounded | 0.422 | [0.245, 0.633] | 0.180 | +0.000 | [+0.000, +0.000] |
| Linear correction | 0.445 | [0.225, 0.691] | 0.123 | +0.023 | [-0.044, +0.107] |
| No route gates | 0.448 | [0.243, 0.682] | 0.187 | +0.026 | [-0.027, +0.106] |
