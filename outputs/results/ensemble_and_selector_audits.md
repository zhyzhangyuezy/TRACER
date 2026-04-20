# Ensemble and learned-selector audits

## ATLASv2 held-out-family score ensembles

Uniform and rank ensembles average only fixed-family DLinear, Small-Transformer, and Prefix-Only predictions. The single-family oracle is an ex-post upper bound over choosing one fixed family per seed and is not a deployable stacker.

| Method | AUPRC | Delta vs TRACER |
|---|---:|---:|
| DLinear | 0.723 +/- 0.037 | -0.040 |
| Small-Transformer | 0.713 +/- 0.075 | -0.050 |
| Prefix-Only | 0.669 +/- 0.121 | -0.094 |
| TRACER | 0.763 +/- 0.041 | +0.000 |
| Uniform fixed ensemble | 0.738 +/- 0.055 | -0.026 |
| Rank fixed ensemble | 0.847 +/- 0.079 | +0.084 |
| Single-family oracle | 0.764 +/- 0.032 | +0.001 |

## LOPO learned split-level selectors

Selectors are trained leave-one-fold-out using only the other processed-window LOPO folds and train/dev regime statistics. The single-family oracle is an ex-post upper bound over DLinear, Small-Transformer, and Prefix-Only.

| Selector | Macro AUPRC | Delta vs adaptive |
|---|---:|---:|
| Adaptive policy | 0.423 | +0.000 |
| LOFO ridge selector | 0.354 | -0.069 |
| Nearest-fold selector | 0.372 | -0.051 |
| Global-mean selector | 0.328 | -0.094 |
| Single-family oracle | 0.492 | +0.069 |
