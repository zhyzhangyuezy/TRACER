# Clean mechanism ablation audit

All rows use real three-seed reruns or existing seed-matched result files with the same ATLASv2 public split and 20-epoch budget.

| Mechanism | Changed setting | Chron. AUPRC | HF AUPRC | Delta HF | HF AF@5 | HF TTE@1 |
|---|---|---:|---:|---:|---:|---:|
| Full TRACER core | all mechanisms on | 0.575 +/- 0.065 | 0.767 +/- 0.000 | +0.000 | 69.5 +/- 2.6 | 25.3 +/- 1.6 |
| No auxiliary horizon | auxiliary loss off | 0.605 +/- 0.023 | 0.742 +/- 0.045 | -0.024 | 73.3 +/- 1.3 | 26.1 +/- 2.7 |
| No hard negatives | contrastive hard weighting off | 0.607 +/- 0.055 | 0.767 +/- 0.000 | +0.000 | 72.0 +/- 3.7 | 24.2 +/- 0.0 |
| No contrastive loss | contrastive objective off | 0.689 +/- 0.133 | 0.722 +/- 0.119 | -0.044 | 73.9 +/- 1.6 | 25.8 +/- 1.4 |
| No correction | final score is base fusion | 0.571 +/- 0.019 | 0.767 +/- 0.000 | +0.000 | 71.6 +/- 4.8 | 26.1 +/- 2.7 |
| Linear correction | unbounded linear logit gap | 0.552 +/- 0.066 | 0.783 +/- 0.012 | +0.017 | 71.4 +/- 3.5 | 27.8 +/- 2.6 |
| Forecast-only base gate | base gate fixed to 1 | 0.573 +/- 0.075 | 0.773 +/- 0.013 | +0.007 | 68.3 +/- 3.7 | 24.2 +/- 3.6 |
| Retrieval-only base gate | base gate fixed to 0 | 0.375 +/- 0.104 | 0.676 +/- 0.051 | -0.091 | 72.0 +/- 3.1 | 30.0 +/- 0.0 |
| No route gates | shift/aggressive gates off | 0.591 +/- 0.049 | 0.767 +/- 0.000 | +0.000 | 73.4 +/- 1.2 | 25.0 +/- 3.8 |
