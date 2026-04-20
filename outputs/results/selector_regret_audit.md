# Selector regret audit

The oracle single-family baseline is an ex-post upper bound over choosing one fixed DLinear, small-transformer, or prefix-retrieval family in the same released audit.

| Benchmark | Evidence | Adaptive AUPRC | Oracle single-family | Oracle AUPRC | Gap |
|---|---|---:|---|---:|---:|
| ATLASv2 chronology | seeded_3 | 0.677 | Small-Transformer-Forecaster | 0.538 | +0.139 |
| ATLAS-Raw chronology | seeded_3 | 0.461 | Small-Transformer-Forecaster | 0.350 | +0.111 |
| ATLAS-Raw event-disjoint | seeded_3 | 0.403 | Prefix-Only-Retrieval + Fusion | 0.259 | +0.144 |
| ATLASv2 held-out-family | seeded_20 | 0.763 | DLinear-Forecaster | 0.723 | +0.040 |
| AIT-ADS chronology | seeded_20 | 0.532 | Small-Transformer-Forecaster | 0.536 | -0.004 |
| AIT-ADS held-out-scenario | seeded_20 | 0.450 | DLinear-Forecaster | 0.450 | +0.000 |

Mean gap to oracle single-family: +0.072.
Nonnegative gaps: 5 / 6.
Worst gap: -0.004.
