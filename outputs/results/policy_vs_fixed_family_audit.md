# Policy vs Fixed-Family Audit

Workbook probe excluded from this audit because it is a weaker chronological-only cold-start probe with a different candidate family set and no held-out-family split.

| Method | Best/tie-best | Top-two | Mean rank |
| --- | ---: | ---: | ---: |
| TRACER | 5 | 6 | 1.25 |
| Small-Transformer-Forecaster | 1 | 3 | 2.33 |
| DLinear-Forecaster | 1 | 2 | 3.08 |
| Prefix-Only-Retrieval + Fusion | 0 | 1 | 3.33 |

| Setting | TRACER | DLinear | Small-Transformer | Prefix |
| --- | ---: | ---: | ---: | ---: |
| ATLASv2 chronology | 0.677 | 0.490 | 0.538 | 0.274 |
| ATLAS-Raw chronology | 0.461 | 0.004 | 0.350 | 0.321 |
| ATLAS-Raw event-disjoint | 0.403 | 0.002 | 0.030 | 0.259 |
| ATLASv2 held-out-family | 0.763 | 0.723 | 0.713 | 0.669 |
| AIT-ADS chronology | 0.532 | 0.512 | 0.536 | 0.530 |
| AIT-ADS held-out-scenario | 0.450 | 0.450 | 0.427 | 0.416 |
