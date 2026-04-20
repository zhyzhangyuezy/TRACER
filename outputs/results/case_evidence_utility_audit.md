# Rubric-based case evidence utility audit

This is a deterministic analyst-utility proxy, not a human user study. A retrieved case is reviewable when it shares current-prefix tactic evidence (Jaccard >= 0.25) or a high-risk channel/stage with the query. Escalation support additionally requires a positive retrieved train label. Unanchored positive evidence means a positive retrieved label without current-prefix semantic support.

| Setting | Method | Top10 reviewable | Top10 support | Top10 clean support | Top10 unanchored+ | Pos support | RelPosPrec | kappa tactic/risk |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ATLASv2 held-out-family | TRACER route | 85.2 | 44.4 | 44.4 | 7.4 | 50.0 | 86.7 | 0.92 |
| ATLASv2 held-out-family | Prefix-Only | 74.1 | 33.3 | 33.3 | 0.0 | 44.4 | 100.0 | 0.93 |
| ATLASv2 held-out-family | Pure-kNN | 44.4 | 44.4 | 44.4 | 55.6 | 33.3 | 35.0 | 0.93 |
| ATLASv2 held-out-family | Shared-Encoder | 96.3 | 81.5 | 81.5 | 3.7 | 61.1 | 65.7 | 0.93 |
| AIT-ADS chronology | TRACER route | 100.0 | 53.9 | 53.9 | 0.0 | 62.8 | 100.0 | 0.00 |
| AIT-ADS chronology | Prefix-Only | 100.0 | 60.8 | 60.8 | 0.0 | 65.1 | 100.0 | 0.00 |

## Additional all-window metrics

| Setting | Method | Reviewable | Support | Clean support | Unanchored+ |
|---|---|---:|---:|---:|---:|
| ATLASv2 held-out-family | TRACER route | 44.6 | 4.8 | 4.8 | 1.6 |
| ATLASv2 held-out-family | Prefix-Only | 44.6 | 3.6 | 3.6 | 0.0 |
| ATLASv2 held-out-family | Pure-kNN | 44.6 | 7.2 | 7.2 | 15.7 |
| ATLASv2 held-out-family | Shared-Encoder | 44.6 | 8.8 | 8.8 | 7.2 |
| AIT-ADS chronology | TRACER route | 100.0 | 10.5 | 10.5 | 0.0 |
| AIT-ADS chronology | Prefix-Only | 100.0 | 13.0 | 13.0 | 0.0 |
