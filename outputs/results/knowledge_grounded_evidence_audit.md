# Knowledge-grounded retrieved-evidence audit

The audit uses current-prefix ATT&CK-like feature channels and, for AIT-ADS, incident-level stage profiles. It does not use future-signature distances, so it complements AF@5 and TTE-Err@1 as a knowledge-consistency check for case-based evidence.

## ATLASv2 held-out-family

| Method | K-Hit@5 | K-Jacc@5 | HighRisk-Hit@5 | Pos K-Jacc@5 | Top10 K-Jacc@5 | Stage-Hit@5 | HighStage-Hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRACER route | 44.6 | 33.4 | 40.6 | 58.8 | 71.0 | -- | -- |
| Prefix-Only | 44.6 | 33.9 | 41.0 | 57.3 | 61.6 | -- | -- |
| Pure-kNN | 44.6 | 38.4 | 41.0 | 64.4 | 34.8 | -- | -- |
| Shared-Encoder | 44.6 | 35.0 | 41.0 | 59.8 | 81.2 | -- | -- |

## AIT-ADS chronology

| Method | K-Hit@5 | K-Jacc@5 | HighRisk-Hit@5 | Pos K-Jacc@5 | Top10 K-Jacc@5 | Stage-Hit@5 | HighStage-Hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRACER route | 54.9 | 48.3 | 45.8 | 61.1 | 45.9 | 100.0 | 100.0 |
| Prefix-Only | 54.6 | 47.3 | 45.6 | 59.2 | 49.3 | 100.0 | 100.0 |
