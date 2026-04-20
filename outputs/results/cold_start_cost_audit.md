# Cold-start and cost audit

The table is computed from released seed result JSONs and train/dev policy statistics; bank footprint is the dense float32 embedding-bank estimate for retrieval-active routes.

| Setting | Train +/N | Dev + | Families +/N | Regime | Route | Retrieval | AUPRC | Bank MiB |
|---|---:|---:|---:|---|---|---|---:|---:|
| ATLASv2 chronology | 12/438 | 4 | 2/23 | sparse_diverse_chrono_spiky | tcn | no | 0.677 +/- 0.081 | 0.00 |
| ATLASv2 held-out-family | 12/438 | 4 | 2/23 | sparse_diverse | campaign_mem_decomp_modular | yes | 0.763 +/- 0.041 | 0.21 |
| AIT-ADS chronology | 813/9119 | 230 | 4/4 | dense_low_diversity | campaign_mem_v3 | yes | 0.532 +/- 0.021 | 4.45 |
| AIT-ADS held-out-scenario | 813/9119 | 230 | 4/4 | dense_low_diversity_event | dlinear | no | 0.450 +/- 0.013 | 0.00 |
| ATLAS-Raw chronology | 89/46156 | 18 | 7/8 | extreme_sparse | campaign_mem_v3 | yes | 0.461 +/- 0.078 | 22.54 |
| ATLAS-Raw event-disjoint | 89/46156 | 18 | 7/8 | extreme_sparse | campaign_mem_v3 | yes | 0.403 +/- 0.097 | 22.54 |
| Synthetic CAM-LDS | 429/640 | 116 | 18/18 | simple_dense | tail_risk_linear | no | 0.975 +/- 0.001 | 0.00 |
| Workbook probe | 10/1593 | 0 | 4/6 | cold_start_sparse | lstm | no | 0.133 +/- 0.141 | 0.00 |
