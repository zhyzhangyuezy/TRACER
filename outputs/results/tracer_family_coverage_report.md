# TRACER Family Coverage Report

- Archived ranks below are computed against the full seeded_3 leaderboard for each dataset/split, so they include historical sweeps beyond the paper's direct policy reruns.
- This report is diagnostic about family coverage; it does not replace the paper's claim-bearing ATLASv2 held-out-family significance audit.

| Dataset | Split | TRACER mode | Bound experiment | Archived seeded_3 rank | Mean AUPRC | Evidence |
| --- | --- | --- | --- | ---: | ---: | --- |
| data/atlasv2_public | test | sparse_diverse_chrono_spiky -> TCN | r005_tcn_forecaster_atlasv2_public | 2 | 0.6771 | seeded_3 |
| data/atlasv2_public | test_event_disjoint | sparse_diverse_event -> decomposition-guided TRACER core | r215_campaign_mem_decomp_modular_patch_atlasv2_public | 21 | 0.7667 | seeded_3 |
| data/ait_ads_public | test | dense_low_diversity -> balanced TRACER | r154_campaign_mem_auto_balanced_ait_ads_public | 1 | 0.5478 | seeded_3 |
| data/ait_ads_public | test_event_disjoint | dense_low_diversity_event -> DLinear | r068_dlinear_forecaster_ait_ads_public | 3 | 0.4541 | seeded_3 |
| data/atlas_raw_public | test | extreme_sparse -> conservative TRACER | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4612 | seeded_3 |
| data/atlas_raw_public | test_event_disjoint | extreme_sparse -> conservative TRACER | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4027 | seeded_3 |
| data/synthetic_cam_lds | test | simple_dense -> linear | r229_tracer_auto_synthetic_cam_lds | 1 | 0.9767 | seeded_3 |
| data/synthetic_cam_lds | test_event_disjoint | simple_dense -> linear | r229_tracer_auto_synthetic_cam_lds | 1 | 0.9596 | seeded_3 |
| data/atlasv2_workbook | test | cold_start_sparse -> LSTM | r019_lstm_forecaster_atlasv2_workbook | 1 | 0.2320 | seeded_3 |

- `wins`: 4/9 = 0.444
- `top2`: 7/9 = 0.778
