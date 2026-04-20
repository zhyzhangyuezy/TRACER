# TRACER Adaptive Policy Coverage Report

- Archived ranks below are computed against the full seeded_3 leaderboard for each dataset/split, so they include historical family sweeps and exploratory variants in addition to the paper's direct reruns.
- On ATLASv2 held-out-family, paper-level judgment is governed by the 20-seed incident-block audit rather than by archived rank alone.

## Published Route Bindings

| Dataset | Split | Published route | Bound experiment | Archived seeded_3 rank | Mean AUPRC | Evidence |
| --- | --- | --- | --- | ---: | ---: | --- |
| data/atlasv2_public | test | sparse_diverse_chrono_spiky -> TCN | r005_tcn_forecaster_atlasv2_public | 2 | 0.6771 | seeded_3 |
| data/atlasv2_public | test_event_disjoint | sparse_diverse -> decomposition-guided TRACER core | r215_campaign_mem_decomp_modular_patch_atlasv2_public | 21 | 0.7667 | seeded_3 |
| data/ait_ads_public | test | dense_low_diversity -> balanced TRACER | r154_campaign_mem_auto_balanced_ait_ads_public | 1 | 0.5478 | seeded_3 |
| data/ait_ads_public | test_event_disjoint | dense_low_diversity_event -> DLinear | r242_tracer_adaptive_event_ait_ads_public | 2 | 0.4541 | seeded_3 |
| data/atlas_raw_public | test | extreme_sparse -> conservative TRACER | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4612 | seeded_3 |
| data/atlas_raw_public | test_event_disjoint | extreme_sparse -> conservative TRACER | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4027 | seeded_3 |
| data/synthetic_cam_lds | test | simple_dense -> linear | r229_tracer_auto_synthetic_cam_lds | 1 | 0.9767 | seeded_3 |
| data/synthetic_cam_lds | test_event_disjoint | simple_dense -> linear | r229_tracer_auto_synthetic_cam_lds | 1 | 0.9596 | seeded_3 |
| data/atlasv2_workbook | test | cold_start_sparse -> LSTM | r019_lstm_forecaster_atlasv2_workbook | 1 | 0.2320 | seeded_3 |

- `strict wins`: 4/9 = 0.444
- `top2`: 8/9 = 0.889

## Direct tracer_adaptive Policy Reruns

| Dataset | Split | Requested objective | Experiment | Archived seeded_3 rank | Mean AUPRC | Gap to best | Tie-best |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| data/atlasv2_public | test | chronology | r239_tracer_adaptive_chronology_atlasv2_public | 3 | 0.6771 | -0.0134 | no |
| data/atlasv2_public | test_event_disjoint | event-disjoint | r240_tracer_adaptive_event_atlasv2_public | 7 | 0.7750 | -0.0040 | no |
| data/ait_ads_public | test | balanced | r241_tracer_adaptive_ait_ads_public | 2 | 0.5478 | 0.0000 | yes |
| data/ait_ads_public | test_event_disjoint | event-disjoint | r242_tracer_adaptive_event_ait_ads_public | 2 | 0.4541 | -0.0008 | no |
| data/atlas_raw_public | test | balanced | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4612 | 0.0000 | yes |
| data/atlas_raw_public | test_event_disjoint | balanced | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4027 | 0.0000 | yes |
| data/synthetic_cam_lds | test | balanced | r244_tracer_adaptive_synthetic_cam_lds | 2 | 0.9746 | -0.0021 | no |
| data/synthetic_cam_lds | test_event_disjoint | balanced | r244_tracer_adaptive_synthetic_cam_lds | 2 | 0.9564 | -0.0032 | no |
| data/atlasv2_workbook | test | balanced | r245_tracer_adaptive_atlasv2_workbook | 4 | 0.1331 | -0.0989 | no |

- `strict wins`: 0/9 = 0.000
- `tie-best`: 3/9 = 0.333
- `top2`: 6/9 = 0.667

- The first table records the exact route-table bindings implied by the published predicates; it is a diagnostic of how the frozen policy maps benchmark traits to family members.
- The direct `tracer_adaptive` reruns are the claim-bearing evidence used in the paper; the main residual gaps now come from archived family-sweep headroom on ATLASv2 held-out-family, workbook stability, and a small synthetic reproduction gap.
