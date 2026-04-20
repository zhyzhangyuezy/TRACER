# Experiment Leaderboard

- This report ranks all archived experiments found in `outputs/results`, including historical sweeps and exploratory family variants.
- These leaderboard ranks are therefore broader than the paper's claim gate; for ATLASv2 held-out-family, the manuscript relies on the frozen-policy rerun plus the 20-seed incident-block audit rather than leaderboard rank alone.

## Paper Claim-Bearing Rows

| Dataset | Split | Experiment | Archived seeded_3 rank | Mean AUPRC | Evidence | Paper role |
| --- | --- | --- | ---: | ---: | --- | --- |
| data/atlasv2_public | test | r239_tracer_adaptive_chronology_atlasv2_public | 3 | 0.6771 | seeded_3 | Primary chronology route in the paper; direct frozen-policy rerun. |
| data/atlasv2_public | test_event_disjoint | r240_tracer_adaptive_event_atlasv2_public | 7 | 0.7750 | seeded_3 | Primary held-out-family route; interpreted together with the 20-seed incident-block audit. |
| data/ait_ads_public | test | r241_tracer_adaptive_ait_ads_public | 2 | 0.5478 | seeded_3 | Supplementary chronology benchmark for the same frozen policy. |
| data/ait_ads_public | test_event_disjoint | r242_tracer_adaptive_event_ait_ads_public | 2 | 0.4541 | seeded_3 | Supplementary held-out benchmark for the same frozen policy. |
| data/atlas_raw_public | test | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4612 | seeded_3 | Supplementary raw-observation benchmark under the same policy. |
| data/atlas_raw_public | test_event_disjoint | r243_tracer_adaptive_atlas_raw_public | 2 | 0.4027 | seeded_3 | Supplementary raw held-out benchmark under the same policy. |
| data/atlasv2_workbook | test | r245_tracer_adaptive_atlasv2_workbook | 4 | 0.1331 | seeded_3 | Workbook stress probe; diagnostic only, not claim-bearing. |

## data/ait_ads_public

### test

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r154_campaign_mem_auto_balanced_ait_ads_public | 0.5478 | 0.8825 | 0.6235 | 0.4143 | 0.0118 | seeded_3 |
| 2 | r241_tracer_adaptive_ait_ads_public | 0.5478 | 0.8813 | 0.6265 | 0.3679 | 0.0134 | seeded_3 |
| 3 | r227_tracer_auto_ait_ads_public | 0.5336 | 0.8816 | 0.6313 | 0.3293 | 0.0156 | seeded_3 |
| 4 | r135_campaign_mem_abstain_auprc_select_ait_ads_public | 0.5324 | - | - | - | - | seeded_3 |
| 5 | r235_tracer_auto_chronology_ait_ads_public | 0.5319 | 0.8844 | 0.6343 | 0.3240 | 0.0132 | seeded_3 |
| 6 | r071_prefix_retrieval_ait_ads_public | 0.5316 | - | - | - | - | seeded_3 |
| 7 | r072_campaign_mem_ait_ads_public | 0.5304 | - | - | - | - | seeded_3 |
| 8 | r098_campaign_mem_abstain_ait_ads_public | 0.5286 | - | - | - | - | seeded_3 |
| 9 | r081_campaign_mem_staged_ait_ads_public | 0.5269 | - | - | - | - | seeded_3 |
| 10 | r117_campaign_mem_dual_selector_proxy_strict_ait_ads_public | 0.5268 | - | - | - | - | seeded_3 |

Seeded partial-progress leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r155_campaign_mem_auto_chronology_ait_ads_public | 0.5485 | - | - | - | - | seeded_partial_1 |

Seeded two-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r133_campaign_mem_abstain_proxy_select_ait_ads_public | 0.5319 | - | - | - | - | seeded_partial_2 |
| 2 | r134_campaign_mem_auprc_select_ait_ads_public | 0.5299 | - | - | - | - | seeded_partial_2 |
| 3 | r132_campaign_mem_proxy_select_ait_ads_public | 0.5225 | - | - | - | - | seeded_partial_2 |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r233_tracer_auto_event_ait_ads_public | 0.5362 | 0.8808 | 0.6298 | 0.3426 | 0.0117 | single_run |
| 2 | r075_campaign_mem_schedule_ait_ads_public | 0.5261 | - | - | - | - | single_run |
| 3 | r237_tracer_auto_v2_event_ait_ads_public | 0.4933 | 0.8666 | 0.6253 | 0.3586 | 0.0274 | single_run |
| 4 | r067_tail_risk_linear_ait_ads_public | 0.4620 | - | - | - | - | single_run |

### test_event_disjoint

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r069_tcn_forecaster_ait_ads_public | 0.4549 | - | - | - | - | seeded_3 |
| 2 | r242_tracer_adaptive_event_ait_ads_public | 0.4541 | 0.8654 | 0.5550 | 0.0945 | 0.0215 | seeded_3 |
| 3 | r068_dlinear_forecaster_ait_ads_public | 0.4541 | - | - | - | - | seeded_3 |
| 4 | r154_campaign_mem_auto_balanced_ait_ads_public | 0.4529 | 0.8634 | 0.5464 | 0.0904 | 0.0239 | seeded_3 |
| 5 | r241_tracer_adaptive_ait_ads_public | 0.4529 | 0.8683 | 0.5502 | 0.0879 | 0.0226 | seeded_3 |
| 6 | r227_tracer_auto_ait_ads_public | 0.4518 | 0.8687 | 0.5478 | 0.1111 | 0.0217 | seeded_3 |
| 7 | r081_campaign_mem_staged_ait_ads_public | 0.4466 | - | - | - | - | seeded_3 |
| 8 | r235_tracer_auto_chronology_ait_ads_public | 0.4451 | 0.8664 | 0.5464 | 0.0896 | 0.0215 | seeded_3 |
| 9 | r098_campaign_mem_abstain_ait_ads_public | 0.4415 | - | - | - | - | seeded_3 |
| 10 | r135_campaign_mem_abstain_auprc_select_ait_ads_public | 0.4399 | - | - | - | - | seeded_3 |

Seeded partial-progress leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r155_campaign_mem_auto_chronology_ait_ads_public | 0.4332 | - | - | - | - | seeded_partial_1 |

Seeded two-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r134_campaign_mem_auprc_select_ait_ads_public | 0.4370 | - | - | - | - | seeded_partial_2 |
| 2 | r132_campaign_mem_proxy_select_ait_ads_public | 0.4352 | - | - | - | - | seeded_partial_2 |
| 3 | r133_campaign_mem_abstain_proxy_select_ait_ads_public | 0.4216 | - | - | - | - | seeded_partial_2 |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r233_tracer_auto_event_ait_ads_public | 0.4501 | 0.8739 | 0.5523 | 0.0796 | 0.0264 | single_run |
| 2 | r067_tail_risk_linear_ait_ads_public | 0.4381 | - | - | - | - | single_run |
| 3 | r237_tracer_auto_v2_event_ait_ads_public | 0.4251 | 0.8350 | 0.4857 | 0.1625 | 0.0083 | single_run |
| 4 | r075_campaign_mem_schedule_ait_ads_public | 0.4155 | - | - | - | - | single_run |

## data/atlas_raw_public

### test

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r088_campaign_mem_conservative_atlas_raw_public | 0.4612 | - | - | - | - | seeded_3 |
| 2 | r243_tracer_adaptive_atlas_raw_public | 0.4612 | 0.9669 | 0.4956 | 0.1667 | 0.0058 | seeded_3 |
| 3 | r228_tracer_auto_atlas_raw_public | 0.4608 | 0.9669 | 0.4956 | 0.1389 | 0.0058 | seeded_3 |
| 4 | r047_campaign_mem_v3_dlinear_atlas_raw_public | 0.3619 | - | - | - | - | seeded_3 |
| 5 | r032_transformer_forecaster_atlas_raw_public | 0.3500 | - | - | - | - | seeded_3 |
| 6 | r036_campaign_mem_atlas_raw_public | 0.3454 | - | - | - | - | seeded_3 |
| 7 | r121_campaign_mem_dual_selector_proxy_strict_atlas_raw_public | 0.3411 | - | - | - | - | seeded_3 |
| 8 | r035_prefix_retrieval_atlas_raw_public | 0.3206 | - | - | - | - | seeded_3 |
| 9 | r077_campaign_mem_schedule_lite_atlas_raw_public | 0.3143 | - | - | - | - | seeded_3 |
| 10 | r089_campaign_mem_selector_conservative_atlas_raw_public | 0.2748 | - | - | - | - | seeded_3 |

Seeded partial-progress leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r156_campaign_mem_auto_balanced_atlas_raw_public | 0.4526 | - | - | - | - | seeded_partial_1 |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r074_campaign_mem_schedule_atlas_raw_public | 0.2858 | - | - | - | - | single_run |
| 2 | r092_campaign_mem_selector_transformer_atlas_raw_public | 0.2472 | - | - | - | - | single_run |
| 3 | r095_campaign_mem_selector_teacher_transformer_atlas_raw_public | 0.2295 | - | - | - | - | single_run |
| 4 | r080_campaign_mem_staged_atlas_raw_public | 0.1988 | - | - | - | - | single_run |
| 5 | r083_campaign_mem_selector_staged_atlas_raw_public | 0.1709 | - | - | - | - | single_run |
| 6 | r100_campaign_mem_abstain_detached_atlas_raw_public | 0.1454 | - | - | - | - | single_run |
| 7 | r086_campaign_mem_selector_proxy_atlas_raw_public | 0.1185 | - | - | - | - | single_run |
| 8 | r101_campaign_mem_shift_selector_atlas_raw_public | 0.1040 | - | - | - | - | single_run |
| 9 | r097_campaign_mem_abstain_atlas_raw_public | 0.0875 | - | - | - | - | single_run |
| 10 | r093_campaign_mem_selector_conservative_topk3_atlas_raw_public | 0.0734 | - | - | - | - | single_run |

### test_event_disjoint

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r088_campaign_mem_conservative_atlas_raw_public | 0.4027 | - | - | - | - | seeded_3 |
| 2 | r243_tracer_adaptive_atlas_raw_public | 0.4027 | 0.6690 | 0.5897 | 0.1111 | 0.0015 | seeded_3 |
| 3 | r228_tracer_auto_atlas_raw_public | 0.3662 | 0.6256 | 0.5000 | 0.1111 | 0.0015 | seeded_3 |
| 4 | r089_campaign_mem_selector_conservative_atlas_raw_public | 0.2924 | - | - | - | - | seeded_3 |
| 5 | r035_prefix_retrieval_atlas_raw_public | 0.2587 | - | - | - | - | seeded_3 |
| 6 | r036_campaign_mem_atlas_raw_public | 0.2141 | - | - | - | - | seeded_3 |
| 7 | r077_campaign_mem_schedule_lite_atlas_raw_public | 0.1133 | - | - | - | - | seeded_3 |
| 8 | r047_campaign_mem_v3_dlinear_atlas_raw_public | 0.1019 | - | - | - | - | seeded_3 |
| 9 | r032_transformer_forecaster_atlas_raw_public | 0.0299 | - | - | - | - | seeded_3 |
| 10 | r026_tail_risk_linear_atlas_raw_public | 0.0025 | - | - | - | - | seeded_3 |

Seeded partial-progress leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r156_campaign_mem_auto_balanced_atlas_raw_public | 0.3342 | - | - | - | - | seeded_partial_1 |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r095_campaign_mem_selector_teacher_transformer_atlas_raw_public | 0.0469 | - | - | - | - | single_run |
| 2 | r080_campaign_mem_staged_atlas_raw_public | 0.0356 | - | - | - | - | single_run |
| 3 | r093_campaign_mem_selector_conservative_topk3_atlas_raw_public | 0.0045 | - | - | - | - | single_run |
| 4 | r097_campaign_mem_abstain_atlas_raw_public | 0.0025 | - | - | - | - | single_run |
| 5 | r083_campaign_mem_selector_staged_atlas_raw_public | 0.0025 | - | - | - | - | single_run |
| 6 | r092_campaign_mem_selector_transformer_atlas_raw_public | 0.0025 | - | - | - | - | single_run |
| 7 | r086_campaign_mem_selector_proxy_atlas_raw_public | 0.0025 | - | - | - | - | single_run |
| 8 | r091_campaign_mem_selector_conservative_mid2_atlas_raw_public | 0.0021 | - | - | - | - | single_run |
| 9 | r100_campaign_mem_abstain_detached_atlas_raw_public | 0.0014 | - | - | - | - | single_run |
| 10 | r090_campaign_mem_selector_conservative_mid_atlas_raw_public | 0.0013 | - | - | - | - | single_run |

## data/atlasv2_public

### test

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r250_campaign_mem_decomp_modular_patch_proxybank_possqrt_atlasv2_public | 0.6905 | 0.9816 | 0.7368 | 0.2381 | 0.0430 | seeded_3 |
| 2 | r005_tcn_forecaster_atlasv2_public | 0.6771 | - | - | - | - | seeded_3 |
| 3 | r239_tracer_adaptive_chronology_atlasv2_public | 0.6771 | 0.9796 | 0.7123 | 0.2857 | 0.0304 | seeded_3 |
| 4 | r143_campaign_mem_dual_selector_no_aux_proxy_strict_noaf_atlasv2_public | 0.6541 | 0.9790 | 0.7368 | 0.1429 | 0.0216 | seeded_3 |
| 5 | r120_campaign_mem_dual_selector_no_auxiliary_atlasv2_public | 0.6432 | - | - | - | - | seeded_3 |
| 6 | r246_campaign_mem_decomp_modular_patch_stableavg_atlasv2_public | 0.6353 | 0.9755 | 0.6788 | 0.2381 | 0.0298 | seeded_3 |
| 7 | r247_campaign_mem_decomp_modular_patch_stableavg_earlysel_atlasv2_public | 0.6353 | 0.9755 | 0.6788 | 0.2381 | 0.0298 | seeded_3 |
| 8 | r166_campaign_mem_modular_shift_multi_proxy_atlasv2_public | 0.6333 | - | - | - | - | seeded_3 |
| 9 | r236_tracer_auto_v2_chronology_atlasv2_public | 0.6305 | 0.9790 | 0.7123 | 0.1905 | 0.0303 | seeded_3 |
| 10 | r248_campaign_mem_decomp_modular_patch_conservative_event_atlasv2_public | 0.6230 | 0.9793 | 0.7246 | 0.1429 | 0.0297 | seeded_3 |

Seeded partial-progress leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r220_campaign_mem_decomp_modular_patch_noaux_prior20_atlasv2_public | 0.6648 | - | - | - | - | seeded_partial_1 |
| 2 | r153_campaign_mem_auto_chronology_atlasv2_public | 0.6529 | - | - | - | - | seeded_partial_1 |
| 3 | r164_campaign_mem_modular_shift_open_atlasv2_public | 0.6429 | - | - | - | - | seeded_partial_1 |
| 4 | r165_campaign_mem_modular_shift_no_aux_atlasv2_public | 0.6365 | - | - | - | - | seeded_partial_1 |
| 5 | r158_campaign_mem_modular_no_aux_atlasv2_public | 0.6338 | - | - | - | - | seeded_partial_1 |
| 6 | r207_campaign_mem_modular_tri_router_direct_late_noabstain_atlasv2_public | 0.6172 | - | - | - | - | seeded_partial_1 |
| 7 | r159_campaign_mem_modular_open_atlasv2_public | 0.6093 | - | - | - | - | seeded_partial_1 |
| 8 | r162_campaign_mem_modular_open_top3_atlasv2_public | 0.6077 | - | - | - | - | seeded_partial_1 |
| 9 | r179_campaign_mem_modular_shift_aggressive_supervised_mid_atlasv2_public | 0.6070 | - | - | - | - | seeded_partial_1 |
| 10 | r182_campaign_mem_modular_shift_aggressive_supervised_open_proxy_atlasv2_public | 0.5952 | - | - | - | - | seeded_partial_1 |

Seeded two-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r138_campaign_mem_dual_selector_no_aux_proxy_mid_noaf_atlasv2_public | 0.6717 | - | - | - | - | seeded_partial_2 |
| 2 | r139_campaign_mem_dual_selector_no_hard_neg_proxy_mid_noaf_atlasv2_public | 0.6372 | - | - | - | - | seeded_partial_2 |
| 3 | r136_campaign_mem_dual_selector_proxy_mid_noaf_atlasv2_public | 0.6223 | - | - | - | - | seeded_partial_2 |
| 4 | r141_campaign_mem_dual_selector_proxy_strict_avg3_atlasv2_public | 0.6223 | - | - | - | - | seeded_partial_2 |
| 5 | r131_campaign_mem_v2_tcn_proxy_select_no_aux_atlasv2_public | 0.6205 | - | - | - | - | seeded_partial_2 |
| 6 | r130_campaign_mem_v2_tcn_proxy_select_atlasv2_public | 0.6134 | - | - | - | - | seeded_partial_2 |
| 7 | r142_campaign_mem_dual_selector_no_aux_proxy_strict_avg3_atlasv2_public | 0.6015 | - | - | - | - | seeded_partial_2 |
| 8 | r137_campaign_mem_dual_selector_proxy_light_noaf_atlasv2_public | 0.5997 | - | - | - | - | seeded_partial_2 |
| 9 | r140_campaign_mem_dual_selector_proxy_strict_noaf_atlasv2_public | 0.5938 | - | - | - | - | seeded_partial_2 |
| 10 | r125_campaign_mem_v2_tcn_stabilized_no_aux_atlasv2_public | 0.5815 | - | - | - | - | seeded_partial_2 |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public_seed610 | 0.7048 | 0.9825 | 0.7368 | 0.1429 | 0.0288 | single_run |
| 2 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public_seed89 | 0.6571 | 0.9816 | 0.7368 | 0.1429 | 0.0308 | single_run |
| 3 | r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public_seed233 | 0.6429 | 0.9807 | 0.7368 | 0.1429 | 0.0310 | single_run |
| 4 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public_seed377 | 0.6327 | 0.9798 | 0.7368 | 0.1429 | 0.0249 | single_run |
| 5 | r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public_seed34 | 0.6315 | 0.9781 | 0.7000 | 0.1429 | 0.0285 | single_run |
| 6 | r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public_seed377 | 0.6190 | 0.9798 | 0.7368 | 0.1429 | 0.0327 | single_run |
| 7 | r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public_seed89 | 0.6190 | 0.9798 | 0.7368 | 0.1429 | 0.0371 | single_run |
| 8 | r215_campaign_mem_decomp_modular_patch_atlasv2_public_seed233 | 0.6190 | 0.9798 | 0.7368 | 0.1429 | 0.0287 | single_run |
| 9 | r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public_seed377 | 0.6190 | 0.9798 | 0.7368 | 0.1429 | 0.0276 | single_run |
| 10 | r215_campaign_mem_decomp_modular_patch_atlasv2_public_seed610 | 0.6172 | 0.9772 | 0.7000 | 0.1429 | 0.0260 | single_run |

### test_event_disjoint

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r232_tracer_auto_event_atlasv2_public | 0.7790 | 0.9646 | 0.8000 | 0.6667 | 0.0499 | seeded_3 |
| 2 | r191_campaign_mem_modular_delta_router_mid_soft_proxy_top3_atlasv2_public | 0.7768 | - | - | - | - | seeded_3 |
| 3 | r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public | 0.7768 | - | - | - | - | seeded_3 |
| 4 | r197_campaign_mem_modular_delta_router_mid_soft_proxy_top3_late_atlasv2_public | 0.7752 | - | - | - | - | seeded_3 |
| 5 | r192_campaign_mem_modular_delta_router_mid_soft_proxy_top1_atlasv2_public | 0.7750 | - | - | - | - | seeded_3 |
| 6 | r196_campaign_mem_modular_delta_router_mid_soft_later_top3_atlasv2_public | 0.7750 | - | - | - | - | seeded_3 |
| 7 | r240_tracer_adaptive_event_atlasv2_public | 0.7750 | 0.9632 | 0.8000 | 0.6111 | 0.0402 | seeded_3 |
| 8 | r195_campaign_mem_modular_delta_router_mid_soft_late_top3_atlasv2_public | 0.7734 | - | - | - | - | seeded_3 |
| 9 | r200_campaign_mem_modular_delta_router_mid_soft_proxy_top3_late_open20_atlasv2_public | 0.7734 | - | - | - | - | seeded_3 |
| 10 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public | 0.7725 | - | - | - | - | seeded_3 |

Seeded partial-progress leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r157_campaign_mem_modular_mid_atlasv2_public | 0.7917 | - | - | - | - | seeded_partial_1 |
| 2 | r062_campaign_mem_structured_atlasv2_public | 0.7719 | - | - | - | - | seeded_partial_1 |
| 3 | r063_campaign_mem_structured_loose_atlasv2_public | 0.7719 | - | - | - | - | seeded_partial_1 |
| 4 | r164_campaign_mem_modular_shift_open_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_1 |
| 5 | r171_campaign_mem_modular_shift_aggressive_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_1 |
| 6 | r172_campaign_mem_modular_shift_aggressive_open_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_1 |
| 7 | r173_campaign_mem_modular_shift_aggressive_tight_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_1 |
| 8 | r174_campaign_mem_modular_shift_aggressive_no_aux_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_1 |
| 9 | r177_campaign_mem_modular_shift_aggressive_supervised_tight_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_1 |
| 10 | r178_campaign_mem_modular_shift_aggressive_supervised_no_aux_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_1 |

Seeded two-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r149_campaign_mem_dual_selector_no_aux_multi_proxy_atlasv2_public | 0.7768 | - | - | - | - | seeded_partial_2 |
| 2 | r144_campaign_mem_dual_selector_proxy_strict_possqrt_atlasv2_public | 0.7667 | - | - | - | - | seeded_partial_2 |
| 3 | r142_campaign_mem_dual_selector_no_aux_proxy_strict_avg3_atlasv2_public | 0.7411 | - | - | - | - | seeded_partial_2 |
| 4 | r140_campaign_mem_dual_selector_proxy_strict_noaf_atlasv2_public | 0.7411 | - | - | - | - | seeded_partial_2 |
| 5 | r151_campaign_mem_dual_selector_no_aux_mid_multi_proxy_atlasv2_public | 0.7316 | - | - | - | - | seeded_partial_2 |
| 6 | r137_campaign_mem_dual_selector_proxy_light_noaf_atlasv2_public | 0.7310 | - | - | - | - | seeded_partial_2 |
| 7 | r150_campaign_mem_dual_selector_proxy_mid_multi_proxy_atlasv2_public | 0.7310 | - | - | - | - | seeded_partial_2 |
| 8 | r136_campaign_mem_dual_selector_proxy_mid_noaf_atlasv2_public | 0.7286 | - | - | - | - | seeded_partial_2 |
| 9 | r141_campaign_mem_dual_selector_proxy_strict_avg3_atlasv2_public | 0.7286 | - | - | - | - | seeded_partial_2 |
| 10 | r146_campaign_mem_dual_selector_no_aux_possqrt_atlasv2_public | 0.7286 | - | - | - | - | seeded_partial_2 |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public_seed34 | 0.7917 | 0.9719 | 0.8000 | 0.6667 | 0.0492 | single_run |
| 2 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public_seed55 | 0.7917 | 0.9719 | 0.8000 | 0.6667 | 0.0563 | single_run |
| 3 | r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public_seed233 | 0.7917 | 0.9719 | 0.8000 | 0.5000 | 0.0408 | single_run |
| 4 | r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public_seed377 | 0.7917 | 0.9719 | 0.8000 | 0.6667 | 0.0469 | single_run |
| 5 | r215_campaign_mem_decomp_modular_patch_atlasv2_public_seed55 | 0.7917 | 0.9719 | 0.8000 | 0.5000 | 0.0514 | single_run |
| 6 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public_seed144 | 0.7843 | 0.9675 | 0.8000 | 0.6667 | 0.0506 | single_run |
| 7 | r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public_seed233 | 0.7843 | 0.9675 | 0.8000 | 0.6667 | 0.0470 | single_run |
| 8 | r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public_seed55 | 0.7843 | 0.9675 | 0.8000 | 0.6667 | 0.0545 | single_run |
| 9 | r215_campaign_mem_decomp_modular_patch_atlasv2_public_seed233 | 0.7843 | 0.9675 | 0.8000 | 0.6667 | 0.0495 | single_run |
| 10 | r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public_seed34 | 0.7778 | 0.9632 | 0.8000 | 0.6667 | 0.0451 | single_run |

## data/atlasv2_workbook

### test

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r019_lstm_forecaster_atlasv2_workbook | 0.2320 | - | - | - | - | seeded_3 |
| 2 | r013_prefix_retrieval_atlasv2_workbook | 0.1620 | - | - | - | - | seeded_3 |
| 3 | r013_transformer_atlasv2_workbook | 0.1495 | - | - | - | - | seeded_3 |
| 4 | r245_tracer_adaptive_atlasv2_workbook | 0.1331 | 0.6472 | 0.2184 | 0.0000 | 0.0207 | seeded_3 |
| 5 | r230_tracer_auto_atlasv2_workbook | 0.1036 | 0.6746 | 0.1527 | 0.0513 | 0.0222 | seeded_3 |
| 6 | r013_campaign_mem_atlasv2_workbook | 0.0704 | - | - | - | - | seeded_3 |
| 7 | r019_tcn_forecaster_atlasv2_workbook | 0.0324 | - | - | - | - | seeded_3 |
| 8 | r019_tail_risk_linear_atlasv2_workbook | 0.0233 | - | - | - | - | seeded_3 |

Seeded partial-progress leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r129_lstm_forecaster_proxy_weighted_atlasv2_workbook | 0.1197 | - | - | - | - | seeded_partial_1 |
| 2 | r128_lstm_forecaster_proxy_atlasv2_workbook | 0.1100 | - | - | - | - | seeded_partial_1 |
| 3 | r127_campaign_mem_v3_dlinear_proxy_atlasv2_workbook | 0.1026 | - | - | - | - | seeded_partial_1 |
| 4 | r126_campaign_mem_v3_lstm_proxy_atlasv2_workbook | 0.0997 | - | - | - | - | seeded_partial_1 |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r238_tracer_auto_v2_balanced_atlasv2_workbook | 0.1142 | 0.6865 | 0.1429 | 0.0769 | 0.0218 | single_run |

### test_event_disjoint

Seeded three-run leaderboard:

| - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - |

Seeded partial-progress leaderboard:

| - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - |

Exploratory single-run leaderboard:

| - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - |

## data/synthetic_cam_lds

### test

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r229_tracer_auto_synthetic_cam_lds | 0.9767 | 0.9466 | 0.9214 | 0.9695 | 0.2147 | seeded_3 |
| 2 | r244_tracer_adaptive_synthetic_cam_lds | 0.9746 | 0.9419 | 0.9207 | 0.9695 | 0.1921 | seeded_3 |

Seeded partial-progress leaderboard:

| - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r004_tail_risk_linear_smoke | 0.9750 | - | - | - | - | single_run |
| 2 | r009_campaign_mem_smoke | 0.9697 | - | - | - | - | single_run |
| 3 | r005_tcn_forecaster_smoke | 0.9298 | - | - | - | - | single_run |
| 4 | r006_transformer_forecaster_smoke | 0.9249 | - | - | - | - | single_run |
| 5 | r008_prefix_retrieval_smoke | 0.9202 | - | - | - | - | single_run |
| 6 | r007_pure_knn_smoke | 0.8417 | - | - | - | - | single_run |

### test_event_disjoint

Seeded three-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r229_tracer_auto_synthetic_cam_lds | 0.9596 | 0.9200 | 0.8807 | 0.9098 | 0.2200 | seeded_3 |
| 2 | r244_tracer_adaptive_synthetic_cam_lds | 0.9564 | 0.9113 | 0.8705 | 0.9071 | 0.2364 | seeded_3 |

Seeded partial-progress leaderboard:

| - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - |

Exploratory single-run leaderboard:

| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | r004_tail_risk_linear_smoke | 0.9546 | - | - | - | - | single_run |
| 2 | r005_tcn_forecaster_smoke | 0.9379 | - | - | - | - | single_run |
| 3 | r008_prefix_retrieval_smoke | 0.9306 | - | - | - | - | single_run |
| 4 | r006_transformer_forecaster_smoke | 0.9228 | - | - | - | - | single_run |
| 5 | r007_pure_knn_smoke | 0.9109 | - | - | - | - | single_run |
| 6 | r009_campaign_mem_smoke | 0.8531 | - | - | - | - | single_run |
