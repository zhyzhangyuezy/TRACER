# ATLASv2 Frontier Snapshot (2026-04-03)

This snapshot freezes the strongest seeded ATLASv2 results after the first large-architecture tri-router jump.

## Claim-bearing seeded anchors

- `r120_campaign_mem_dual_selector_no_auxiliary_atlasv2_public`
  - chronology: `0.6432 +- 0.0508`
  - event-disjoint: `0.7530 +- 0.0298`
  - current chronology-focused Campaign-MEM-family anchor

- `r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public`
  - chronology: `0.6020 +- 0.0525`
  - event-disjoint: `0.7725 +- 0.0083`
  - current main-text held-out-family Campaign-MEM anchor

## Best seeded appendix-level family variants so far

- `r191_campaign_mem_modular_delta_router_mid_soft_proxy_top3_atlasv2_public`
  - chronology: `0.5790 +- 0.0484`
  - event-disjoint: `0.7768 +- 0.0108`
  - first seeded delta-router variant to exceed `r115` on held-out-family AUPRC

- `r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public`
  - chronology: `0.5957 +- 0.0620`
  - event-disjoint: `0.7768 +- 0.0108`
  - strongest late-selected delta-router event result with slightly better chronology than `r191`

- `r197_campaign_mem_modular_delta_router_mid_soft_proxy_top3_late_atlasv2_public`
  - chronology: `0.5982 +- 0.0646`
  - event-disjoint: `0.7752 +- 0.0124`
  - closest new balanced point to `r115`, but still slightly below it on chronology

## Best TSLib-inspired fixed recipe so far

- `r215_campaign_mem_decomp_modular_patch_atlasv2_public`
  - chronology: `0.6223 +- 0.0567`
  - event-disjoint: `0.7623 +- 0.0061`
  - strongest new fixed cross-objective model from the TSLib-inspired search
  - structure: trend `DLinear` + residual `PatchTST` inside the modular Campaign-MEM calibration skeleton
  - integrated final alias: `campaign_mem_final`
  - integrated default config: `configs/experiments/r224_campaign_mem_final_atlasv2_public.yaml`

## Working conclusion before the next structural jump

- Late proxy-aware checkpoint selection solved part of the delta-router instability.
- The remaining gap is no longer mainly a selection bug.
- The first larger architectural jump has now been tested through the tri-expert soft-router family (`r204`--`r209`).
- The only tri-expert variant worth remembering is `r206_campaign_mem_modular_tri_router_direct_late_atlasv2_public`:
  - seed 7: `0.6648 / 0.7667`
  - seeded three-run: `0.6047 +- 0.0953 / 0.7369 +- 0.0333`
  - interpretation: the larger structure can reach the right balanced seed-7 region, but its held-out-family behavior is too unstable to replace the current anchors.
- The TSLib-inspired search adds one better conclusion:
  - naive SOTA-encoder transplant does not help
  - a decomposition-guided expert split **does** help, and `r215` is now the best new fixed recipe to carry forward
- The next meaningful move should be router regularization or staged router training, not more small recipe tuning and not a blind expansion to even more experts.
