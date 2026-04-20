# TSLib-Inspired Problem-Driven Model Summary (2026-04-03)

## Goal

Use ideas from `C:\Users\Administrator\OneDrive\跑程序\TSLib\Time-Series-Library-main` to make a more fundamental architecture change for `ATLASv2`, rather than only stacking more experts or sweeping small selection weights.

## What Was Tried

### 1. Naive encoder transplant

- `r210_campaign_mem_v3_patchtst_itransformer_atlasv2_public`
- `r211_campaign_mem_v3_timesnet_itransformer_atlasv2_public`

Seed-7 result:

- `r210`: `0.0600 / 0.0995`
- `r211`: `0.0528 / 0.2025`

Conclusion:

- Simply swapping in `TSLib` encoders is not enough.

### 2. From-scratch decomposition router

- `r212_campaign_mem_regime_patch_router_atlasv2_public`
- `r213_campaign_mem_regime_times_router_atlasv2_public`
- `r214_campaign_mem_regime_tsmixer_router_atlasv2_public`

Seed-7 result:

- `r212`: `0.0885 / 0.2631`
- `r213`: `0.0906 / 0.4798`
- `r214`: `0.0776 / 0.2413`

Conclusion:

- The idea is interesting, but changing both the architecture and the optimization dynamics at once was too unstable.

### 3. Decomposition-guided modular family

- `r215_campaign_mem_decomp_modular_patch_atlasv2_public`
- `r216_campaign_mem_decomp_modular_times_atlasv2_public`
- `r217_campaign_mem_decomp_modular_tsmixer_atlasv2_public`
- `r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public`

Three-seed result:

- `r215`: `0.6223 +- 0.0567 / 0.7623 +- 0.0061`
- `r216`: `0.5625 +- 0.0804 / 0.7219 +- 0.0503`
- `r218`: `0.6048 +- 0.0226 / 0.7425 +- 0.0454`

Conclusion:

- This is the first TSLib-inspired direction that really works.

## Final Model Recommendation

### Recommended model

- `r215_campaign_mem_decomp_modular_patch_atlasv2_public`

### Structural idea

- Stable trend path: `DLinear`
- Shock / burst path: residual `PatchTST`
- Retrieval path: `Transformer`
- Top-level decision layer: existing modular Campaign-MEM calibration / gating machinery

### Why this is the best model from this round

- It is not the best on every single metric.
- But it is the strongest new fixed recipe that stays competitive on both chronology and held-out-family evaluation at the same time.
- It improves chronology over the old event-focused anchor `r115` while keeping event-disjoint performance close.
- It is also much more credible than the flashy but unstable new routers.

## Practical Takeaway

- The key lesson from TSLib is not "import a SOTA model name".
- The key lesson is to separate the problem into:
  - stable regime information
  - bursty local shock information
  - retrieval calibration
- Once those roles are assigned clearly, `PatchTST` becomes useful.
- Without that structure, the same component is ineffective.

## Follow-Up Verdict

- Later stabilization attempts around this family did **not** beat `r215`.
- `iTransformer` retrieval variants (`r221`--`r223`) failed badly.
- Mild decomposition-prior routing (`r219`, `r220`) also hurt held-out-family performance.
- So the final recommendation remains unchanged:
  - keep `r215` as the best model from the TSLib-inspired search
  - do not continue the `iTransformer retrieval` or `decomp prior` branches
