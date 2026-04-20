# Campaign-MEM Final Scheme (2026-04-03)

## Final Model

The final integrated model is:

- `campaign_mem_final`

Its default concrete recipe is:

- retrieval encoder: `Transformer`
- stable expert: `DLinear`
- shock expert: `PatchTST`
- fusion skeleton: decomposition-guided modular Campaign-MEM calibration

This is the codified form of the best TSLib-inspired model found in this project:

- seeded reference run: `r215_campaign_mem_decomp_modular_patch_atlasv2_public`
- result: `0.6223 +- 0.0567 / 0.7623 +- 0.0061`

## Why This Is The Final Model

This model keeps only the changes that survived the search:

- Keep `DLinear` for the trend path:
  - it is the most stable way to model the slow regime and calibration baseline.
- Keep `PatchTST` for the residual path:
  - it is the most useful `TSLib` component once the problem is reframed as local shock modeling rather than whole-sequence replacement.
- Keep retrieval as `Transformer`:
  - `iTransformer` retrieval looked attractive in theory but failed badly in this family.
- Keep the modular Campaign-MEM calibration stack:
  - repeated experiments showed that new experts are only useful when inserted into an already stable retrieval-calibration skeleton.

## Final Operating Scheme

### 1. Default unified model

- Config: [r224_campaign_mem_final_atlasv2_public.yaml](C:/Users/Administrator/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/configs/experiments/r224_campaign_mem_final_atlasv2_public.yaml)
- Use this as the primary fixed model when one consistent architecture is needed.

### 2. Chronology diagnostic mode

- Config: [r225_campaign_mem_final_noaux_diag_atlasv2_public.yaml](C:/Users/Administrator/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/configs/experiments/r225_campaign_mem_final_noaux_diag_atlasv2_public.yaml)
- This is not the main final model.
- It is only a no-aux diagnostic operating mode kept because earlier experiments showed auxiliary removal can shift the chronology / held-out-family balance.

### 3. Objective-specific anchors to keep in reporting

- Chronology-focused anchor:
  - [r120_campaign_mem_dual_selector_no_auxiliary_atlasv2_public.yaml](C:/Users/Administrator/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/configs/experiments/r120_campaign_mem_dual_selector_no_auxiliary_atlasv2_public.yaml)
- Held-out-family focused anchor:
  - [r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public.yaml](C:/Users/Administrator/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/configs/experiments/r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public.yaml)
- Old robust fixed baseline for comparison:
  - [r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public.yaml](C:/Users/Administrator/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/configs/experiments/r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public.yaml)

## Final Training Scheme

- 20 epochs
- proxy-aware balanced model selection
- `model_selection_start_epoch = 8`
- `checkpoint_average_top_k = 3`
- auxiliary horizon on by default
- contrastive retrieval training on
- hard negatives on
- modular calibration with abstention, shift gate, and aggressive delta routing

## What Was Explicitly Rejected

- Naive `TSLib` encoder transplant into old families:
  - `r210`, `r211`
- From-scratch decomposition router:
  - `r212`--`r214`
- Tri-expert soft router as final mainline:
  - `r206`
- `iTransformer` retrieval inside the decomposition-patch family:
  - `r221`--`r223`
- Mild decomposition-prior routing:
  - `r219`, `r220`

## Final Recommendation For The Paper

- Present `campaign_mem_final` as the final integrated architecture.
- Use the `r224` recipe as the main fixed unified model.
- Keep `r120` and `r201` as objective-specific supporting anchors rather than pretending one fixed model dominates every split.
- Frame the main architectural insight as:
  - retrieval calibration works best when the forecasting side is decomposed into a stable trend path and a local shock path.
