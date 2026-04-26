# Blinded Security-Rater Evidence-Rating Study

This package supports a blinded security-rater evidence-rating study for TRACER. It is designed to measure human-rated evidence quality, not live SOC performance.

## Study Design

The study uses 80 paired query cases sampled from ATLASv2 held-out-family and AIT-ADS chronology. Each case shows one current alert-prefix summary and two anonymized evidence sets, `set_a` and `set_b`. One set is produced by TRACER route retrieval and the other by Prefix-Only retrieval. The order is randomized per case. Raters do not see model identities, exact warning scores, true query futures, incident identifiers, or family identifiers.

Each evidence set contains top-5 train-memory analogs. Each analog is represented by current-prefix channels, high-risk channels, stage profile, historical escalation outcome, and historical time-to-escalation when applicable. Query future labels are hidden.

## Files

- `pairwise_items_blinded.csv`: master blinded A/B packet.
- `rater_01_sheet.csv`, `rater_02_sheet.csv`, `rater_03_sheet.csv`: randomized response sheets for three raters.
- `rater_01_completed.csv`, `rater_02_completed.csv`, `rater_03_completed.csv`: completed de-identified response sheets.
- `pairwise_key_private.csv`: post-study analysis key that maps A/B sets to methods and contains true query outcomes. The word `private` means hidden from raters during annotation; the released key is used only after all ratings are complete.
- `study_summary.json`: sample counts and strata.
- `RATER_INSTRUCTIONS.md`: concise instructions to send to raters.

## Rater Task

For each row, compare `set_a` and `set_b` using the same rubric. Fill all 1--5 fields, then choose one pairwise preference:

- `A`: set A is more useful for triage.
- `B`: set B is more useful for triage.
- `Tie`: both are similarly useful.
- `Neither`: neither set is useful.

Use the free-text field for short reasons, especially when an evidence set appears misleading.

## Rating Rubric

Use 1--5 ordinal scores:

- Relevance: 1 = unrelated, 5 = highly similar to the query context.
- Supportiveness: 1 = contradicts or does not support escalation review, 5 = strongly supports escalation review.
- Actionability: 1 = no next-step investigation value, 5 = clearly suggests useful next steps.
- Explanation quality: 1 = unsuitable for a triage note, 5 = could be directly cited as supporting evidence.
- Misleading safety: 1 = high misleading risk, 5 = low misleading risk.

## Analysis Plan

After collecting completed rater sheets, join responses with `pairwise_key_private.csv`. Report method-level means, paired TRACER-minus-Prefix differences, bootstrap confidence intervals, and pairwise preference percentages. Agreement should be reported with ordinal agreement for Likert fields and Fleiss-style agreement for preference labels when three raters are available.

## Claim Boundary

A positive result supports the claim that TRACER analogs have higher human-rated evidence quality under the audited settings. It does not prove improved SOC efficiency, deployment-level triage gain, or universal user benefit.
