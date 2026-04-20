# Expert Evidence-Rating Packet

This packet is prepared for a future human analyst study. It is not a completed user study and should not be reported as human-evaluation evidence until independent raters fill the blinded CSV.

## Files

- `annotation_items_blinded.csv`: blinded items for raters. It hides the model family, true future label, exact score, incident IDs, and retrieved train labels.
- `annotation_key_private.csv`: private key for post-study analysis. Do not share this file with raters.
- `annotation_packet_summary.json`: sampling summary and item counts.

## Sampling

The packet contains 160 item-method rows from ATLASv2 held-out-family and AIT-ADS chronology. Queries are sampled deterministically from positives, top-decile scores, median-score windows, and random background windows. Each sampled query is paired across TRACER route and Prefix-Only retrieval so a paired analysis can compare evidence quality without exposing the method name to raters.

## Rating Rubric

For each item, inspect the query channels and the top-5 retrieved analog channel summaries.

- `semantic_similarity_0_2`: 0 = unrelated, 1 = partially related, 2 = clearly related.
- `shared_high_risk_precursor_0_1`: 1 if at least one analog shares a plausible high-risk precursor with the query.
- `supports_escalation_0_2`: 0 = does not support escalation, 1 = weak support, 2 = strong support.
- `misleading_0_1`: 1 if the analog set appears likely to mislead escalation judgment.
- `confidence_1_5`: rater confidence in the above judgment.
- `free_text_rationale`: short reason, especially when marking misleading evidence.

## Recommended Study Protocol

Use 2 or 3 raters with security operations or intrusion-analysis experience. Randomize item order per rater. Compute Cohen/Fleiss kappa for binary fields, weighted kappa or ICC for ordinal fields, and paired bootstrap differences between TRACER and Prefix-Only after joining ratings with `annotation_key_private.csv`.

## Claim Boundary

This packet only prepares a human evidence-rating study. Until ratings are collected, it should be cited as a released annotation protocol/artifact, not as evidence that TRACER improves analyst decisions.
