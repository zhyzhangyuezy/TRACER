# Blinded Expert Evidence-Rating Study Package

This directory contains the ready-to-use paired A/B evidence-rating packet for the TRACER KBS submission.

## Use

Send one randomized rater sheet to each evaluator:

- `rater_01_sheet.csv`
- `rater_02_sheet.csv`
- `rater_03_sheet.csv`

Also send:

- `RATER_INSTRUCTIONS.md` or `RATER_INSTRUCTIONS_CN.md`

Do not send the private key until all ratings are complete. In the public repository, `pairwise_key_private.csv` is intentionally omitted from the latest tree. The private workspace keeps this key for post-study analysis because it maps the anonymized A/B evidence sets back to TRACER route and Prefix-Only retrieval.

## After Rating

After each evaluator completes a sheet, save the completed files in this directory using names such as:

- `rater_01_completed.csv`
- `rater_02_completed.csv`
- `rater_03_completed.csv`

Then run:

```bash
python scripts/analyze_blinded_pairwise_evidence_ratings.py
```

The analysis script writes JSON results to `outputs/results/` and paper-ready LaTeX tables to `figures/`.

## Claim Boundary

This package supports human-rated evidence quality analysis. It does not by itself prove improved SOC triage performance or deployment-level user benefit.
