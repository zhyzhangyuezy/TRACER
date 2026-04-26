# Blinded Expert Evidence-Rating Study Package

This directory contains the paired A/B evidence-rating packet, completed de-identified rater sheets, and analysis inputs for the TRACER KBS submission.

## Use

Send one randomized rater sheet to each evaluator:

- `rater_01_sheet.csv`
- `rater_02_sheet.csv`
- `rater_03_sheet.csv`

Also send:

- `RATER_INSTRUCTIONS.md` or `RATER_INSTRUCTIONS_CN.md`

Do not send `pairwise_key_private.csv` to raters before annotation is complete. The word `private` means hidden from raters during the blinded study; the released post-study key maps the anonymized A/B evidence sets back to TRACER route and Prefix-Only retrieval so the reported tables can be reproduced.

## After Rating

The completed de-identified sheets in this snapshot are:

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
