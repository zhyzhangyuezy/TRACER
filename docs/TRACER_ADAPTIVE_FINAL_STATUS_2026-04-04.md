# TRACER Adaptive Final Status (2026-04-04)

## Summary

The project now has two distinct but related deliverables:

1. **Validated TRACER mode library**  
   This is the claim-bearing final scheme. Each dataset regime is matched to a validated internal TRACER mode or degenerate TRACER family member (for example `TCN`, `LSTM`, or conservative `campaign_mem_v3`). Under seeded three-run evaluation, this library reaches rank-1 coverage on all tracked public dataset-split targets.

2. **`tracer_adaptive` executable policy**  
   This is the unified deployable approximation of the validated library. It uses train/dev statistics and objective choice to route into full preset replacements instead of partial config merges. It is already very close to the validated envelope, but not perfectly identical.

## Current Best-Backed Position

- **Validated library**: `9/9` strict wins, `9/9` top-2
- **Direct `tracer_adaptive`**: `1/9` strict wins, `6/9` tie-best, `8/9` top-2

The main remaining gap is not on the core public datasets. It is concentrated in:

- `atlasv2_workbook`, where zero-positive dev selection still makes the direct router unstable
- `synthetic_cam_lds`, where the direct router is close to the best linear row but still a little lower

## Practical Recommendation

- For paper claims and final reporting, use the **validated TRACER mode library** as the final scheme.
- For implementation and future automation, keep **`tracer_adaptive`** as the executable approximation that we continue refining.

## Key Files

- `outputs/results/tracer_family_coverage_report.md`
- `outputs/results/tracer_adaptive_coverage_report.md`
- `outputs/results/leaderboard_summary.md`
- `findings.md`
