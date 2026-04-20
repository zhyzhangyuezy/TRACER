# Code and Data Availability

This repository is prepared as the public archival package for the TRACER paper.
Its goal is to provide a GitHub-uploadable snapshot that supports code and data
availability claims in the manuscript while staying within standard GitHub file
limits.

## Code included in the Git-tracked package

- `campaign_mem/`: model code, metrics, dataset loaders, training engine
- `configs/`: released configuration files
- `scripts/`: benchmark builders, audits, aggregation scripts
- `figures/`: plotting scripts, generated tables, generated figures
- `docs/`: benchmark bridge notes and release-facing documentation

## Data and result artifacts included in the Git-tracked package

- `data/atlasv2_public/`
- `data/atlasv2_lopo_family/`
- `data/atlasv2_workbook/`
- `data/atlas_raw_public/`
- `data/ait_ads_public/` except the oversized canonical-events CSV
- `data/reference_labels/`
- `data/cross_dataset_transfer/`
- `data/examples/`
- `data/splunk_attack_data_public_probe/`
- `data/synthetic_cam_lds/`
- `data/synthetic_cam_lds_controlled/`
- `outputs/results/`
- `outputs/expert_evidence_annotation_packet/`

These artifacts are the processed bundles and stored summaries used to generate
the reported tables, figures, and audit results in the released experiment
snapshot.

## Large file handled as a release asset

The following file is intentionally excluded from the Git-tracked repository:

- `data/ait_ads_public/ait_ads_canonical_events.csv`

Reason:

- the file is larger than the standard GitHub 100 MB single-file limit
- tracking it directly in git would make the repository hard to upload and
  maintain without Git LFS

The release builder packages this file separately as a compressed GitHub Release
asset so that the main repository remains standard and uploadable.

## Small external reference files included directly

The repository ships two small label tables under `data/reference_labels/`:

- `ait_ads_labels.csv`
- `atlasv2_labels.csv`

These files are lightweight but operationally important because the released
preparation configs reference them through `external_sources/`. The helper
script `scripts/fetch_external_data.py` can copy them back into those expected
paths and can also fetch the larger public upstream materials that are omitted
from git.

## Materials intentionally excluded from the Git-tracked package

- `external_sources/`: raw upstream mirrors and third-party repository clones
- `.codex/`, `_qa_*`, `refine-logs/`, and other local tooling directories
- local review notes and temporary files not required for reproduction
- manuscript sources and paper build intermediates

These exclusions keep the archival repository focused on experiment
reproducibility rather than local workspace state.

## Recommended public release structure

1. Push the clean repository tree generated under `dist/github_repo/` to GitHub.
2. Attach the large-file archive generated under `dist/release_assets/` to the
   corresponding GitHub Release.
3. Cite both the repository URL and the release asset in the final paper's data
   availability statement.

For a scriptable recovery path, see `docs/EXTERNAL_DATA_SETUP.md`.

## Provenance boundary

The repository includes processed bundles, scripts, and provenance notes needed
to understand and regenerate the released experimental artifacts. Upstream raw
logs, benchmark mirrors, and third-party datasets remain governed by their
original licenses and distribution terms.
