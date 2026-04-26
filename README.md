# TRACER: Knowledge-Guided Case-Based Warning for Alert Escalation Forecasting

This repository is the archival code-and-data release for the TRACER paper.
It contains the model code, processed benchmark bundles, stored result summaries,
and figure-generation scripts used to support the experimental claims in the
manuscript.

## Local workspace note

Inside the broader private research workspace, this study now lives under
`studies/TRACER/`. Run the commands below from that directory when working in
the private workspace. Private planning, refinement, and review notes are kept
under `notes/` and are intentionally excluded from the public GitHub package.

## What is included

- `campaign_mem/`: TRACER model, data pipeline, and training utilities
- `configs/`: experiment configuration files
- `scripts/`: benchmark builders, audit scripts, and analysis utilities
- `figures/`: figure/table generators and paper-ready artifacts
- `data/`: processed benchmark bundles and reproducibility examples
- `outputs/results/`: stored result summaries and audit exports
- `outputs/expert_evidence_annotation_packet/`: blinded annotation scaffold
- `outputs/blinded_expert_evidence_rating_study/`: paired A/B rater sheets, completed de-identified ratings, and analysis key for the blinded security-rater evidence-rating study
- `docs/`: organized bridge notes, contracts, runbooks, release notes, and status memos
- `notes/`: private planning, refinement, and review materials excluded from the public release

## Data availability

The Git-tracked package includes the processed benchmark splits used in the
paper, together with stored result summaries that regenerate the reported tables
and figures.

The repository also includes small upstream label tables under
`data/reference_labels/` and a helper script `scripts/fetch_external_data.py`
that can recover the remaining public upstream inputs into `external_sources/`
when you want to rebuild the bridges from source materials.

One oversized file, `data/ait_ads_public/ait_ads_canonical_events.csv`, is not
placed in the Git-tracked package because it exceeds the standard GitHub file
size limit. The release builder packages it separately as a GitHub Release asset
so that the repository remains uploadable without Git LFS.

Additional third-party raw-source mirrors under `external_sources/` are also
excluded from the Git-tracked package. The repository instead provides the
processed manifests, builders, and provenance notes required to reconstruct the
released bundles from the cited public sources.

The public GitHub package intentionally excludes manuscript source files and
compiled PDFs. The repository is limited to experiment code, processed data,
stored results, and release-facing documentation.

See [CODE_AND_DATA_AVAILABILITY.md](CODE_AND_DATA_AVAILABILITY.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details. For upstream
fetch instructions, see [docs/runbooks/EXTERNAL_DATA_SETUP.md](docs/runbooks/EXTERNAL_DATA_SETUP.md).

## Environment

For a CPU-only environment:

```bash
conda env create -f environment.cpu.yml
conda activate tracer-cpu
```

For a GPU-capable environment:

```bash
conda env create -f environment.gpu.yml
conda activate tracer-gpu
```

If you prefer `pip`, a baseline dependency list is provided in
`requirements.txt`.

## Reproducing the paper artifacts

Representative commands:

```bash
python figures/gen_tables_public_results.py
python scripts/build_public_event_significance_audit.py
python scripts/build_cross_benchmark_rank_ensemble_audit.py
```

The stored summaries under `outputs/results/` are the authoritative inputs for
the released result tables and figures in this archival snapshot. The public
GitHub package intentionally excludes manuscript sources, so PDF rebuilds are
performed only in the private paper workspace rather than in this repository.

## Recovering omitted upstream inputs

To fetch the public upstream materials that are intentionally not fully embedded
in git, run:

```bash
python scripts/fetch_external_data.py --targets all
```

This command restores the small reference labels into `external_sources/`,
downloads the large AIT-ADS canonical-events release asset, clones the public
AIT-ADS and ATLAS upstream repositories, and downloads the curated Splunk probe
raw logs used by the released bridge.

## Building the GitHub release package

Run:

```bash
python scripts/build_github_release.py
```

This creates:

- a clean GitHub-ready repository tree under `dist/gh_repo/`
- a zip archive for direct upload
- a separate release asset for the oversized AIT-ADS canonical events table
- a manifest and checksum summary

## Citation

Please cite the paper and the software snapshot if you use this repository.
Machine-readable citation metadata are provided in `CITATION.cff`.

## License

This repository is released under the MIT License for the code written in this
project. Third-party datasets, mirrors, and upstream resources remain subject
to their own licenses and terms.
