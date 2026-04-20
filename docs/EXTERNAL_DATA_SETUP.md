# External Data Setup

The public TRACER repository already includes the processed benchmark bundles
and stored result summaries used by the paper. This note explains how to
recover the few omitted upstream inputs when you want to rebuild parts of the
pipeline from source materials.

## What is already in the Git repository

- Processed benchmark bundles under `data/`
- Stored result summaries under `outputs/results/`
- Small reusable label tables under `data/reference_labels/`

## What is not fully embedded in Git

- `data/ait_ads_public/ait_ads_canonical_events.csv`
  - distributed as a GitHub Release asset because it is too large for a normal
    git-tracked file
- Raw upstream mirrors under `external_sources/`
  - excluded to keep the repository uploadable and license-aware

## One-command setup

Run:

```powershell
python .\scripts\fetch_external_data.py --targets all
```

This script will:

- copy the bundled reference labels into the expected `external_sources/` paths
- download and unzip the public `ait_ads.zip` source archive from Zenodo
- clone the public `ait-aecid/alert-data-set` repository
- download the released `ait_ads_canonical_events.csv` asset from this GitHub
  repository's `v1.0.0` release
- clone the public `purseclab/ATLAS` repository
- download the curated Splunk Attack Data probe logs used by the released
  `splunk_attack_data_public_probe`

The script writes a machine-readable summary to
`outputs/results/external_data_fetch_status.json`.

## Selective setup

Examples:

```powershell
python .\scripts\fetch_external_data.py --targets reference-labels
python .\scripts\fetch_external_data.py --targets ait-ads-canonical
python .\scripts\fetch_external_data.py --targets splunk-probe-raw atlas-source
```

## Public source locations

- AIT-ADS repository:
  `https://github.com/ait-aecid/alert-data-set`
- AIT-ADS source zip:
  `https://zenodo.org/record/8263181/files/ait_ads.zip`
- ATLAS repository:
  `https://github.com/purseclab/ATLAS`
- Splunk Attack Data repository:
  `https://github.com/splunk/attack_data`
- TRACER release page for large assets:
  `https://github.com/zhyzhangyuezy/TRACER/releases/tag/v1.0.0`

## Notes

- The included `data/reference_labels/*.csv` files are intentionally shipped
  inside the public repository because they are small, stable, and required by
  the published preparation configs.
- The repository still treats upstream raw logs and third-party mirrors as
  external dependencies governed by their original distribution terms.
