# Data Directory

This directory contains the processed benchmark bundles and small reproducibility
examples released with the TRACER archival package.

Included bundles:

- `atlasv2_public/`
- `atlasv2_lopo_family/`
- `atlasv2_workbook/`
- `atlas_raw_public/`
- `ait_ads_public/` (Git-tracked split bundles; the oversized canonical-events
  CSV is released separately as a GitHub Release asset)
- `reference_labels/` (small upstream label tables included directly for
  rebuild convenience)
- `cross_dataset_transfer/`
- `splunk_attack_data_public_probe/`
- `synthetic_cam_lds/`
- `synthetic_cam_lds_controlled/`
- `examples/`

Each benchmark directory contains processed split artifacts such as `.npz`
bundles and `metadata.json` files used by the released scripts and paper
artifacts.

If you want to recover omitted upstream raw inputs into `external_sources/`, run
`python scripts/fetch_external_data.py --targets all`.
