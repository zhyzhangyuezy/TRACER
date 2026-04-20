# Reference Labels

This directory contains small upstream label tables that are required to
rebuild some processed public bundles from raw-source inputs but are lightweight
enough to ship directly inside the Git-tracked release.

Included files:

- `ait_ads_labels.csv`: copied from the public AIT-ADS distribution used by
  `external_sources/AIT-ADS/labels.csv`
- `atlasv2_labels.csv`: copied from the public REAPr label table used by
  `external_sources/reapr-ground-truth/atlasv2/atlasv2_labels.csv`

The helper script `scripts/fetch_external_data.py` can materialize these files
back into the expected `external_sources/` locations if you want to rerun the
preparation bridges from raw inputs.
