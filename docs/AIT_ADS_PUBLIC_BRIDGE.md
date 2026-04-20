# AIT-ADS Public Bridge

This bridge converts the public AIT Alert Data Set into the Campaign-MEM rolling-window contract.

## Source Data

- Raw alerts: `external_sources/AIT-ADS/unzipped/*.json`
- Stage intervals: `external_sources/AIT-ADS/labels.csv`
- Output benchmark: `data/ait_ads_public/`

## Mapping

- Each raw source file becomes one incident:
  - `aitads/<scenario>-wazuh`
  - `aitads/<scenario>-aminer`
- Each scenario becomes one family:
  - `aitads/<scenario>`
- Stage labels are assigned by interval lookup against `labels.csv`.
- High-risk stages are:
  - `webshell`
  - `reverse_shell`
  - `privilege_escalation`
  - `service_stop`
  - `dnsteal`

## Feature Contract

- Window size: `4` bins
- Bin width: `5` minutes
- Main horizon: `30` minutes
- Auxiliary horizon: `10` minutes
- Prefix channels:
  - alert-category counts
  - `event_count`
  - `severity_mean`
  - `severity_max`
  - `high_risk_count`
  - `host_count`
- Count channels are compressed with `log1p` before saving to `npz`.

## Split Protocol

- Chronological train/dev/test uses incident-level ordering.
- `test_event_disjoint` holds out full scenario families.
- Current held-out families are recorded in `data/ait_ads_public/metadata.json`.

## Commands

```powershell
python .\scripts\prepare_ait_ads_public.py --config .\configs\data\ait_ads_public.yaml
python .\scripts\audit_campaign_dataset.py --dataset-dir .\data\ait_ads_public --output .\outputs\results\r066_ait_ads_public_audit.json
```

## Initial Readout

- Audit:
  - `incident_leakage_free=True`
  - `event_disjoint_family_free=True`
- Initial single-seed comparison:
  - summary file: `outputs/results/r067_r072_ait_ads_public_summary.json`
  - current best chronological model: `Campaign-MEM` (`AUPRC=0.5532`)
  - current best event-disjoint `AUPRC`: `DLinear` (`0.4560`)
  - current best retrieval metrics on event-disjoint: `Campaign-MEM` (`AF@5=95.38`, `LeadTime@P80=17.5`)
