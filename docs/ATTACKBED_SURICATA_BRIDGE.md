# AttackBed Suricata Bridge

If the material exported from `attackbed` is raw Suricata EVE JSON rather than a cleaned canonical CSV, use this path:

1. `Suricata EVE JSON -> canonical raw alerts`
2. `stage interval labeling`
3. `canonical alerts -> Campaign-MEM npz`

## 1. Normalize Suricata alerts

```powershell
python .\scripts\normalize_suricata_eve.py --config .\configs\data\attackbed_suricata_template.yaml
```

The normalized CSV should contain at least:

- `timestamp`
- `incident_id`
- `family_id`
- `alert_type`
- `severity`
- `report_text`
- `host_hash`

## 2. Apply canonical stage intervals

```powershell
python .\scripts\apply_stage_intervals.py --config .\configs\data\cam_lds_stage_labeling_template.yaml
```

Use the generic template at [CAM_LDS_STAGE_INTERVALS_TEMPLATE.csv](/C:/Users/zhyzh/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/templates/CAM_LDS_STAGE_INTERVALS_TEMPLATE.csv) when labeling real exports.

## 3. Prepare Campaign-MEM windows

```powershell
python .\scripts\prepare_canonical_alerts.py --config .\configs\data\cam_lds_template.yaml
```

## Smoke Example

The repository includes a self-contained smoke path under `data/examples/suricata/` that exercises the full bridge end to end:

```powershell
python .\scripts\normalize_suricata_eve.py --config .\configs\data\attackbed_suricata_example.yaml
python .\scripts\apply_stage_intervals.py --config .\configs\data\cam_lds_stage_labeling_example.yaml
python .\scripts\prepare_canonical_alerts.py --config .\configs\data\cam_lds_example.yaml
python .\scripts\audit_campaign_dataset.py --dataset-dir .\data\examples\cam_lds_attackbed_example --output .\outputs\results\attackbed_suricata_example_audit.json
python .\scripts\run_campaign_experiment.py --config .\configs\experiments\smoke_campaign_mem_attackbed_example.yaml
```

This example uses:

- sample inputs in [data/examples/suricata](/C:/Users/zhyzh/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/data/examples/suricata)
- stage intervals in [sample_stage_intervals.csv](/C:/Users/zhyzh/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/data/examples/suricata/sample_stage_intervals.csv)
- a sample dataset config in [cam_lds_example.yaml](/C:/Users/zhyzh/OneDrive/跑程序/auto科研/Auto-claude-code-research-in-sleep-main/configs/data/cam_lds_example.yaml)

## Notes

- This bridge is the landing scaffold for the real `CAM-LDS` benchmark.
- It assumes you can recover stage time intervals from scenario logs, attack manifests, or a manually curated timeline.
- If the final export includes Wazuh, Sigma, or Zeek notice streams in addition to Suricata, merge them into the same canonical CSV schema before running `prepare_canonical_alerts.py`.
