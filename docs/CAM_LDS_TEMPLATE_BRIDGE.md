# CAM-LDS Template Bridge

这个桥接脚本面向已经完成 canonical label mapping 的 `CAM-LDS` 事件表。

## 入口脚本

```powershell
python .\scripts\prepare_canonical_alerts.py --config .\configs\data\cam_lds_template.yaml
```

## 输入事件表要求

至少包含这些列：

- `timestamp`
- `incident_id`
- `family_id`
- `alert_type` 或 `report_text`
- `severity`
- `is_high_risk` 或 `stage`

可选列：

- `host_hash`
- `src_role`
- `dst_role`

## 当前用途

- 这是 `CAM-LDS` 主 benchmark 的预处理模板
- 一旦 `attackbed / IDS` 导出的 canonical CSV 准备好，就可以直接生成 `train/dev/test.npz`
- 它默认 incident-level split，不会像 `ATLASv2 workbook` 那样为了保留 benign 背景而跨 split 切 benign incident
