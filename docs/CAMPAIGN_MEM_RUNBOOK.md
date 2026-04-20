# Campaign-MEM Runbook

这个 runbook 对应根目录 `refine-logs/EXPERIMENT_PLAN.md` 的 M0-M3 首批实验。

## 1. 生成 smoke 数据

```powershell
python .\scripts\generate_synthetic_cam_lds.py --config .\configs\data\synthetic_cam_lds.yaml
```

## 2. 做协议审计

```powershell
python .\scripts\audit_campaign_dataset.py `
  --dataset-dir .\data\synthetic_cam_lds `
  --output .\outputs\results\r001_protocol_audit_smoke.json
```

## 3. 跑首批基线与主模型

```powershell
python .\scripts\run_campaign_experiment.py --config .\configs\experiments\r004_tail_risk_linear.yaml
python .\scripts\run_campaign_experiment.py --config .\configs\experiments\r007_pure_knn.yaml
python .\scripts\run_campaign_experiment.py --config .\configs\experiments\r008_prefix_retrieval.yaml
python .\scripts\run_campaign_experiment.py --config .\configs\experiments\r009_campaign_mem.yaml
```

## 4. 当前代码覆盖的实验块

- B0: dataset audit / leakage audit 脚本已实现
- B1: `tail_risk_linear`, `tcn`, `transformer`, `pure_knn`, `prefix_retrieval`, `campaign_mem`
- B2: `campaign_mem` 可通过 config 切换 `use_contrastive / use_hard_negatives / use_auxiliary / use_utility`
- B3: 支持额外读取 `test_event_disjoint.npz`

## 5. 当前还缺什么

- 真实 `CAM-LDS` 的 canonical label mapping 导出
- 真实 `ATLASv2` 的 fixed detector bundle projection 导出
- 对 `utility term` 的 gate 逻辑做真实 dev-set 判定

## 6. 输出位置

- 日志建议写到 `outputs/logs/`
- JSON 结果写到 `outputs/results/`
- 真实数据跑通后，把摘要补到根目录 `refine-logs/`
