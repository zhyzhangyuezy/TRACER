# ATLASv2 Workbook Bridge

这条桥接路径不依赖 `ATLASv2` 的 12GB 压缩包，而是先使用仓库自带的 `edr_alert_labels.xlsx` 做 appendix-style robustness prototype。

## 输入

- `external_sources/atlasv2/docs/edr_alert_labels.xlsx`
- 可选：`external_sources/reapr-ground-truth/atlasv2/atlasv2_labels.csv`

## 输出

- `data/atlasv2_workbook/train.npz`
- `data/atlasv2_workbook/dev.npz`
- `data/atlasv2_workbook/test.npz`
- `data/atlasv2_workbook/metadata.json`

## 当前做法

- 把 workbook 中的 alert-level rows 正规化为统一事件表
- 将 `report_text` collapse 到固定 `alert_type`
- 按 `host + attack_window` 构造 incident
- 以 5 分钟 bin 聚合为 rolling windows
- 用未来 `malicious` alert onset 生成 `30min / 10min` 标签与 future signature
- 为了让 `dev/test` 保留足够 benign 背景，当前版本会把 benign incident 按时间切片分到不同 split；attack incident 仍保持 incident-level 隔离

## 限制

- 这是 `ATLASv2` appendix robustness 的可执行入口，不是 `CAM-LDS` headline benchmark
- 当前版本优先使用 workbook 里的 event labels；REAPr process labels 只做元数据汇总，尚未完全回写到 event labeling
- 当前 split 不是严格 `incident_leakage_free`，因为 benign 窗口按时间切片跨 split 分配；这一点必须在结果解释里显式写明
- 如果后续下载到完整 `ATLASv2` 原始包，应再增加 raw telemetry -> alert stream 的完整导出脚本
