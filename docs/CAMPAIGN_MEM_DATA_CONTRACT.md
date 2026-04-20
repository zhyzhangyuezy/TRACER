# Campaign-MEM Data Contract

`Campaign-MEM` 当前已经有可跑的实验代码，但真实 `CAM-LDS / ATLASv2` 还没有放进仓库，所以先固定“处理后数据”的输入契约。

## 目录结构

每个数据集目录至少包含：

- `metadata.json`
- `train.npz`
- `dev.npz`
- `test.npz`
- `test_event_disjoint.npz`：可选，但强烈建议保留给 B3

## `.npz` 必需字段

- `prefix`: `float32 [N, T, D]`
  当前 query 窗口的 alert-only prefix 特征
- `label_main`: `float32 [N]`
  主任务标签，表示未来 30 分钟内是否首次进入 high-risk escalation
- `label_aux`: `float32 [N]`
  辅助任务标签，表示未来 10 分钟内是否首次进入 high-risk escalation
- `future_signature`: `float32 [N, S]`
  训练 `risk-shape retrieval` 的未来风险轨迹签名
- `time_to_escalation`: `float32 [N]`
  到首次 escalation onset 的分钟数；负例可用大于 horizon 的截断值
- `incident_id`: `str [N]`
  唯一 incident / episode 标识，用于 leakage audit
- `family_id`: `str [N]`
  campaign family / scenario family 标识，用于 event-disjoint 评估
- `timestamp`: `int64 [N]`
  query 时刻时间戳，便于做 chronological split 检查

## `metadata.json` 建议字段

- `dataset_name`
- `main_horizon_minutes`
- `aux_horizon_minutes`
- `analog_fidelity_distance_threshold`
- `notes`

## 对应 refine-logs 中的协议

- M0 / R001: `incident_id` 和 `family_id` 必须可审计
- B1 / B2: `future_signature` 是 `Campaign-MEM` 和 `Prefix-Only-Retrieval` 的共同评测接口
- B3: `test_event_disjoint.npz` 应保证 `family_id` 不与 `train` 交叉
- B4: `ATLASv2` 的 fixed detector bundle 与 vocabulary collapse 需要在导出前完成，而不是训练时临时处理

## 当前状态

- 已内置 `data/synthetic_cam_lds/` 的合成 smoke 数据格式生成脚本
- 真实 `CAM-LDS` 和 `ATLASv2` 仍需按上面格式导出后才能启动真实 runs
