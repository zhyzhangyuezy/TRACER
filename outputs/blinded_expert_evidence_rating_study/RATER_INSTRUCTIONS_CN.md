# 评估者说明

感谢你参与本次证据质量评估。本研究不是安全运营中心上线实验，也不要求你重新标注数据集。你的任务是比较两组匿名历史案例证据，判断哪一组更适合支持当前告警升级风险的分析。

请注意，模型名称、当前 query 的真实未来标签、真实 incident 标识和 family 标识都已隐藏。请不要尝试推断哪一组来自哪个模型。

对每一行样本，请按以下步骤完成：

1. 阅读当前 query 的告警通道和阶段摘要。
2. 阅读 `set_a` 与 `set_b` 中各自的 top-5 historical analogs。
3. 分别为 `set_a` 与 `set_b` 填写 1--5 分评分：相关性、支持性、可行动性、解释质量、误导安全性。
4. 在 `preferred_set` 中填写 `A`、`B`、`Tie` 或 `Neither`。
5. 如果某一组明显更好、两组都很弱，或某一组可能误导判断，请在 `free_text_rationale` 中写一句简短理由。

评分含义如下：

- Relevance：1 = 几乎无关，5 = 与当前告警上下文高度相似。
- Supportiveness：1 = 不支持或与升级判断相矛盾，5 = 明确支持升级风险分析。
- Actionability：1 = 对下一步调查没有帮助，5 = 能清楚提示调查方向。
- Explanation quality：1 = 不适合写入 triage note，5 = 可以直接作为解释性证据。
- Misleading safety：1 = 误导风险高，5 = 误导风险低。

请不要打开或使用 `pairwise_key_private.csv`。该文件只用于所有评分完成后的统计分析。
