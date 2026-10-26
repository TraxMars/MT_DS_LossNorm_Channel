# D1-A 第一轮 FINDINGS

已完成 LA 10% 与 Pancreas 10% 的 frozen eval、匹配 eval mini-batch 与真实 2L+2U train-mode 梯度诊断。原始结果分别保存，统一比较见 [跨数据集报告](CROSS_DATASET_REPORT.md)。

- **LA**：D1/D1 eval 有效病例组合的负余弦比例为 40.0%，有 4/16 个通道的探索性病例区间位于负侧。详见 [LA FINDINGS](results/round2_checked/LA/FINDINGS.md)。
- **Pancreas**：D1/D1 eval 有效病例组合的负余弦比例为 49.8%，有 0/16 个通道的探索性病例区间位于负侧。详见 [Pancreas FINDINGS](results/round2_checked/Pancreas/FINDINGS.md)。

**负余弦不能解释为有害通道。**两套独立模型不按通道编号匹配；没有基于任一数据集修改筛选或交互规则。单数据集成立不代表跨数据集普遍性。

两套工程验证通过；本轮可继续扩大冻结诊断，证据不足以直接固定通道筛选或启动 POS/交换。完整限制、共同点、差异和每数据集继续条件均在独立报告中列明。
