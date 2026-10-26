# 全局真实数据实验约束（自 Stage D0 起）

本项目后续真实数据诊断、测试、干预及正式训练必须同时覆盖 LA 10% 和 Pancreas 10%。人工张量单元测试可独立执行。

- LA：`Results/seed_42/LA_8l/`，8 L + 72 U，patch 112×112×80。
- Pancreas：`Results/seed_42/Pancreas_6l/`，6 L + 56 U，patch 96×96×96。
- 两个数据集采用统一设计、公式、指标、有效性阈值和统计方法；保留各自原 patch、标注比例和原始划分。
- 分别加载各自同一序列化 checkpoint 的 Student/Teacher；核验 iteration、配置、权重键、shape、有限性和 SHA256，禁止混用数据及权重。
- U GT 不得作为梯度、通道选择或交互的输入。诊断路径只读取 U `image`，不得沿用原 `VolumeDataset` 会读取 U label 的行为。
- 分数据集保存原始结果，同时生成跨数据集比较；每阶段分别报告发现、相同点、差异、局限和继续条件。
- 不得因单数据集效果修改筛选或交互规则；单数据集成立的现象不能宣称跨数据集普遍性；独立训练模型不按通道编号匹配。
- 不覆盖原 Baseline、Stage1A/B/C、历史 Results 或历史分析产物。新运行必须创建不存在的独立目录。
- D1-A 只做梯度诊断：复用 Stage1A independent_losses、参数梯度和总目标 LossNorm；复用 C0 通用节点身份规则，不引入边界目标或位置对目标。
- D1-A 先 frozen eval，再真实 2L+2U train-mode；保护并恢复参数、BN buffers、Dropout RNG、grad slots、requires_grad 和模块模式。不创建 optimizer、不执行 optimizer.step、不做 POS 或通道交换。
- 负余弦仅说明当前目标在同一参数坐标上的方向关系，不能解释为有害通道或因果证据。
