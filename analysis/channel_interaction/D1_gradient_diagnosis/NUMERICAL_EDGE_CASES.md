# 极小梯度与历史后端差异

Pancreas 的 U 病例 image0022（repeat 1）与 image0030（repeat 0）在 D1 上，最大 raw 通道范数分别约 1.31e-13 与 2.54e-14；两条观测的 16 个通道全部 below_threshold。LossNorm 重构相对 L2 为 2.16e-4 / 6.93e-4，而最大系数绝对误差仅 1.92e-12 / 3.24e-12。原 Stage1A checker 使用相对与绝对 L2 的联合拒绝条件，故这些极小量通过原绝对误差条件；没有放宽阈值，不据这些 raw 逐裁块方向作结论。病例均值仍严格沿用原 Stage1A 的两裁块平均规则，所有原始状态与空值保留。

LA 历史 GPU FP32 与当前 CPU FP32 的逐观测核梯度比较，最大 relative L2=2.782e-3，中位数约 2.905e-6；最大项来自 U/EJ5V7SPR4961JWD6SS8V/repeat 1/E3/D4，最大系数差约 3.059e-6。该差异归类为跨后端复现差异，未声称逐 bit 或统一 1e-4 误差界成立，也未断言具体算子的原因。四个预指定 D1 通道的病例关系成功复现。完整逐项值在 LA/historical_gradient_replay.csv。
