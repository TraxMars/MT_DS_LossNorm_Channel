# 配对数值校准

首次完整运行 round1 保留 LA train mini 0、repeat 1 的重构失败记录；它不是完成结果。相同原始 checkpoint、2L+2U、完整 patch、seed=10001 和 repeat=1，在两套数据上各作 FP32/FP64 的 U MT 重构校准；L forward 保留原 BN/RNG 顺序。正式诊断对 L 和 U 都执行完整重构检查，任一侧触发都重放全单元。U 仅用 image，无参数更新。

| 数据集 | 精度 | U MT 的 D1 LossNorm 重构相对 L2 | 最大绝对误差 |
| --- | --- | --- | --- |
| LA | FP32 | 2.190e-04 | 1.035e-05 |
| LA | FP64 | 1.703e-14 | 2.914e-16 |
| Pancreas | FP32 | 2.757e-05 | 2.429e-06 |
| Pancreas | FP64 | 1.727e-14 | 4.200e-16 |

校准程序仅为记录误差，按 AST 精确选择原 parameter_gradients 函数并屏蔽其中两个重构异常抛出，保留所有公式及检查数值；这个记录版本不用于正式接受梯度。正式采集始终调用原 Stage1A 严格 checker。

统一处理规则：首先 FP32；若原 checker 抛出梯度重构异常，恢复模型、BN、RNG 后，以 FP64 重放整个 L/U 单元，仍使用原阈值。两域同时接受同一精度的结果；失败的局部观测不进入正式汇总。FP64 之后恢复原 FP32 模型状态。新增人工测试模拟 U 失败，验证 L/U 都重放且参数、buffers、dtype 完全恢复。触发机制与余弦符号和数据集效果无关。

目录 calibration_fp32 / calibration_fp64 对应 LA，calibration_pancreas_fp32 / calibration_pancreas_fp64 对应 Pancreas；保留最初单独启动的校准目录，并由本报告组成配对证据。当前 calibrate.py 默认一次执行两套数据集，避免后续遗漏。
