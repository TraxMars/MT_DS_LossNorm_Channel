# Channel interaction：全局要求与 D0 / D1-A

自 D0 起，所有真实数据实验同时覆盖 LA 10%（8 L / 72 U，112×112×80）与 Pancreas 10%（6 L / 56 U，96³）。分别使用 `Results/seed_42/LA_8l/` 和 `Results/seed_42/Pancreas_6l/` 的原始配对 checkpoint。项目根目录的 [AGENTS.md](../../AGENTS.md) 将此要求持续应用于后续阶段。

两个数据集统一公式、指标、有效性阈值、统计方法和分析代码；保留原划分和 patch。分别保存原始结果，再比较数据集层面的关系；不匹配独立模型的通道编号，不因单数据集结果调整通道规则。U 只读取 HDF5 `image`，不把 U GT 用于梯度、筛选或交互。人工张量单元测试可以独立执行。

## 独立目录

- `D0_integrity/code/`、`tests/`、`results/`：SHA256、模型/划分检查、历史保护。
- `D1_gradient_diagnosis/code/`、`tests/`、`results/`：原始梯度、统计、验证及报告。
- 完成结论分别见两个阶段的 `FINDINGS.md`；跨数据集比较见 D1 的 `CROSS_DATASET_REPORT.md`。

不会修改原 Baseline、原 Stage1A/B/C 或旧 Results。所有新执行目录必须不存在，写入受阶段 `results/` 限制。D1-A 不创建 optimizer，不执行参数更新、EMA 更新、POS、通道交换或边界/位置对目标。

## D1-A 预先固定设计

两套模型均使用 `last_checkpoint.pth` 内同 iteration=15000 的 Student/Teacher，严格加载。首先复现 Stage1A 的 eval、seed=142、每病例两裁块、全部 L、从各自 U ID 无放回选 16 例、batch=1。LA 选择与原 Stage1A 完全一致。关注 D1/D1 的原通道 0、4、10、14，但保存 D1 全部通道及 E3 对 D4/D3/D2/D1 的梯度。Pancreas 分析自身完整通道，不按 LA 编号匹配。

之后每数据集固定三个 2L+2U 病例组，每组两个裁块重复。先做相同输入的 eval mini-batch，再做 train mini-batch；每单元从原 checkpoint BN 状态开始，Student L → Student U → Teacher U。全组完整重放一次以检验 Dropout RNG 的可复现性。LA 的 mini 使用前 6/8 个 L，Pancreas 使用全部 6 个 L；两套均使用前 6/16 个选定 U。这里只比较训练模式的梯度，不声称复现训练数据增强或连续迭代的 BN 轨迹。

采用 Stage1A 的 `independent_losses` 与 `parameter_gradients`。令 `g_{r,h}` 为独立头的参数梯度，`A_r=mean_h L_{r,h}`：

```text
raw head gradient:              g_{r,h}
raw mean-head contribution:    g_{r,h}/4
raw weighted contribution:     w_r*g_{r,h}/4
LossNorm effective contribution: w_r*g_{r,h}/(4*max(stopgrad(A_r),1e-8))
w_L=1, w_U=0.3
```

不是每个头单独归一化。每观测都实际对 aggregate 和 LossNorm aggregate 求导，验证独立头梯度重构。C0 仅加载 `NODE_SPECS` 和 `Capture` 的原 AST，验证真实节点对象、调用次数和 skip fusion；不会执行 C0 的 `support`、边界目标、位置对目标或衰减函数。参数梯度对应 E3 最后 Conv3d、D1 最后 Conv3d 的输出通道核坐标，不是空间特征梯度。

## 统一统计与解释

- 有效性：有限、非零，范数超过 `max(1e-12, 1e-6*同观测同头最大通道范数)`。不可达保留 NaN，JSON/CSV 显式空值；不生成零余弦。
- 病例级：两裁块先平均 raw 或 effective 向量，计算全部 L×U 组合的有符号余弦。分别重采样 L、U 病例 200 次估计探索性区间，不把病例组合视为独立样本。覆盖率至少 75% 才标为有效覆盖。
- 冲突：只表示有效余弦 `<0` 的比例，同时报告区间、病例/裁块方向重复性、mini-batch 符号重复性和无效比例。
- 抵消：`rho=||g_L+g_U||/(||g_L||+||g_U||)`，抵消程度 `1-rho`。raw contribution 保留 U 权重 0.3；effective 使用原 LossNorm。另记录各域跨头抵消。
- 同观测正标量缩放不会改变余弦；先归一化再跨裁块平均可以改变病例均值方向。
- train mini 的六个单位有病例重复，不作 iid 推断或置信区间。BN 与 Dropout 同时改变，未单独归因。
- 单数据集成立的指标必须报告适用范围；负余弦不能解释为有害通道，不足以证明协同、互补、边界语义或训练性能因果效应。

## 复现

使用 `/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python -B`，以下命令的结果目录应更换为不存在的名称。`run.py` 始终执行两个数据集。

```bash
python -B analysis/channel_interaction/D0_integrity/tests/test_integrity.py
python -B analysis/channel_interaction/D1_gradient_diagnosis/tests/test_diagnostics.py
python -B -m unittest discover -s tests -p test_stage1a_diagnostics.py -v
python -B tests/test_stage_features.py --report analysis/channel_interaction/D0_integrity/tests/feature_regression_new.json
python -B analysis/channel_interaction/D0_integrity/code/audit.py --output analysis/channel_interaction/D0_integrity/results/new_audit
python -B analysis/channel_interaction/D0_integrity/code/supplement.py --output analysis/channel_interaction/D0_integrity/results/new_supplement
python -B analysis/channel_interaction/D1_gradient_diagnosis/code/run.py --d0 analysis/channel_interaction/D0_integrity/results/new_audit/integrity.json --output analysis/channel_interaction/D1_gradient_diagnosis/results/new_run --device cpu --threads 4
python -B analysis/channel_interaction/D1_gradient_diagnosis/code/summarize.py --run analysis/channel_interaction/D1_gradient_diagnosis/results/new_run
python -B analysis/channel_interaction/D1_gradient_diagnosis/code/verify.py --run analysis/channel_interaction/D1_gradient_diagnosis/results/new_run
python -B analysis/channel_interaction/D0_integrity/code/audit.py --verify analysis/channel_interaction/D0_integrity/results/new_audit/protection_before.json --output analysis/channel_interaction/D0_integrity/results/new_postcheck
```

GPU 不可用时保留完整 patch 使用 CPU，首先使用 FP32。若原 Stage1A 重构检查失败，恢复状态后用 FP64 重放整个单元的 L/U，仍使用原阈值；两数据集规则一致，与余弦符号和效果无关。逐观测记录精度，失败尝试及恢复证据保留。LA 历史 GPU 与本轮 CPU 的差异另列于 `historical_gradient_replay.csv`，不能宣称逐 bit 一致。
