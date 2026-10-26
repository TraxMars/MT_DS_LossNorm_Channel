# 阶段 1A：多尺度参数梯度关系诊断

仅在冻结的 LA 10% 模型上诊断；不新增训练目标、边界头、通道选择、交换或交互。所有模型、训练/评估代码和六组历史结果保持原样。

主结果目录：[`runs/la10_iter15000_seed142_fp32/`](runs/la10_iter15000_seed142_fp32/)。先阅读该目录的 `REPORT.md`、`scale_summary.csv`、`validity_matrices.json`，然后按需查看逐通道及病例对表。

## 模型与病例设计

- 使用 `Results/seed_42/LA_8l/last_checkpoint.pth`，其中 Student/Teacher 与 iteration=15000 共同序列化，能明确保证同迭代。没有将纯 state_dict 的最佳模型文件时间戳当作同迭代证据。
- Student 和 Teacher 都保持 eval。Student 参数允许 autograd，但不创建 optimizer，也不改变 `.grad`；Teacher requires_grad=False 且 forward 使用 no_grad。
- 固定 seed=142；使用全部 8 个标注 train 病例，从保存的 72 个无标签 ID 中无放回选 16 个。选择只依据 ID，未读取无标签 GT。
- 每病例两个固定种子的均匀随机裁块，Patch=112×112×80；不旋转、不按 GT 前景筛选裁块。病例 ID、裁块起点、padding、原图尺寸完整记录。
- 诊断 batch_size=1；原 Dice 跨 batch/空间求和，因此本结果是病例级诊断，不声称复现原 batch_size=2 子批的训练更新。
- 使用 float32、禁用诊断的 cuDNN/matmul TF32，避免 TF32 舍入影响四个独立梯度与总梯度重构。原训练代码的数值设置没有修改。
- 无标签专用 loader 只查询 HDF5 `image`。没有调用会同时读取 label 的 `VolumeDataset`/`load_case`，没有根据无标签 GT 筛病例、裁块或作指标。

## 九阶段参数映射

程序按 block 角色和模块类型查找真实卷积，不复制 CHAP 的固定列表索引。权重名称是检查后的结果，而非发现规则。只比较同一阶段同一参数坐标，不跨阶段匹配不同维度的卷积核，也不跨病例逐点比较特征图梯度。

| 阶段 | 参数 | 权重 shape | 输出通道轴 | 通道向量长度 | 特征关系 |
| --- | --- | --- | --- | --- | --- |
| E1 | block_one.conv.0.weight | 16×1×3×3×3 | 0 | 27 | block 最后 Conv3d，后接 BN/ReLU |
| E2 | block_two.conv.3.weight | 32×32×3×3×3 | 0 | 864 | 同上 |
| E3 | block_three.conv.6.weight | 64×64×3×3×3 | 0 | 1728 | 同上 |
| E4 | block_four.conv.6.weight | 128×128×3×3×3 | 0 | 3456 | 同上 |
| E5 | block_five.conv.6.weight | 256×256×3×3×3 | 0 | 6912 | 后接 BN/ReLU/原可选 Dropout；eval 不丢弃 |
| D4 | block_five_up.conv.0.weight | 256×128×2×2×2 | 1 | 2048 | 反卷积分支经 BN/ReLU 后再加 E4 |
| D3 | block_six_up.conv.0.weight | 128×64×2×2×2 | 1 | 1024 | 反卷积分支经 BN/ReLU 后再加 E3 |
| D2 | block_seven_up.conv.0.weight | 64×32×2×2×2 | 1 | 512 | 反卷积分支经 BN/ReLU 后再加 E2 |
| D1 | block_nine.conv.0.weight | 16×16×3×3×3 | 0 | 432 | 后接 BN/ReLU/原可选 Dropout，生成最终 head 的输入 |

D4/D3/D2 的参数梯度仅对应可学习上采样分支，不等于整个 skip 相加后特征通道的“全部梯度”；对应编码器参数还会通过 skip 和深路径受到目标影响。各阶段核向量维度不同，原始范数不用于跨阶段判定重要性。

结构可达性：

| 参数阶段 \ 损失头 | D4 | D3 | D2 | D1 |
| --- | --- | --- | --- | --- |
| E1–E5、D4 | 可达 | 可达 | 可达 | 可达 |
| D3 | 不可达 | 可达 | 可达 | 可达 |
| D2 | 不可达 | 不可达 | 可达 | 可达 |
| D1 | 不可达 | 不可达 | 不可达 | 可达 |

上述矩阵经每个实际病例/损失的 `autograd.grad(..., allow_unused=True)` 验证。不可达梯度保存为 NaN/空值并明确注明原因，未用可靠零分替代。

## 损失、梯度及 LossNorm

每头监督项直接调用 `supervised_loss([logit_s], labeled_gt)`，保留 nearest GT 重采样、原 CE 和前景 Dice，以及二者 /2。MT 项直接调用 `consistency_loss([student_s],[teacher_s])`，保留全类别 softmax 概率 MSE。使用原函数验证四个单尺度项的均值与原四尺度目标一致。

令 g_s=∂L_s/∂W，原始头贡献是 g_s/4。训练原实现先对四尺度平均的总监督/一致性目标进行 LossNorm，因而归一化后的单头贡献分别为：

```text
supervised: g_s / (4 * max(stopgrad(Lsup), eps))
MT:         0.3 * g_s / (4 * max(stopgrad(Lmt), eps))
```

没有把四个头各自独立归一化；那会改变原 Baseline。原始独立参数梯度由 autograd.grad 获得，另实际计算总损失及 LossNorm 总损失的参数梯度来核验重构，不向 `.grad` 累积。

对每通道记录原始范数、g_s/4 范数和归一化贡献范数。有效 L/U 有符号余弦保持在 [-1,1]，不取绝对值，也不乘 CHAP 的幅值权重。零范数/非有限/过小梯度不生成余弦；阈值为 max(1e-12, 1e-6×该病例该阶段该尺度最大通道范数)。同一裁块的正标量归一化不改变方向；跨裁块平均前的重新加权可能改变病例均值方向，分别报告 raw/normalized。

## 统计解释

- 两个裁块的原始参数梯度平均得到每病例向量；有标签/无标签病例群体没有自然配对，比较全部 8×16 个病例组合，只在相同权重坐标比较方向。
- 每通道记录 L/U 有效病例对覆盖率、余弦分布、范数和病例间 CV；至少 75% 病例对有效才标为可靠覆盖。其余保留部分观测值和无效原因，不当作可靠评分。
- L/U 平均余弦区间用 200 次分别重采样 L/U 病例的 bootstrap。病例对并非独立样本，不能把 128 对当作 n=128 的独立证据。区间与符号稳定性均属探索性，不是多重检验校正后的发现。
- 同病例不同裁块与不同病例的方向相似度分别记录；跨尺度关系只比较同病例同参数上各 head 的梯度。SVD 使用按头单位化的梯度 Gram 矩阵，抵消比为 ||Σg_s||/Σ||g_s||。
- 通道 PCA/聚类使用按病例平均的 L/U 跨尺度范数占比，排除不可达维度与低覆盖通道。两域占比各自和为 1，代数秩上限为 2×(可达尺度数−1)。D1 只有一个可达头，比例恒定，不做有意义的聚类。
- 仅探索 k=2/3/4，记录 silhouette、各组大小；对 k=2 作 100 次病例 bootstrap ARI 检查。没有固定 Top 25% 或任何用于训练的通道筛选。
- 参数梯度与特征图梯度是不同对象。本诊断没有计算跨病例空间位置的特征图梯度余弦。梯度几何相似/低秩/分组稳定均不能直接证明通道协同、互补或边界作用。

## 参考审计

[CHAP GradSim_3d](https://github.com/gardnerzhou/CHAP/blob/8754bfa5e5d269719b0ca45345c8a28742b13c0d/utils/grad.py) 在编码器末次卷积参数上比较梯度，并组合余弦和相对无监督范数，采用梯度移动平均。这里仅借鉴共享参数坐标概念；不复制其 optimizer.zero_grad/backward 流程、固定索引或训练干预。

[DuCiSC 四头路径](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/networks/VNet.py) 与阶段 0 审计一致。本地 Baseline 的 MT 是概率 MSE，与上游完整 DuCiSC 的硬伪标签、prototype、mix 等目标有区别。本次诊断的是本地 MT+DS+LossNorm，不声称诊断了完整上游训练方法。精确 Git blob/commit 见 `SOURCES.json`。

## 复现与文件

```bash
STAGE1A_PY=/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python
$STAGE1A_PY -B tests/test_stage1a_diagnostics.py

# 需要可访问 GPU 的环境。新运行必须使用不存在的新输出目录。
$STAGE1A_PY -B analysis/stage1a_gradients/run.py --device cuda:1 \
  --output analysis/stage1a_gradients/runs/another_frozen_run \
  --seed 142 --unlabeled-cases 16 --repeats 2

$STAGE1A_PY -B analysis/stage1a_gradients/summarize.py \
  --run analysis/stage1a_gradients/runs/another_frozen_run

$STAGE1A_PY -B analysis/stage1a_gradients/verify_outputs.py \
  --run analysis/stage1a_gradients/runs/another_frozen_run
```

主结果中包括：

- `REPORT.md`：实际结果与限制，含九阶段×四尺度矩阵及边界假设讨论。
- `scale_summary.csv`、`validity_matrices.json`：36 单元矩阵及有效性。
- `per_channel.csv`：逐通道 L/U 范数、相似度、稳定性、bootstrap、无效原因。
- `per_observation_channels.csv`：每裁块/尺度/通道的原始与归一化贡献范数。
- `lu_case_pairs.csv`：每个有效/无效病例对的有符号余弦及两侧状态。
- `cross_scale_summary.csv`、`per_observation_cross_scale.csv`、`per_channel_multiscale.csv`：跨头关系、有效秩与抵消。
- `channel_profiles.csv`、`pca_clusters.json`：描述性 PCA/聚类及病例 bootstrap。
- `figures/`：有效性/LU、病例稳定性、跨尺度、PCA 的 PNG/PDF。
- `design.json`、`parameter_mapping.json`、`losses.csv`、`norm_scaling.json`：设计与原始量。
- `hdf5_access_log.json`、`gradient_reconstruction.json`、`protection_before.json`、`protection_after.json`、`extraction_results.json`：验证证据。
- `raw/`：完整真实卷积参数梯度 NPY，轴为 observation×head×output-channel×flattened-kernel；大文件留在本地，不纳入 Git。

首次目录 `runs/la10_iter15000_seed142/` 保留失败验证记录，不能当作完成分析；最终结果使用 `_fp32` 目录。原始大数组、展开表和图片可由代码复现；其本地路径保持可访问，紧凑结果、报告与脚本纳入版本管理。

当前只有区域分割和 MT 梯度，**不能直接称为边界相关梯度**。后续边界目标、负对照及留出病例验证方案见主结果 `REPORT.md`，本阶段未实现这些目标或训练。
