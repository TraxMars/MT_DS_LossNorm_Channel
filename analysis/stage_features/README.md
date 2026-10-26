# VNetDS 九阶段特征接口（阶段 0）

工作目录为 `/aita_data/home/aita_duanxuelong/MT_DS_LossNorm_Boundary`。
本阶段提供已有特征的可微分访问接口和兼容性验证，不包含边界损失、边界头、通道筛选、特征交换或双向交互，不启动正式训练。

## 使用方法

将项目 `code/` 加入 Python 导入路径，与原训练入口一致：

```python
from networks import VNetDS
from boundary import get_stage_feature, stage_feature_info, describe_stage_features

model = VNetDS(
    n_channels=1, n_classes=2, n_filters=16,
    normalization="batchnorm", has_dropout=True,
)

outputs, features = model(images, deepsup=True, return_features=True)
e5 = get_stage_feature(features, "encoder", "E5")
d1 = get_stage_feature(features, "decoder", "D1")
info = stage_feature_info(features, "decoder", "D1")
# info["channels"], info["spatial_shape"], info["requires_grad"], info["source"]
all_info = describe_stage_features(features)
```

辅助函数返回原张量引用及普通元数据，检查分组、阶段名、缺失值、类型和五维非空 shape。通道数来自真实张量，支持使用非默认 n_filters 的模型，不执行 detach、clone、前向重算、hook 或模型状态更新。不要对这些张量做原地修改。

四种 forward 组合：

| 调用 | 返回值 |
| --- | --- |
| `model(images)` | 原最终 logits tensor |
| `model(images, deepsup=True)` | 原四尺度 logits list |
| `model(images, return_features=True)` | `(最终 logits, features)` |
| `model(images, deepsup=True, return_features=True)` | `(四尺度 logits list, features)` |

已有 `encoder(images)` 返回五元素 list、`decoder(encoder_features)` 返回四尺度 list 的行为也保留。新增 `decoder(..., return_features=True)` 返回 `(四尺度 list, decoder_features)`。

## 阶段位置与尺寸

features 的键严格为 `encoder: E1,E2,E3,E4,E5` 和 `decoder: D4,D3,D2,D1`。
下表通道数基于 n_filters=16；输入空间维应为 16 的正整数倍。

| 名称 | 真实变量 | 通道 | 相对输入空间尺寸 | CPU 32³ Patch 空间尺寸 | 原始模块/位置 |
| --- | --- | --- | --- | --- | --- |
| E1 | x1 | 16 | H×W×D | 32×32×32 | block_one 输出 |
| E2 | x2 | 32 | H/2×W/2×D/2 | 16×16×16 | block_two 输出 |
| E3 | x3 | 64 | H/4×W/4×D/4 | 8×8×8 | block_three 输出 |
| E4 | x4 | 128 | H/8×W/8×D/8 | 4×4×4 | block_four 输出 |
| E5 | x5 | 256 | H/16×W/16×D/16 | 2×2×2 | block_five，原可选 Dropout 后 |
| D4 | x5_up | 128 | H/8×W/8×D/8 | 4×4×4 | block_five_up(x5)+x4，out_conv_4 输入 |
| D3 | x6_up | 64 | H/4×W/4×D/4 | 8×8×8 | block_six_up(block_six(...))+x3，out_conv_3 输入 |
| D2 | x7_up | 32 | H/2×W/2×D/2 | 16×16×16 | block_seven_up(block_seven(...))+x2，out_conv_2 输入 |
| D1 | x9 | 16 | H×W×D | 32×32×32 | block_nine，原可选 Dropout 后，out_conv 输入 |

对可训练 Student，train 与 eval 中启用 autograd 时，上述九层均 `requires_grad=True`。
`eval()` 本身不关闭梯度；`torch.no_grad()` / inference 或冻结的 Teacher 则返回 `requires_grad=False`。
本接口保持调用者的 autograd 上下文，不自行强开梯度。

| 名称 | LA 112×112×80 | Pancreas / BraTS19 96³ |
| --- | --- | --- |
| E1 / D1 | 112×112×80 | 96×96×96 |
| E2 / D2 | 56×56×40 | 48×48×48 |
| E3 / D3 | 28×28×20 | 24×24×24 |
| E4 / D4 | 14×14×10 | 12×12×12 |
| E5 | 7×7×5 | 6×6×6 |

outputs 维持 `[out_conv_4(D4), out_conv_3(D3), out_conv_2(D2), out_conv(D1)]` 的低到高分辨率顺序。
上游 DuCiSC 的 `valid_feat[0]` 是 block_nine 之前的 x8_up，本接口的 D1 是 x9，两者不应混用。E5、D1 采用 Dropout 后的原路径张量，训练时不额外消耗随机数。

## 复现验证

在项目目录执行，使用现有 SSL29 环境，无新增测试依赖：

```bash
STAGE0_PY=/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python

$STAGE0_PY -B tests/test_stage_features.py --report analysis/stage_features/CPU_RESULTS.json
$STAGE0_PY -B tests/verify_core.py --original analysis/stage_features/reference/upstream_vnet.py
$STAGE0_PY -B tests/verify_stage0_integrity.py

# 需在可访问 GPU 的环境执行；选择有足够空闲显存的设备。
$STAGE0_PY -B tests/test_stage_features.py --gpu-smoke --gpu-index 4 \
  --report analysis/stage_features/GPU_RESULTS.json
```

CPU 测试使用标准库 unittest，包含原 Baseline、当前默认接口、当前特征接口的逐项精确比较。
GPU smoke 按每个数据集原 Patch 大小读取一个 2 标注+2 无标签 batch，在 train/eval 中进行前向与反向验证；使用原 Student 和同一步 Teacher checkpoint，不调用训练入口、不执行 optimizer.step、不保存 checkpoint。

GPU 保留原训练的 `cudnn.deterministic=True`、`benchmark=False`。
当前 PyTorch 的 CUDA CE reduction 不提供全局严格确定性实现。因此 GPU 网络输出、原始损失梯度、BN 和 RNG 仍要求 atol=rtol=0；损失标量及 LossNorm 组合梯度显式使用 rtol=1e-6、atol=1e-7，并提供原 Baseline 自身重复运行对照。CPU 的这些比较全部使用零容差。初次未通过的 GPU 尝试也保留记录，详见 `GPU_ATTEMPTS.json`。

完整测试需要六组原最佳 checkpoint；它们保留在本地 Results 中，未纳入 Git。`reference/vnet_ds_baseline.py` 为修改前原文件，测试先校验 SHA256；`reference/upstream_vnet.py` 为在线上游参考文件，保留文件末尾换行，其内容与本地上游仅末尾换行不同。

## 记录与边界

- `AUDIT.md`：训练流程、上游对应关系及既有协议限制。
- `BASELINE_MANIFEST.json`：修改前本地 main、关键文件及完整受保护内容 SHA256。
- `EXPERIMENT_CONFIGS.json`：六组已完成实验的配置、划分和完成状态。
- `CPU_RESULTS.json`、`GPU_RESULTS.json`、`INTEGRITY_RESULTS.json`：机器可读的实际结果。
- `VERIFICATION.md`：实际执行情况、修改文件清单和兼容性结论。

历史 Results/summary/config 中记录的原绝对路径保留；这些历史结果不表示在新目录重新完成了实验。后续正式实验需使用新的输出目录，不能复用历史目录。本阶段没有调用任何 launcher。

九层均已保留反向图，后续可在各层接独立边界监督分支。边界标签构造、head 结构、损失权重、归一化与 warm-up 设计均属于后续阶段，需要独立实现与验证；当前只完成特征访问的基础条件。
