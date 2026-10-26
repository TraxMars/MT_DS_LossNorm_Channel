# 阶段 0：MT + DS + LossNorm 审计

审计日期：2026-10-21（Asia/Shanghai）。本报告在修改网络接口之前建立。

## 范围与来源

- 工作目录：`/aita_data/home/aita_duanxuelong/MT_DS_LossNorm_Boundary`。
- 原 Baseline：`/aita_data/home/aita_duanxuelong/MT_DS_LossNorm`，只读使用。
- 新目录已存在完整副本，修改前除缓存外与原目录一致。
- 本地没有可用的原 Git 历史，因此创建的是当前文件系统的 **本地 main 快照**，不宣称恢复了原历史提交。完整 main SHA 见 `BASELINE_MANIFEST.json`。
- 已先建立 `experiment/stage0-feature-interface` 分支，再记录 SHA256、六组配置、数据列表与文件内容。原 Results、checkpoint、summary、实验配置、快照、数据及划分文件均列入保护清单。
- 重点审读：`code/networks/vnet_ds.py`、`code/objectives.py`、`code/train.py`、`code/data.py`、`code/evaluate.py`、`tests/verify_core.py`，并核对六组 Results 配置、保存的源码和本地 DuCiSC 参考。
- 在线核对的上游 main：[`cc77a8308d0aa1c59a7c29556478bfd33d804a1a`](https://github.com/ShanghaiTech-IMPACT/DuCiSC/commit/cc77a8308d0aa1c59a7c29556478bfd33d804a1a)。[VNet.py](https://github.com/ShanghaiTech-IMPACT/DuCiSC/blob/cc77a8308d0aa1c59a7c29556478bfd33d804a1a/code/networks/VNet.py) 的 Git blob SHA 为 `6918ef43514e3b1c5a57ac1584240f41fbe9dc57`。在线文本与本地上游原文件相同，SHA256 为 `3875141b68fd9823798f54bbb50bf99031ec075f852d73feb0f15bba57feb93a`；参考副本仅增加末尾换行，SHA256 为 `0600d4d0ca7a226f84829820f3ef35bf3f4dbec4a6c269f58f602c725e126a55`。上游既有行尾空白保留。

## 审计发现与处理决定

| 发现 | 对解释实验的影响 | 阶段 0 处理 |
| --- | --- | --- |
| LA、Pancreas 以 `test.list` 在线选最佳 Student，再在同一列表报告最终测试 | 这些指标使用了测试集选模，不能解释为独立留出测试结果；BraTS19 用独立 val/test | 保留协议，明确记录，不修改划分或选模 |
| warm-up 期间 Student 无标签前向、Teacher 前向、BN 更新、Dropout 及 EMA 均继续 | 2000 步 warm-up 仅关闭一致性项的梯度贡献，不是冻结 Teacher 或跳过无标签分支 | 保留调用顺序及行为 |
| LossNorm 的数值通常恒为 1，组合通常为 1 或 1.3 | 数值不反映原始目标下降；梯度仍随原损失及其倒数缩放，须看原损失与梯度日志 | 保留每目标归一化及 eps |
| Teacher BN 不参与 EMA，Dropout 保持 train | Teacher 统计由自身无标签前向更新，与 Student 统计不同 | 保留参数 EMA 与独立统计 |
| 上游 `valid_feat[0]` 是 `x8_up`，本次 D1 明确为 `x9` | 二者都是 16 通道、全分辨率，但位置和数值不同；D1 应是最终 head 的真实输入 | 接口选择 Dropout 后的 x9，不冒充原 valid_feat[0] |
| 源码恢复了模型、SGD 和 RNG，但没有恢复 DataLoader 采样位置/worker 状态 | resume 不能保证逐 batch 位级重现 | 保留恢复语义，接口测试从同一初始状态独立比较 |
| 现有 `verify_core.py` 未覆盖有 Dropout 时新接口开关、九阶段张量与 BN/RNG | 单靠原测试不足以证明新增接口兼容 | 增加独立测试，不改变原测试 |

这些项目中，有的是既定协议或实现选择，不能作为阶段 0 擅自修正 Baseline 的理由。本次不改变训练损失、采样、归一化、warm-up、EMA、优化器、验证或推理。

## 网络真实路径

输入为 `[B,1,H,W,D]`，默认基础通道 `n_filters=16`，三个空间维应为 16 的正整数倍。

编码路径：`block_one → down → block_two → down → block_three → down → block_four → down → block_five → 可选 dropout`，得到 `[x1,x2,x3,x4,x5]`。x1–x4 同时供跳连；E5 指编码器实际返回的 x5，`has_dropout=True` 时为 Dropout 后张量。

解码路径：

1. `block_five_up(x5) + x4 → x5_up → out_conv_4`，然后进入 block_six。
2. `block_six_up(block_six(x5_up)) + x3 → x6_up → out_conv_3`。
3. `block_seven_up(block_seven(x6_up)) + x2 → x7_up → out_conv_2`。
4. `block_eight_up(block_eight(x7_up)) + x1 → x8_up → block_nine → 可选 dropout → x9 → out_conv`。

四尺度 logits 始终按 `[D4,D3,D2,D1]` 排列，对应空间比例 `[1/8,1/4,1/2,1]`。分割 backbone、head 参数键与上游 DuCiSC 对应；Baseline 移除了原型分支输出及其他未使用类。

上游 decoder 的 `valid_feat` 是 `[x8_up,x7_up,x6_up,x5_up]`；其 `forward(deepsup=True)` 返回 `(out,valid_feat)`。本地 VNetDS 的既定接口则返回四元素 logits list，默认仅返回最终 logits。新增可选特征接口以 **本地既定行为** 为兼容基准。

## 损失与训练时序

监督：每尺度用 nearest 下采样二值 GT；CE 使用 PyTorch 默认 mean。前景 softmax p 的 Dice 为 `1-(2*sum(p*y)+1e-8)/(sum(p)+sum(y)+1e-8)`，对整个 batch 和空间求和，不是逐病例 Dice 后平均。四尺度 CE 等权平均、Dice 等权平均，最后 `Lsup=(CE+Dice)/2`。此公式与本地 DuCiSC `get_loss` 和 `Binary_dice_loss` 对应。

一致性：各尺度 `MSE(softmax(student),softmax(teacher.detach()))`，对 batch、类别、空间做 mean，再对四尺度等权平均。它是概率 MSE；与上游完整 DuCiSC 的硬伪标签 CE+Dice、prototype、mix 和不确定性筛选不是同一种完整训练目标。

`normalize_loss(L,eps)=L/stop_gradient(max(L,eps))`。在 `L>eps` 时前向值为 1，但梯度为 `dL/L`；在分母被截断时为 `dL/eps`。detach 只作用于分母与 Teacher 目标，不能移到特征接口。极小一致性损失可能放大其梯度，这是既有 LossNorm 语义。

每步：Student 有标签前向 → Student 无标签前向 → Teacher 无标签前向（no_grad）→ 计算损失 → backward → 记录梯度范数 → SGD.step → 参数 EMA → iteration 加一。warm-up 判断在加一之前：计数 0–1999 的前 2000 次更新权重为 0，第 2001 次更新起一致性权重为 0.3。warm-up 中仍计算并检查一致性数值，仍消耗原有 Dropout RNG。

Student 和 Teacher 都在 train 模式。Teacher 从 Student deepcopy 并 `requires_grad_(False)`；Teacher no_grad 不会关闭 Dropout 或 BN buffer 更新。Student BN 每步更新两次，Teacher 每步一次。Dropout3d(p=0.5,inplace=False) 每次前向在 x5、x9 各调用一次，各次前向按顺序独立使用 RNG。

优化器为 SGD，lr=0.01、momentum=0.9、weight_decay=1e-4，学习率恒定。`clip_grad_norm_(...,inf,error_if_nonfinite=True)` 用于观察范数，不作有限阈值裁剪。EMA 系数 `min(1-1/(global_step+1),0.99)`，第一个 step 系数为 0，完全复制更新后的 Student 参数；只迭代 parameters，不复制 BN buffers。

## 六组完成实验与数据协议

共同配置：seed=42、15000 次更新、2000 warm-up、batch=4（2 标注+2 无标注）、单通道输入与二分类输出、三维 BN、Dropout、每目标 LossNorm eps=1e-8。

| 数据集 | 标注数（10% / 20%） | train / val / test | Patch | 在线选模 |
| --- | --- | --- | --- | --- |
| LA | 8 / 16 | 80 / 无 / 20 | 112×112×80 | test |
| Pancreas | 6 / 12 | 62 / 无 / 20 | 96×96×96 | test |
| BraTS19 | 25 / 50 | 250 / 25 / 60 | 96×96×96 | val |

标注病例为 train.list 顺序的前 N 个，不是随机重新划分；20% 与 10% 标注集嵌套。TwoStreamBatchSampler 每轮打乱标注索引、丢弃不足整批的尾部，无标签流独立无限重排，batch 内标注在前。load_case 验证单通道 3D、有限值、二值标签。LA/BraTS19 在前两轴 rot90、flip，Pancreas 无此旋转；均使用原 padding 和随机裁块。无标签 GT 虽随数据读取，训练仅将标注子批的 GT 放入损失。

split_manifest 记录 ID、列表 SHA、标注/无标注 ID，检查 train 与 val/test 无交集以及文件存在；未强制 val/test 彼此无交集，因此额外只读核对当前列表。当前划分与六组保存 manifest 的对应关系由验证产物记录。

## Checkpoint、恢复与推理

- 从第 800 步起每 200 步验证 Student，只按最终全分辨率 head 的 Dice 选优。
- `best_model.pth` 为纯 Student state_dict；`teacher_at_best_student.pth` 保存同一步 Teacher；不按 Teacher 指标选优。
- `last_checkpoint.pth` 保存 Student、Teacher、SGD、iteration、best_dice、配置、Python/NumPy/Torch/CUDA RNG，在第 1 步、每 1000 步及最后一步写入。
- resume 校验关键配置和划分，默认 load_state_dict 为 strict=True；恢复采样轨迹的限制见上表。原路径中的 output_dir、provenance 保持原样，不能把复制出的历史结果误认为新实验。
- 训练结束严格加载最佳 Student，然后在 test.list 测试。本次只读加载已有 checkpoint，不执行 train() 或写回任何实验目录。
- evaluate 用 eval + no_grad，退出时恢复原 training 状态。predict_volume 使用默认 forward，滑窗累加前景概率并平均后以 `>0.5` 二值化，补零并裁回原尺寸，无最大连通域后处理。
- LA stride=(18,18,4)，其余=(16,16,16)。Dice/JC 按病例平均；空预测或空 GT 的 HD95/ASD 记 null，报告可定义病例数；表面距离为体素单位，未传体素间距。

## 阶段 0 实施边界

仅在当前 VNetDS 增加可选返回已有张量的路径，新增无参数的安全访问辅助函数与测试、报告。E5/D1 采用真实 head 路径上的 Dropout 后张量；未来若需 Dropout 前边界监督，应另行定义阶段和实验语义。当前不添加边界头、边界目标、通道筛选、交换、交互或任何训练项。

兼容性和实测结论见 `VERIFICATION.md`，不能从本审计的静态阅读直接推断测试已经通过。
