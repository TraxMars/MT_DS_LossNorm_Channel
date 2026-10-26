# 阶段 0 实际验证结果

日期：2026-10-21（Asia/Shanghai）。工作目录：`/aita_data/home/aita_duanxuelong/MT_DS_LossNorm_Boundary`。

## 结论

**已满足在 E1–E5、D4–D1 九层接入独立边界监督所需的特征访问和可微分条件。**
九层都是当前同一次前向中产生的真实张量，未 detach、未重算，未新增模型参数、卷积层或 BN。
本次仅完成阶段 0，未实现边界损失或边界头，未启动正式训练，也未实施下一阶段。

原 Baseline 兼容性成立：默认/深监督旧接口、state_dict 和旧 checkpoint 兼容；原训练、优化器、LossNorm、warm-up、EMA、数据与评估语义保持。
CPU 上所有数值比较为零容差。GPU 的实际观察中，分割输出、原始监督损失 Lsup、一致性损失 Lmt、组合损失、原始及组合参数梯度、RNG 与 BN 也均精确一致；**单独的 CE 统计量有原生 CUDA reduction 的浮点非确定性**，不能承诺任意 CUDA 环境下所有标量位级一致。

## 基准与保护

- 分支：`experiment/stage0-feature-interface`。
- 修改前本地 main：`28ae37d1564c53f46b65c92263898a44b0308cc0`。
- 该 main 是新目录已有完整副本的本地文件系统快照；两个源目录原先没有可用 Git 历史，未宣称它是某个原历史提交。
- 修改前网络文件 SHA256：`57f29fc308d484c09cd2f080c5e283aea7a59e99eb4b0bf5612a25bd0e1e41b2`。
- 修改前清单包含 3020 个文件、5,852,632,015 字节，涵盖原 MT_DS_LossNorm、新目录既有副本以及 Dataset；唯一允许改变的原文件为新目录 `code/networks/vnet_ds.py`。
- 实际重新校验 **3019 个受保护文件通过，无缺失、内容变化或受保护树新增文件**。包括两处六组 Results、所有已有 checkpoint、summary、实验配置、源码快照、数据和划分。
- 六组实验均 completed、15000 iterations；原配置、数据列表、标注前缀、列表互斥和保存源码与修改前 active code 的一致性均通过。历史日志第 1、2000、2010 步的一致性权重分别为 0、0、0.3。
- `code/train.py`、`objectives.py`、`data.py`、`evaluate.py`、原 `tests/verify_core.py` 均未改动。

完整哈希、实验配置和只读核验记录见 `BASELINE_MANIFEST.json`、`EXPERIMENT_CONFIGS.json`、`INTEGRITY_RESULTS.json`。

## 环境

Python 3.10.22；PyTorch 2.9.1+cu126；NumPy 1.26.4；h5py 3.16.0；SciPy 1.11.4；MedPy 0.5.2。CPU 测试使用 2 threads；测试框架为标准库 unittest，未安装额外依赖。

沙箱内最初无法访问 NVIDIA 驱动，宿主机 GPU 检查成功后，在宿主机 GPU 4（NVIDIA A100-SXM4-40GB）完成真实数据验证。
原训练的 CuDNN 设置保留为 deterministic=True、benchmark=False。

## 自动化测试与实际执行

在项目目录使用 `/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python -B` 执行：

| 命令后缀 | 实际结果 |
| --- | --- |
| `tests/test_stage_features.py --report analysis/stage_features/CPU_RESULTS.json` | **7 tests，0 failures，0 errors，0 skipped，11.617 秒** |
| `tests/verify_core.py --original analysis/stage_features/reference/upstream_vnet.py` | 所有原核心检查通过，exit code 0 |
| `tests/verify_stage0_integrity.py` | 3019 个受保护文件通过；6 组配置/快照/划分/历史 warm-up 记录通过；14.407 秒 |
| `tests/test_stage_features.py --gpu-smoke --gpu-index 4 --report analysis/stage_features/GPU_RESULTS.json` | LA、Pancreas、BraTS19 各 1 个真实完整 Patch batch；各自 train/eval 均通过；总计 64.324 秒 |
| `git diff --check` | 已跟踪源码改动通过；新增文件的最终 staged 检查排除原样保存的 reference 快照后通过 |

原 Baseline 目录的未修改 `tests/verify_core.py` 也曾以 `-B` 只读运行，所有检查通过。新目录的原核心检查涵盖上游预测/参数、Teacher 梯度隔离、四头梯度、EMA 首步、LossNorm 与滑窗 padding/overlap/恢复尺寸。

完整 staged whitespace 检查曾指出 `reference/upstream_vnet.py` 中上游既有的十处行尾空白。为保留参考源码未清理它们；排除 reference 快照后，新增实现、测试和报告的 staged 检查通过。

CPU 七个测试的覆盖：

1. 修改前网络快照 SHA256、与旧模型及上游的 state_dict 键/形状/dtype、注册模块一致。共 **211 个 state_dict 条目、124 个参数张量、9,449,320 个参数**。
2. train/eval × Dropout 开/关，比较修改前、修改后四种 forward 组合和上游 DuCiSC；四尺度输出、默认最终输出、RNG、BN 均使用 `atol=rtol=0`。
3. 九层 shape、requires_grad、与编码器真实输出/分割头真实输入的对象身份；encoder/decoder 每次只运行一次，各叶模块一次，复用的 Dropout 恰好两次。每层独立 `autograd.grad(stage.sum(), producing_weight)` 均有限且非零，原监督 backward 也到达全部九层。
4. 原三次训练前向结构（Student 标注、Student 无标注、Teacher 无标注），train/eval × warm-up 权重 0/0.3；分别比较原始监督梯度、原始一致性梯度及 LossNorm 组合梯度。特意使用不同 Teacher bias，eval 中一致性损失与梯度也为非零。损失、梯度、RNG、BN 均为零容差。
5. 六组 `best_model.pth` 逐个在旧/新模型中 `strict=True` 加载；missing/unexpected keys 均为空；每组四尺度推理结果精确一致。
6. Teacher no_grad、train-mode Dropout 和独立 BN；首步/后续参数 EMA、EMA 不覆盖 BN buffers；LossNorm 正常、eps 以下、零值的梯度行为。
7. 安全辅助函数返回原张量、实际通道和空间尺寸、不消耗 RNG；错误阶段、缺失分组、非 Tensor 或错误维数等拒绝行为。

CPU 使用 B=2、32³ 验证九阶段；B=4、16³ 验证与训练一致的 2+2 分支损失和梯度；还验证了非立方 16×32×16 输入。每阶段通道、空间尺寸、requires_grad 和原模块已记录在 `CPU_RESULTS.json`；人类可读对应表见 `README.md`。

## GPU 真实数据 Smoke Test

每数据集读取 train.list 中两个标注前缀病例、两个无标签病例，按原数据读取和裁块流程构成一个 batch；仅前两个病例 GT 进入监督损失。严格加载该数据集原 10% 最佳 Student 及同一步 Teacher，分别比较原 Baseline 自身重复运行、修改后默认接口、修改后特征接口。

| 数据集 | Batch | 模式 | 峰值已分配显存 MiB | CE 最大绝对差（接口/旧模型） | Lsup、Lmt、组合损失和全部参数梯度最大差 |
| --- | --- | --- | --- | --- | --- |
| LA | 4×1×112×112×80 | train / eval | 3981.139 | train 9.313225746e-10，eval 0 | 0 |
| Pancreas | 4×1×96×96×96 | train / eval | 5154.584 | train 3.725290298e-9，eval 0 | 0 |
| BraTS19 | 4×1×96×96×96 | train / eval | 4978.128 | 0 | 0 |

所有比较的 Student/Teacher 四尺度输出、原始监督/一致性参数梯度、RNG、BN buffers 均由零容差断言通过。组合梯度允许容差的比较中，实际观测差也全部为 0。

GPU 标量和组合梯度的显式测试阈值为 rtol=1e-6、atol=1e-7，实际差逐项保存在 JSON；不能把“在阈值内”泛称为全部位级一致。
本次非零差仅出现在单独 CE 统计量，Pancreas 未修改 Baseline 自身重复运行也出现 `3.725290298e-9` 的同样差异。监督损失 `(CE+Dice)/2` 在本次浮点舍入后相同，因此 Lsup 与组合梯度本次均精确相同。这不是接口改变了损失计算。

九层真实 Patch shape、通道、requires_grad、原模块及病例 ID 均记录于 `GPU_RESULTS.json`。三个数据集的全部九层可训练 Student 特征均 requires_grad=True。所有 smoke 均 **optimizer_steps=0、checkpoint_writes=0**。

初次 GPU 尝试未伪记为通过：

1. 全局 `torch.use_deterministic_algorithms(True)` 遇到原 CUDA NLL/CE 未实现严格确定性而退出。后续采用原 train.py 设置，未改生产损失。
2. 零容差比较单独 CE 标量失败，差值 `9.313225746154785e-10`。后续加入未修改 Baseline 重复运行对照，保留网络/RNG/BN/原始梯度精确检查，只对已说明的 CUDA 浮点项显式使用阈值。

初次记录见 `GPU_ATTEMPTS.json`；最终实际通过结果见 `GPU_RESULTS.json`。未启动迭代训练、验证集遍历或新实验。

## 修改文件清单

唯一修改的既有源码：

- `code/networks/vnet_ds.py`：decoder/forward 增加 return_features=False 可选参数，返回现有九阶段张量，保留默认路径及四尺度顺序。

新增生产辅助模块（无参数、无损失）：

- `code/boundary/__init__.py`
- `code/boundary/features.py`

新增自动验证：

- `tests/test_stage_features.py`
- `tests/verify_stage0_integrity.py`

新增审计与验证产物，均位于 `analysis/stage_features/`：

- `README.md`、`AUDIT.md`、`VERIFICATION.md`
- `BASELINE_MANIFEST.json`、`EXPERIMENT_CONFIGS.json`
- `CPU_RESULTS.json`、`GPU_RESULTS.json`、`GPU_ATTEMPTS.json`、`INTEGRITY_RESULTS.json`
- `reference/vnet_ds_baseline.py`、`reference/upstream_vnet.py`

除此之外没有修改原 Baseline 的源码、配置、实验产物、数据或划分。

## 后续工作的条件与限制

已具备九层独立边界监督的接口基础：层名/位置明确、通道/尺寸可安全查询、特征保持梯度、已有训练结果与模型加载兼容。E5、D1 明确采用原 Dropout 后张量；若后续想监督 Dropout 前张量，需另行定义并验证语义。

LA/Pancreas 既有测试集选模协议、LossNorm 的梯度缩放及 resume 不逐批复现等限制已在 AUDIT.md 记录并保留。本次未验证新增边界方法的有效性、显存增量或最终性能，也没有实现下一阶段。
