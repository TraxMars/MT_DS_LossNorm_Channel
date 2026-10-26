# 阶段 1A 验证与交付记录

日期：2026-10-21（Asia/Shanghai）。开发分支：`experiment/stage1a-gradient-diagnostics`；基于阶段 0 提交 `44873676c3faca78abcd02eac35dfca21910592c`。

## 实际执行

解释器：`/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python -B`。环境沿用 PyTorch 2.9.1+cu126、NumPy 1.26.4、h5py 3.16.0；新增统计使用已安装的 scikit-learn 1.7.2、Matplotlib 3.10.9，未安装额外依赖。

| 命令 | 结果 |
| --- | --- |
| `tests/test_stage1a_diagnostics.py` | **10 tests，0 failures，0 errors；0.946 秒**，记录于 TEST_RESULTS.json |
| `analysis/stage1a_gradients/run.py --device cuda:1 --output analysis/stage1a_gradients/runs/la10_iter15000_seed142_fp32 --seed 142 --unlabeled-cases 16 --repeats 2` | GPU A100 40GB，48 个完整真实 Patch；PASS；44.873 秒；峰值已分配 1118.467 MiB |
| `analysis/stage1a_gradients/summarize.py --run analysis/stage1a_gradients/runs/la10_iter15000_seed142_fp32` | PASS；69.471 秒；36 格矩阵、2944 个通道×尺度记录；200 次病例 bootstrap、100 次分组 bootstrap |
| `analysis/stage1a_gradients/verify_outputs.py --run analysis/stage1a_gradients/runs/la10_iter15000_seed142_fp32` | PASS；141312 条范数记录、354304 条病例对记录、24 个独立重算余弦与 398 个保护哈希通过；实测结果见 output_verification.json |

没有启动原 `train.py`、launcher、optimizer、EMA 更新或任何训练循环；optimizer_instances=0、optimizer_steps=0、checkpoint_writes=0。

## 必须项的验证证据

| 要求 | 验证方式与结果 |
| --- | --- |
| 阶段参数映射正确 | 按模块类型查找；检查参数对象身份、实际 feature 通道；CPU 验证精确名称和 shape；无 BN 的变体证明不是复制固定索引 |
| ConvTranspose3d 通道轴正确 | CPU 构造已知数列梯度，逐输出通道核对 axis=1 展平；拒绝 axis=0 的普通卷积展平 |
| 保留原四尺度公式 | 每头直接调用原 supervised_loss/consistency_loss；四尺度均值对照原总目标；实际最大标量差 3.259629011e-9 |
| 梯度 shape 和图可达性 | 每病例×每尺度验证参数 shape，None 与预期 6 个不可达格完全对应；原始数组按 observation×head×output-channel×kernel 保存 |
| 使用 autograd.grad 且不污染 .grad | 正式诊断所有 Student .grad 始终 None；CPU 另预置非零 .grad，证明函数不覆盖已有值 |
| Teacher 不接收梯度 | requires_grad=False、no_grad、targets.requires_grad=False，前后所有 Teacher .grad=None |
| 无标签 GT 不参与 | 无标签专用 loader 无 label 查询；CPU 使用无 label 数据集并拦截 HDF5 非 image 键；实际 16 次无标签访问全部为 image |
| 参数、BN、RNG 不变 | 逐项比较两模型完整 state_dict，Student/Teacher eval 标志及 Torch/CUDA RNG；全部不变 |
| Checkpoint、原代码、历史结果保护 | 提取前后及输出核验重新校验 398 个既有文件，涵盖两目录代码、六组历史 Results/summary/checkpoint/配置/快照和划分列表 |
| 不因保护哈希读取无标签 GT | 不重新哈希 HDF5 容器内容；所有实际 HDF5 只读打开，记录 24 个访问数据文件的 size/mtime 前后不变；不把字节哈希行为混入 image-only 规则 |
| 无效梯度不生成分数 | 显式分 graph_unreachable、exact_zero、below_threshold、nonfinite；合成负余弦/零/微小/NaN 测试；无效值留空/NaN |
| 区分 raw 与 LossNorm | 记录 g_s、g_s/4 与 g_s/[4·max(total,eps)]（MT 含 0.3）；实际重构原总梯度和归一化总梯度 |
| 参数梯度不冒充特征梯度 | 输出对象是卷积权重导数；从未把不同病例特征图空间坐标作逐点匹配 |
| 不从低秩推断协同 | 合成测试证明 rank=1 可同时出现完全对齐与完全冲突；记录 simplex 代数秩上限、D1 常数特征和聚类大小/ARI |

本次重构最大相对 L2 误差：原始总梯度 **5.988096670e-6**，LossNorm 总梯度 **5.429906685e-6**；均小于预定 1e-4（或绝对 L2≤1e-9）阈值。最大单系数绝对差分别为 9.639188647e-8、1.996755600e-6。各值保存在 gradient_reconstruction.json，而非声称浮点反传重排位级相同。

**有效性需区分统计层次。**病例均值逐通道表中，2768 个可达通道×尺度条目均达到有效病例对覆盖标准，另 176 个结构不可达条目。逐裁块范数表则包含 132839 个有效观测、8448 个结构不可达观测和 **25 个低于阈值观测**；这 25 个不生成有效相似度，不能称作全部原始观测有效。当前无精确零或非有限观测；合成测试覆盖这些异常。全部 36 格中 30 格有有效梯度，也不表示其中每个裁块/通道都有效。

## 首次未通过的尝试

`runs/la10_iter15000_seed142/` 的 extraction_results.json 保留 FAIL 和完整 traceback。首次在首个有标签裁块检查“独立头梯度均值”与“总目标一次反传”时退出；逐系数阈值 rtol=2e-5、atol=2e-7 下最大绝对差 1.1632219e-6，未完成分析。

之后在新目录明确禁用诊断 TF32，并用全向量相对 L2 误差处理近零系数相对误差失真的问题。两项调整都只影响诊断精度策略与校验，不修改原模型、损失公式或训练代码；没有独立隔离证明首次误差仅由 TF32 造成，故不作这一因果归因。最终成功结果统一在 `_fp32` 目录，首次尝试不计入通过数量。

## 修改范围与交付

原 `code/`、阶段 0 接口、tests/verify_core.py、训练/评估代码、原配置与历史实验均未修改。新增：

- `analysis/stage1a_gradients/core.py`：参数映射、独立原损失、autograd.grad、image-only 数据读取、有效性规则。
- `analysis/stage1a_gradients/run.py`：冻结模型提取与状态/历史文件保护。
- `analysis/stage1a_gradients/stats_tools.py`、`summarize.py`：病例关系、多尺度方向 SVD、范数占比 PCA/聚类及 PNG/PDF。
- `analysis/stage1a_gradients/verify_outputs.py`：保存产物的独立重算与保护复核。
- `tests/test_stage1a_diagnostics.py`：10 项诊断测试。
- `analysis/stage1a_gradients/README.md`、`FINDINGS.md`、`VERIFICATION.md`、`SOURCES.json`、`TEST_RESULTS.json`、脚本指纹与运行产物。

原始 NPY（约 1.9 GiB/次）、逐裁块展开表和逐病例对展开表留在本地独立目录，并在局部 .gitignore 排除，避免把大体积分析缓存混入源码提交。紧凑 CSV/JSON、报告与图表版本化。

主报告给出普通分割/MT 梯度不能直接称为边界梯度的结论，并提出仅有标签 GT 的边界带、距离场与匹配负对照设计。未实施边界目标、训练或双向交互下一阶段。
