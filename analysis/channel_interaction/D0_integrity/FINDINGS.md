# D0 完整性验证 FINDINGS

状态：PASS。LA 10% 与 Pancreas 10% 均通过，可继续 D1-A 的冻结与不更新参数的梯度诊断。该结论证明输入归属及历史保护，不证明任何跨数据集梯度现象。

| 数据集 | L/U | Patch | Iteration | 配对 checkpoint SHA256 |
| --- | --- | --- | --- | --- |
| LA | 8/72 | 112×112×80 | 15000 | `85489c52c0905aebde65eacf5cea215aac51ee825ed3e5a845da86fe7f23e168` |
| Pancreas | 6/56 | 96×96×96 | 15000 | `55e7b3c4837bc993daa0acd6c23e9a56a3461d08b2be1219c003f3c6376b7835` |

每套同一序列化 checkpoint 内的 Student/Teacher 分别严格加载，211 个状态键及所有 shape、数值有限性通过；与原 MT_DS_LossNorm checkpoint 的文件 SHA 和逐状态摘要一致。Student/Teacher 状态不相同，两数据集的 Student 摘要也不相同，没有混用权重。摘要、全键 shape、划分 SHA 和病例 ID 均保存于 results/。

原项目、复制项目、Boundary 历史共 3063 个文件完成前后 SHA 核查，无变化或新增历史文件；Stage1A/C0 的 573 个复制文件与 Boundary 原件一致，包括原始数组。数据 HDF5 不作整文件哈希，避免读取 U GT；划分 list 全量哈希，病例文件只检查存在性、大小及 mtime。LA 100 条、Pancreas 82 条分割清单记录均存在（train/test 记录，含原验证设计）。

## 源代码和旧结果

data.py、objectives.py、train.py、evaluate.py 与原 Baseline SHA 一致。vnet_ds.py 存在复制项目原有 return_features 接口扩展；差异完整保存在 results/round1/vnet_ds.py.diff，未改变预测路径或 state_dict schema。7 项人工张量特征接口回归通过，验证原预测、梯度和 BN/RNG 等价。

进入任务时已有未跟踪的 Stage1C-C1 目录和 tests/test_stage1c_c1.py；未修改或清理。D0 快照中的 git status 是新目录创建后的状态，不被当作进入任务前工作区干净的证明。新增实验约束记录在项目 AGENTS.md。

## 数据与 U GT 审计

原 VolumeDataset.__getitem__ 无论 L/U 都调用 load_case，实际读取 image 和 label。旧训练只切片 L label 进入损失，但诊断不能复用这条读取路径。新诊断明确使用 Stage1A read_image_only；U label 缺失或访问被拦截的人工测试通过。保留原 loader，不修改 Baseline。两套 train/val/test list 的 SHA、ID、L/U 数量、train 与评估集无交集检查通过。

## 相同点、差异、局限和继续条件

两套完整性结论相同：同迭代配对、权重完整、划分未变。差异是各自训练权重、ID、L/U 数量和 patch，不共享病例或权重。LA 与 Pancreas 都满足进入 D1-A 的工程条件；各自梯度现象必须由真实诊断分别判断。

本机 CUDA 驱动不可用，使用 CPU 保留原 patch；原 Baseline test.list 用于在线选模，不能当作独立留出集。本阶段不读 HDF5 内容，只核验文件与划分归属，因此不声称证明所有病例图像的语义正确性。

1 项新增历史保护单元测试与 7 项特征接口回归通过。补充检查首次直接比较 tuple/list 触发实现层失败，失败记录保留于 round1_supplement/failure.json；JSON 序列化统一后配置完全一致，完成结果在 round1_supplement_checked/，未放宽数据内容要求。最终 D1 完成后另做全量历史 SHA 复查。

## D1 完成后的历史复查

原有 3,063 个受保护文件 SHA 全部一致，changed_or_missing 为空。独立 Boundary 项目运行期间新增 99 个 C15 文件，触发严格目录无新增检查，故 round1_after_D1/protection_after.json 保留 FAIL；concurrency_review.json 另记录 PASS_EXISTING_FILES_WITH_EXTERNAL_ADDITIONS。没有改写失败记录、删除并发文件或将 C15 用作本轮输入。用户要求的历史产物不覆盖检查通过，目录完全静止这一更强条件不成立。
