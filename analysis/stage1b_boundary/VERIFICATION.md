# Stage 1B：执行与验证

最终结果：PASS。这里区分已执行检查、失败记录和未执行项目，不把诊断当作新模型训练。

## 实际执行

解释器为 `/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python`；版本见[ENVIRONMENT.json](ENVIRONMENT.json)。真实数据GPU诊断使用主机可见设备1。所有输出位于新增Stage1B目录。

| 检查/运行 | 实际结果 |
| --- | --- |
| `tests/test_stage1b_boundary.py` | 14/14 PASS；[JSON](TEST_RESULTS.json) |
| 原 `tests/test_stage_features.py` CPU回归 | 7/7 PASS，10.616s；包含train/eval输出、梯度、RNG/BN、九层不重复计算、六组最佳Student strict加载；新报告[STAGE0_REGRESSION](STAGE0_REGRESSION.json) |
| 原 `tests/test_stage1a_diagnostics.py` 导入后运行suite | 10/10 PASS，0.921s；用新路径保存报告，未执行其会覆盖旧报告的main；[STAGE1A_REGRESSION](STAGE1A_REGRESSION.json) |
| 原 `tests/verify_core.py` | exit0；上游四尺度输出/state keys、Teacher梯度隔离、原目标/EMA、LossNorm与滑窗协议全部PASS；[记录](CORE_REGRESSION.json) |
| 首轮 `runs/smoke` | **FAIL**：固定crop元数据中tuple与A JSON list的类型差异；尚未运行模型前向。不删除失败记录 |
| 修复元数据JSON规范化后 `runs/smoke_checked` | 1L+1U真实crop，52目标×尺度PASS；21.762s，峰值1959.6MiB |
| smoke中的A梯度重放 | 60个可达domain/stage/head向量均与原A完全相同，相对L2最大0；[A_replay](runs/smoke_checked/A_replay.json) |
| 完整梯度 `runs/la10_paired` | 48crop、1088目标×尺度PASS；171.293s，峰值2102.9MiB；每crop的原A四头损失均与旧JSON相符 |
| CPU配对分析 | 612个method/stage/head单元、47056逐通道关系、80个逻辑组；282.626s；A原始梯度直接只读复用 |
| 逐通道跨尺度补充 | 5152行主宽度/A/负对照channel记录，52组原损失与固定A分母统计 |
| `runs/ablation_smoke` | 1个probe L crop/E3，9条singleton+6条pair记录PASS；4.375s，峰值1503.3MiB |
| 完整 `runs/la10_paired/ablations` | 4个probe L病例×2crop，7296 singleton/1920 pair记录PASS；272.975s，峰值1761.5MiB；B4 vs B1最大logit误差0 |
| 独立产物验证 | [verification_results.json](runs/la10_paired/verification_results.json) PASS；包括原始梯度形状/NaN掩码、独立重算4个L/U矩阵格、区域质量守恒、匹配计数、能量预算与历史保护 |

14项新增测试覆盖：正号平衡CE/池化质量守恒、原始GT边界先于crop、计数匹配与无放回、空边界/不足区域显式无效、三维差分单位、常Teacher无效、Teacher stop-gradient/置换权重直方图、真实参数映射及已有`.grad`保持、D3最终路径hook、线性logits与非线性CE非加性、L2匹配/alpha上限、共享病例bootstrap索引，pointwise头的远区干预不能改变边界CE，以及结果路径拒绝写入旧Stage1A/Results目录。

## 梯度与目标有效性

所有1088个目标/尺度观测区域覆盖有效，不等于所有通道梯度有效。逐观测范数记录为：

| 状态 | 数量 |
| --- | --- |
| valid | 752699 |
| graph unreachable | 47872 |
| exact zero | 7 |
| below threshold | 190 |
| nonfinite | 0 |

7个精确零发生于L/E5（5）与L/D4（2）；微小梯度主要位于E5及D4。沿用A绝对1e-12、相对1e-6阈值，原始范数仍保留，相似度不把它们当作可靠零分。

117个原始NPY按 `[observation,4heads,C,kernel]` 保存；30/36个stage/head可达，其余6个单元为NaN并标注结构原因。独立扫描确认可达且目标有效处有限、图不可达/目标无效处NaN。矩阵中每通道≥75%有效病例对方可作可靠通道，数值报告保留有效性矩阵。

1个singleton干预（probe病例`1GU15S0GJ6PFNARO469W`、crop0、E4、channel92、B区域）无法在alpha≤0.5下匹配目标能量，状态为`unmatchable_budget_exceeds_alpha_cap`，未生成伪零预测影响。所有联合组预算匹配，联合实际L2²等于两个matched singleton的L2²之和。所有5个详细阶段的干预均实际影响了最终预测，不是只影响辅助头。

## 冻结及数据访问

梯度提取和干预前后逐项精确检查Student/Teacher整个state_dict，包含参数与BN buffers；所有`.grad`槽保持None；Teacher frozen/no_grad；所有模块保持eval；CPU/CUDA RNG完全不变；无遗留hooks。两个模型均strict=True加载原配对checkpoint。没有optimizer，optimizer_steps=0、checkpoint_writes=0。

U loader只调用HDF5 `image`键，GT键没有打开；访问日志位于[梯度run](runs/la10_paired/hdf5_access_log.json)。消融仅加载probe L病例。完整原始GT只用于L的native边界定义；没有读U GT、没有从test.list选择或优化参数。

HDF5使用只读模式，逐已访问文件检查size/mtime保持不变；为避免读取U容器中GT字节，不给整个HDF5计算内容哈希。数据访问保障来自loader结构、测试和键日志，不能把“没有读取U GT”误写成已验证U真实轮廓准确性。

## 历史完整性与分支

分支 `experiment/stage1b-boundary-diagnostics`，起点 `67a0bf11ff4c889787a79962cf3f0b8a83804bb6`。Stage0提交为`44873676c3faca78abcd02eac35dfca21910592c`。

修改前[PROTECTION_BEFORE.json](PROTECTION_BEFORE.json)记录500个既有文件的路径/大小/SHA256，包含生产源码、Stage0/1A源码/报告/原始梯度、项目及原MT_DS_LossNorm历史结果/checkpoint/config/summary和数据划分。独立检查全部一致，changed=[]。checkpoint SHA256：`85489c52c0905aebde65eacf5cea215aac51ee825ed3e5a845da86fe7f23e168`。

所有改动仅新增Stage1B目录和一个新测试文件。没有修改`code/train.py`、`code/objectives.py`、VNetDS、九阶段接口、原训练/evaluate行为；原A文件、六组Results、原summary和checkpoint没有覆盖。新增原始梯度留在本地并排除Git，结果清单提供哈希。

## 尚未执行、不能声称已完成

没有训练或测试新模型、没有边界预测头/新训练损失/通道交换；没有U GT边界准确性验证、增强逆变换稳定性、跨checkpoint/标注seed重复、独立泛化验证、完整体积通道干预或多个扰动幅度的剂量实验。

仅使用已参与Baseline训练的L病例作探索性诊断。病例bootstrap以4个probe病例为重采样单位，crop/随机组不冒充更多病例；区间为多重探索性描述，无显著性或协同证明。
