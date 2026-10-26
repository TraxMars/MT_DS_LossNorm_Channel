# Stage1C C0：实现和产物入口

本次只完成C0，最终判定PASS。没有C1/C2/C3，没有正式训练、通道分组或双向交换。

- [预先固定设计](DESIGN.md)：病例、节点、剂量、数值区间和C0门槛。
- [测试与历史完整性报告](TEST_REPORT.md)：21项C0测试、31项历史回归、原core检查及失败修正记录。
- [最终真实数据smoke报告](SMOKE_REPORT.md)：实际结果、方向不可判定占比、原始损失和局部近似范围。
- [远程/本地核对](REMOTE_AUDIT.json)：GitHub共有290文件一致，本地额外保留22图件；Stage1B原始manifest验证210文件。
- [最终run验证](runs/c0_la10_checked/verification_results.json)：711个历史文件SHA及独立归因重算。

## 新增文件

| 文件 | 作用 |
| --- | --- |
| support.py | 只读A/B桥接、固定病例/剂量、输出保护、数值区间和GxA统计 |
| pairs.py | native6邻接有向位置对、Teacher候选、坐标/粗尺度安全检查及CPU置换辅助 |
| objectives_c0.py | 相互独立的最终D1损失；拒绝U GT和多头列表输入 |
| nodes.py | 五阶段同一节点采集/单通道削弱；实际张量差L2与舍入量测量 |
| smoke.py | 固定2L+2U、原crop、三剂量真实GPU诊断，无optimizer |
| summarize.py | 仅C0数值保真、方向不确定性及预定门槛汇总 |
| verify.py | 独立数组内积/参数范数/位置对/阈值/历史SHA验证 |
| regressions.py | 导入旧suite后运行，报告只写C0 |
| report.py | 生成本目录测试报告、smoke报告及3组PNG/PDF |
| ../../tests/test_stage1c_c0.py | 21项CPU合同测试 |
| PROTECTION_BEFORE.json | 开始前711个已有文件SHA256 |
| REMOTE_* | 远程版本与Git blob清单、完工复核；不发布远程更改 |
| EXECUTION_NOTES.json、TEST_ATTEMPT_* | 保留失败/夹具修正/量测改进记录 |
| ARTIFACT_MANIFEST.json | 最终C0源码和所有本地产物SHA，包括Git忽略的raw，排除自身 |

所有旧生产代码、Stage0/1A/1B、Results、checkpoint、summary和划分文件未修改。

## 最终run数据

`runs/c0_la10_checked/` 是最终结果；`runs/c0_la10_fixed/` 保留初版公式L2量测记录，不作为最终交付run。REPLAY_OF_MEASUREMENT_REFINEMENT.json确认两次run的1440个Delta_hat/actual/epsilon/alpha/预算数值完全一致。

- geometry.json：8个native pair集，数量/类别/边缘/概率关系/覆盖。
- node_identity.json：40个stage/crop真实对象、模块位置、skip等式。
- noise_calibration.json：32个目标/crop各20个零剂量null损失和冻结epsilon。
- deltas.csv：1440个目标差分，预测/实际符号、epsilon、alpha、预算、实际L2和舍入量。
- intervention_metrics.csv：360个单通道干预的最终预测指标；U没有GT准确性指标。
- feature_gradient_stats.csv、parameter_gradient_stats.csv：各9728行，坐标含义分开。
- raw/features/*.npz：40组主目标完整activation/gradient/M。
- raw/parameters/*.npy：160组独立目标卷积参数梯度向量，输出通道轴沿用A/B。
- raw/*pairs.npz：8组native整数坐标。
- reference_replay.json：80个A/B原最终头参数梯度重放。
- AB_compatibility.json：原四头loss重放，以及同公式FP64/原FP32的舍入差。
- fidelity_by_stage_dose.csv、fidelity_gates.json：所有目标的局部保真与预定C0门槛。
- source_at_run.json：实际执行源文件、测试及DESIGN在非零干预前的SHA。

raw仅保留在本地独立目录，不提交到Git；没有压缩/删除历史raw。符号判定只用冻结epsilon。Spearman基于有限数值观测，含方向不可判定观测；符号准确率另排除实际不可判定者，预定Taylor门槛另用共同可辨识集。不把rho解释为弱信号有可靠方向或边界语义。

## 复现命令

在本仓库根目录运行，使用新的不存在的output路径；不覆盖已有run。report.py会生成本C0目录的报告，复核历史报告时应在独立工作副本运行。

```bash
PY=/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python
$PY -B tests/test_stage1c_c0.py
$PY -B analysis/stage1c_boundary_attribution/regressions.py
$PY -B analysis/stage1c_boundary_attribution/smoke.py --device cuda:1 --output analysis/stage1c_boundary_attribution/runs/new_c0
$PY -B analysis/stage1c_boundary_attribution/summarize.py --run analysis/stage1c_boundary_attribution/runs/new_c0
$PY -B analysis/stage1c_boundary_attribution/verify.py --run analysis/stage1c_boundary_attribution/runs/new_c0
```

固定模型、同目标的小扰动Taylor准确性属于数值合同；C0不回答边界专属性、Teacher真实边界可靠性、发现集通道迁移或通道协同问题。
