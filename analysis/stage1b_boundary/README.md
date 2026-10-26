# Stage 1B：CHAP 式双域边界梯度诊断

冻结 LA10 iteration15000 Student/Teacher，配对复用 Stage1A 的8L/16U病例及每例2个固定裁块。A 保持只读。B1替换有标签目标，B2替换无标签目标，B3同时替换。仅独立分析脚本会导入这些诊断目标；生产训练和评估代码没有改动。

- [预定设计及完整公式](DESIGN.md)
- [主要发现和下一步决策](FINDINGS.md)
- [执行与完整性验证](VERIFICATION.md)
- [完整数值报告](runs/la10_paired/REPORT.md)
- [原始 Stage1A](../stage1a_gradients/FINDINGS.md)

## 目录与接口

`common.py` 是只读 A 数据/模型桥接；`losses.py` 实现诊断用体积平均平衡带 CE、三维概率梯度一致性及负对照；`extract.py` 单次 Student 前向提取所有独立目标的参数梯度；`analyze.py` 做 A/B 配对统计及发现病例通道计划；`supplement.py` 保存逐通道跨尺度统计；`interventions.py`/`ablate.py` 做临时 feature attenuation；`summarize_ablations.py` 以病例为单位汇总；`verify.py` 独立检查结果与历史保护；`report.py` 生成数值报告与独立 PNG/PDF。

梯度参数、通道轴和图可达性见 `runs/la10_paired/parameter_mapping.json`，完全等于 A。原始梯度 NPY 为 `[observation,head,channel,kernel_coordinate]`，不是空间特征图梯度。无效或不可达用 NaN 和明确原因保存。每次分割前向只运行一次 encoder/decoder，多个独立 autograd.grad 共享该图，不是重复前向获取特征。

## 主要输出

- `loss_coverage.json`/`coverage_summary.csv`：每目标/尺度的带宽、native计数、软占有率、覆盖、ESS和无效原因。
- `observation_norms.csv`：每个crop/目标/参数/头/通道原始范数、阈值、有效性、四尺度平均贡献与固定 A 分母贡献。
- `expanded_channel_relations.csv`：每通道 L/U 有符号余弦、有效病例对比例、范数、同病例重复/跨病例稳定性及固定 A 分母敏感性。
- `method_scale_summary.csv`：17组对照×9阶段×4头，包括A/B1/B2/B3和带宽/区域负对照；6个结构不可达单元不记为零。
- `paired_target_comparisons.csv`：同病例相同参数坐标上的边界vs分割、edge vs MT、区域对照梯度方向。
- `channel_multiscale.csv`：逐通道跨头余弦、SVD有效秩及梯度抵消比（主宽度2及A/负对照）。其他宽度有原始梯度及聚合跨尺度表。
- `rankings.csv`/`channel_plan.json`：仅发现病例确定的主通道分数与K=2组；ranking_agreement、width_sensitivity及rank_stability记录A/B差异和稳定性。
- `ablations/singletons.csv`：5阶段所有304个通道在8个probe裁块、3个区域的逐通道最终预测干预结果。
- `ablations/groups.csv`：各阶段A/B排名、相关性簇和8个随机K=2组，匹配实际扰动L2及matched singleton非加性。
- `ablations/group_vs_A.csv`、`group_vs_random.csv`、`group_specificity.csv`：按4个病例聚合的随机组对比和边界vs区域的差分对比；1000次病例bootstrap。
- `verification_results.json`：自动化产物检查；`PROTECTION_BEFORE.json`保存修改前500个历史文件哈希。

矩阵 L/U 余弦：先每病例两个裁块梯度取均值，再按同一权重通道计算8×16病例对余弦。通道须≥75%有效病例对，格值为通道病例对平均值的中位数。重复/跨病例稳定性表的聚合为所有有效通道×病例观测的中位数，与原A报告先每通道再聚合的稳定性汇总次序可能略不同；本报告A/B统一使用相同次序。病例对不独立，不能当作128个独立样本。

大文件 `raw/`、expanded CSV、逐观测norm CSV留在本地独立目录，gitignore避免将约11GiB原始梯度纳入Git；其SHA256记录在结果清单。未删除或覆盖历史A输出。

## 复现

运行环境：`/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python`，PyTorch2.9.1+cu126；GPU运行用主机可见的A100设备1。新运行必须用新的不存在的output目录，不覆盖已有run。

```bash
PY=/aita_data/home/aita_duanxuelong/.conda/envs/SSL29/bin/python
$PY -B tests/test_stage1b_boundary.py
$PY -B analysis/stage1b_boundary/extract.py --device cuda:1 --smoke --output analysis/stage1b_boundary/runs/new_smoke
$PY -B analysis/stage1b_boundary/extract.py --device cuda:1 --output analysis/stage1b_boundary/runs/new_paired --smoke-proof analysis/stage1b_boundary/runs/new_smoke/extraction_results.json
$PY -B analysis/stage1b_boundary/analyze.py --run analysis/stage1b_boundary/runs/new_paired
$PY -B analysis/stage1b_boundary/supplement.py --run analysis/stage1b_boundary/runs/new_paired
$PY -B analysis/stage1b_boundary/ablate.py --device cuda:1 --run analysis/stage1b_boundary/runs/new_paired --smoke --output analysis/stage1b_boundary/runs/new_ablation_smoke
$PY -B analysis/stage1b_boundary/ablate.py --device cuda:1 --run analysis/stage1b_boundary/runs/new_paired --output analysis/stage1b_boundary/runs/new_paired/ablations --smoke-proof analysis/stage1b_boundary/runs/new_ablation_smoke/ablation_results.json
$PY -B analysis/stage1b_boundary/summarize_ablations.py --run analysis/stage1b_boundary/runs/new_paired --ablation analysis/stage1b_boundary/runs/new_paired/ablations
$PY -B analysis/stage1b_boundary/verify.py --run analysis/stage1b_boundary/runs/new_paired --ablation analysis/stage1b_boundary/runs/new_paired/ablations
$PY -B analysis/stage1b_boundary/report.py --run analysis/stage1b_boundary/runs/new_paired
```

保护清单是本任务开始时已有产物的只读快照，不要为复现而覆盖。分析脚本从旧 Stage1A 的 A 原始梯度读取，不重新训练或用其他裁块替代。
