# 新增文件与修改边界

所有新增内容均在独立分析分支，生产源码和历史结果修改数为0。

| 新文件 | 作用 |
| --- | --- |
| `analysis/stage1b_boundary/common.py` | 固定A输入、只读模型/数据桥接、哈希保护与autograd提取 |
| `analysis/stage1b_boundary/losses.py` | 仅诊断使用的native GT带、平衡CE、3D概率梯度一致性和区域/置换对照 |
| `analysis/stage1b_boundary/extract.py` | 强制先smoke的配对梯度提取 |
| `analysis/stage1b_boundary/analyze.py` | A/B矩阵、有效性、病例/跨尺度关系和发现集通道计划 |
| `analysis/stage1b_boundary/supplement.py` | 逐通道跨尺度与损失/固定A分母敏感性 |
| `analysis/stage1b_boundary/interventions.py` | 临时feature hook与最终预测指标 |
| `analysis/stage1b_boundary/ablate.py` | 按实际L2匹配的singleton/联合组最终预测干预 |
| `analysis/stage1b_boundary/summarize_ablations.py` | 病例级统计、A/随机/区域对照、共享索引bootstrap |
| `analysis/stage1b_boundary/verify.py` | 独立原始结果/矩阵重算/能量/历史完整性检查 |
| `analysis/stage1b_boundary/report.py` | 完整数值报告及6组PNG/PDF |
| `tests/test_stage1b_boundary.py` | 14项CPU自动测试 |
| `analysis/stage1b_boundary/.gitignore` | 大型原始梯度和展开逐观测CSV保留本地，不进Git |
| `README.md`、`DESIGN.md`、`FINDINGS.md`、`VERIFICATION.md`（本目录） | 复现、公式、结果解释、执行范围和限制 |
| `PROTECTION_BEFORE.json`、各测试/环境JSON（本目录） | 预先历史哈希、真实测试结果和环境 |
| `runs/smoke/` | 保留失败的crop元数据类型检查记录 |
| `runs/smoke_checked/`、`runs/ablation_smoke/` | 成功smoke证据 |
| `runs/la10_paired/` | 新的梯度/覆盖/排名/干预/报告/图及验证产物 |
| `ARTIFACT_MANIFEST.json`（本目录） | 最终源文件/产物SHA256；包括本地未纳入Git的大文件，不含自身 |

未改变：VNetDS/九阶段接口、`code/train.py`、`code/objectives.py`及其他生产代码、Stage0/Stage1A源码与结果、六组历史Results/原始checkpoint/config/summary、原MT_DS_LossNorm、数据内容和划分。

无正式训练、无新预测头、无新损失接入、无通道交换或交互。后续研究建议仅写在FINDINGS中，本次停止于Stage1B。
