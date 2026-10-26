# D1-A 验证记录

LA 72 个、Pancreas 68 个正式观测均完成，涵盖各自全部 L、16 个 U、两裁块、三个真实 2L+2U 病例组的 eval/train 对照及固定 RNG 重放。原始梯度、逐项损失、节点身份、重构、HDF5 访问、BN/RNG 恢复和数据文件 stat 证据分别存储。

34 项人工/回归测试通过：新增完整性 1、原特征接口 7、新诊断 11、原 Stage1A 10、统计管线 5。旧 Stage1A 测试通过 unittest discovery 调用，未运行会写旧 TEST_RESULTS.json 的 __main__。完整日志及机器记录见 ../TEST_RESULTS.json。

正式采集调用原 Stage1A 严格重构函数；LA/Pancreas 分别 2/3 个单元按同一数值规则重放 FP64，未改公式或阈值。Pancreas 两条极小 raw D1 观测的相对误差需按原绝对误差条件理解，详见 NUMERICAL_EDGE_CASES.md。

两套均确认参数/BN/模块模式/requires_grad/grad/RNG 恢复，Teacher 无梯度，训练首组重放误差 0，U GT 读取、optimizer 实例、optimizer step、学习参数更新和 checkpoint 写入均为 0。源文件 SHA 分为采集、汇总、验证和报告步骤，分别保存在 source_manifest.json、summary_source.json、verification_source.json、report_source.json。采集快照可能包含当时尚未执行的报告脚本；执行阶段使用各自单独的 source 记录。

原有 3,063 个受保护历史文件 SHA 全部一致。严格目录检查发现 Boundary 项目并发新增 99 个 C15 文件，因此原 FAIL 记录保留；concurrency_review.json 仅将原有文件不变结论与并发新增分开记录，未删除或使用这些新增产物。
