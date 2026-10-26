# C0 测试与完整性报告

最终判定：**PASS**。实际结果见TEST_RESULTS.json、REGRESSION_RESULTS.json和最终run的verification_results.json。

| 检查 | 实际结果 |
| --- | --- |
| C0 CPU合同测试 | 21/21 PASS，0.477s |
| test_stage_features.py | 7/7 PASS，9.842s |
| test_stage1a_diagnostics.py | 10/10 PASS，0.657s |
| test_stage1b_boundary.py | 14/14 PASS，0.229s |
| verify_core.py | exit0：上游输出、Teacher梯度隔离、目标/EMA、LossNorm和滑窗检查通过 |
| 最终GPU smoke | 4病例/8crop，65.980s，峰值4288.7MiB |
| 历史SHA256 | 711文件一致，changed=[] |
| 主目标归因独立重算 | 360个Delta_hat，最大绝对差8.674e-19 |
| A/B参数梯度兼容 | 80个原向量，最大相对L2差0.000e+00 |

合计52项unittest（C0 21+历史31），另有core验证。历史suite只作为模块导入，报告写到C0，未运行会覆盖Stage1A/1B旧报告的main。

## 覆盖内容

native位置对方向/6邻接/有效边缘；确定性采样；空边界；排序目标平移不变性；最终D1 tensor限制；翻转逆坐标；粗尺度坍缩/占有率；端点置换marginal；Teacher stop-gradient/U GT键拒绝；五个真实计算节点和skip融合；Stage1B干预等价；D3/D2辅助头梯度反例；特征/参数梯度独立与已有.grad保持；小扰动正负方向；净GxA/逐点绝对GxA区别；数值区间；固定剂量/alpha失败；实际张量减法L2与舍入误差；输出路径保护。

## 独立最终检查

| 检查 | 结果 |
| --- | --- |
| executed_only_C0 | True |
| frozen_execution_invariants | True |
| predeclared_design_unchanged_since_execution | True |
| executed_source_files_unchanged | True |
| unit_and_historical_regressions_pass | True |
| eight_native_pair_sets_valid | True |
| 40_live_node_identities | True |
| decoder_skip_fusion_exact | True |
| only_predeclared_discovery_cases | True |
| two_original_crops_per_case | True |
| native_pair_coordinates_and_class_direction | True |
| all_three_fixed_doses_only | True |
| 1440_loss_differences_360_interventions | True |
| actual_energy_matches_budget | True |
| actual_subtraction_energy_and_roundoff_reported | True |
| unmatched_not_fake_zero | True |
| feature_and_parameter_coordinates_separate | True |
| all_recorded_gradient_stats_finite | True |
| uncertainty_calibrated_by_fixed_null_only_formula | True |
| near_noise_directions_never_forced | True |
| residuals_recomputed | True |
| 40_primary_raw_feature_fields_valid | True |
| all_360_primary_predictions_independently_recomputed | True |
| 160_raw_parameter_fields_separate_and_norms_match | True |
| 80_A_B_final_head_parameter_vectors_replayed | True |
| unlabeled_HDF5_image_key_only | True |
| fidelity_predeclared_gates_pass | True |
| 711_historical_SHA256_unchanged | True |
| no_baseline_tracked_changes | True |

## 保留的失败和修正

首轮20项有2个测试夹具失败（TEST_ATTEMPT_01.json）：对两个类别加共同logit偏移，softmax不变；随机模型单个通道可能死ReLU，不能据零响应否定路径。改为只扰动前景logit和已知激活的合成网络后20项通过（TEST_ATTEMPT_02.json）；增加实际L2测试后21项通过。真实checkpoint和选定通道未改变。

第一次真实smoke runs/c0_la10_fixed保留，其L2字段测量的是公式扰动向量。最终runs/c0_la10_checked按实际F_before-F_after测量，同时保存intended_l2和subtraction_roundoff_l2，重跑同一8crop。前向削弱公式、病例、采样、剂量和方向门槛未改变。

初次独立验证的检查和JSON已PASS，但CLI打印numpy bool时exit1；输出改用既有clean序列化后，对最终run重做验证并exit0。可选远程完工复核首个响应未解析到SHA，不视为远程变化；重试commits/main仍为锁定SHA。记录见EXECUTION_NOTES.json和REMOTE_RECHECK*。

## 状态和数据保护

Student/Teacher完整state_dict（参数与BN）精确不变、Teacher frozen/no_grad、均为eval、.grad均为None，CPU/CUDA/Python/numpy RNG不变，无残留hooks。原checkpoint strict=True加载，SHA不变。U loader只读image，GT被U目标接口拒绝，U的GT准确性指标为不可用。独立验证只重读原2L的GT验证方向；HDF5只读且size/mtime不变，不哈希U容器以避免读取标签字节。生产源码、历史Stage0/1A/1B、Results/summary/checkpoint/划分未改。执行源文件及DESIGN与source_at_run中的预先哈希一致。
