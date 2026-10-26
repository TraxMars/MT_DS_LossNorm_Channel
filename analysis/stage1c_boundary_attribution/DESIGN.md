# Stage1C C0：预先固定的正确性与真实数据 smoke 设计

本次只实施C0。起点本地`07a0b90f840dbf26776e2f9df9077be47dd90fa3`，分支`experiment/stage1c-boundary-attribution`，全部新增代码/结果位于本目录及独立测试文件。没有训练入口、optimizer、九层预测头或通道组。C1的完整边界专属性/难度匹配对照、C2的Teacher增强可靠性实验、C3的通道组实验均不执行。

## 起点和历史保护

GitHub main在执行前锁定为`5f6303a4ffde996a8200c294a47d35062f54e8b6`，全部290个远程文件与本地同路径Git blob一致；本地另外保留22个Stage1A/1B图件。远程上传历史与本地提交SHA不同，不声称两个完整Git树相同。代码、数值表、验证记录和公开artifact manifest一致，本地Stage1B manifest的210个文件（含Git忽略的原始梯度）逐个SHA256验证通过。checkpoint本地提供并按已有SHA验证，不能声称GitHub托管了权重/原始数组。证据见REMOTE_AUDIT/REMOTE_INVENTORY；PROTECTION_BEFORE保护711个已有文件，不计算HDF5容器哈希以避免读取U标签字节。

## 固定病例和裁块

沿用Stage1B发现集的前2L、前2U，不读取probe病例：

- L：06SR5RBREL16DQ6M8LWS、1D7CUD1955YZPGK8XHJX。
- U：3C2QTUNI0852XV7ZH4Q1、57SGAJMLCTCH92QUA0EE。

每病例原有repeat0/1固定112×112×80裁块，共8crop，seed142；元数据必须与Stage1A observations逐项一致。均来自Baseline训练病例，不是独立泛化验证。未使用test.list选例、调参或评价。

LA10 iteration15000原始Student/Teacher配对checkpoint，SHA256 `85489c52c0905aebde65eacf5cea215aac51ee825ed3e5a845da86fe7f23e168`。FP32、TF32关闭，eval、Student保留autograd、Teacher冻结/no_grad。损失用FP64运算和归约减轻差分消减误差，模型前向/反向仍为FP32；原A/B兼容重放严格使用原FP32公式。

## 几何和目标

只用最终D1 logits构造目标，分析E3/E4/D3/D2/D1五阶段。L位置对在native GT网格上取6邻接跨类别体素面，定向为FG正端/BG负端。两端均在真实有效图像内，排除crop边缘4体素；不把crop边缘当轮廓。按case/repeat固定种子无放回均匀取至多2048对，至少32对，否则显式无效。重复端点可来自不同相邻面，记录pair数及唯一端点数。

`z=logit_fg-logit_bg`，`Ltransition_m=mean softplus(m-zplus+zminus)`。主margin1.0，0.5和2.0只作既定敏感性检查，不根据结果选择最佳margin。另独立保存普通区域CE与原Stage1B宽度2平衡边界CE。目标互不相加，不独立LossNorm。该排序目标不约束绝对分类阈值，不能替代CE。

U只读image。Teacher最终概率pT构造6邻接候选对（pT>=0.5为正侧），排除crop边缘4体素，要求Teacher概率跨度>=1e-4、端点概率差>1e-3且至少32对；固定种子取至多2048对。Student与Teacher读取同一原始crop和同一整数坐标，无增强、无插值；只验证坐标和stop-gradient机制，**未验证Teacher边界准确性或增强可靠性**。`LUtransition=mean[(pSplus-pSminus)-(stopgrad(pTplus)-stopgrad(pTminus))]^2`。保留原MT和Stage1B edge_w2作为兼容/局部归因参考。候选有效只表示几何/数值可用，不代表可用于正式训练；C2未执行。

CPU检查位置对方向、边缘、空集合、翻转逆坐标、相同端点集合的配对置换，以及粗尺度映射坍缩。C0不在真实数据粗头强制构造位置对；loss head只有D1，与取特征的五个stage不同。

## 同一节点与干预合同

所有梯度采集和干预使用同一NODE_SPECS：E3/block_three输出，E4/block_four输出，D3/block_seven输入，D2/block_eight输入，D1/out_conv输入。基线采集的节点必须与九阶段接口对应tensor为同一个对象，并验证D3=block_six_up输出+E3 skip、D2=block_seven_up输出+E2 skip。使用最终D1损失时，节点梯度应与接口tensor梯度完全相同。

D3/D2在skip融合后下游路径处削弱，不改对应辅助头输入。不得用反卷积参数梯度代替融合特征梯度。CPU显式验证辅助头自身损失的梯度与这个下游干预不一致的反例，防止未来混用四头总损失。参数梯度单独记录shape/范数/原始向量，特征梯度单独记录shape/范数/内积/primary原始梯度，名称和文件目录分开。

L干预native M为Stage1B原始GT内外宽2带union；U M为原Teacher宽2候选权重的正支持，均在基线冻结并平均池化到各特征网格。M、候选对、Teacher目标、剂量和alpha全部stop-gradient；扰动后不重建。

每阶段/crop固定检查通道`[0,C//2,C-1]`，不按分数选通道。相对能量预算r严格为0.01、0.03、0.10，`delta=r*median_nonzero_channels ||F_c*M||2`。每个通道`alpha=delta/||F_c*M||2`，alpha<=0.5；零激活、非有限或超上限明确无效，不截断alpha后伪称能量匹配。模型实际改动`F'_c=F_c-alpha*F_c*M`。每次batch4为3个固定通道及1个未干预控制，所有行独立计算损失，实际L2必须匹配预算。

一阶预测`Delta_hat=-alpha*sum(g_c*F_c*M)`，保留符号；另记录净幅值abs(Delta_hat)和逐位置绝对和`alpha*sum(abs(g_c*F_c*M))`，不混为同一量。实际变化是同batch内`Lperturbed-Lcontrol`。保存各独立目标的Delta_hat/actual及最终全CE、边界CE、Dice、轮廓F1、预测改变比例。不聚类、不排名、不生成TopK。

## 预先定义数值不确定区间

每个crop/目标在任何非零干预前，执行五个stage的零alpha batch4前向，得到20个null损失相对单样本基线的差值。固定规则：

`epsilon=max(1e-9,1e-6*max(abs(L0),1e-3),5*max(abs(null_loss-L0)))`。

所有损失以原始单位记录，区间为[-epsilon,+epsilon]。实际Delta落在区间内记`direction_indeterminate`；预测同样单独判定。空目标/非有限/无效预算为NA，不能算成零效应。报告不可判定比例、预测覆盖率、实际可判定样本的符号准确率，以及有正负两类时的balanced accuracy；另报总预测多数方向的基线。不因方向比例不好而移动epsilon。

小扰动受FP32前向量化、ReLU转折及相消影响，并非要求每个通道逐档完全一致。报告每档MAE、相对残差、Spearman及固定通道三档配对趋势；禁止把接近噪声的方向强制正负分类。

## C0通过条件（先于真实非零干预固定）

1. CPU合同测试、原Stage0/1A/1B回归、strict checkpoint及A/B损失/最终头参数梯度兼容检查通过。
2. 八个固定crop几何/目标有效，五stage节点身份和skip等式通过；Teacher/.grad/模型参数/BN/RNG/原始文件全部不变；预算有效处实测L2达标。
3. 对主Ltransition_m1及Utransition，固定通道中三档均有效、最小档abs(Delta_hat)>5epsilon且最小档alpha<=0.05者作为同一组局部可辨识观测。每域至少10个此类观测；不足标记数值证据不足，不能自动补选通道。
4. 上述同一组观测：最小剂量median相对残差`abs(actual-hat)/abs(hat)`<=0.25；其值不超过最大剂量中位数+0.02；最小剂量median绝对残差不超过最大剂量+median epsilon。主目标最小剂量在实际可判定观测上的符号准确率>=0.80。大剂量不要求完全一致；不满足如实记录FAIL/证据不足，不更改剂量或按效果取通道。

这些是C0实现和局部近似可用性的工程门槛，不是边界专属性、可靠通道或训练收益证明。即使C0通过，也不自动进入C1/C2/C3。
