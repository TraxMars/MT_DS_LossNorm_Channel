# Stage 1B：预先固定的诊断设计

本阶段是冻结模型的探索性诊断，不训练、不修改生产代码、不建立新预测头。A 是只读的 Stage1A；B1=(GT boundary, original MT)，B2=(original segmentation, Teacher boundary consistency)，B3=(GT boundary, Teacher boundary consistency)。所有参数坐标、图可达性、有效性规则、checkpoint、病例及裁块与 A 一致。

## 配对条件

LA 10%，iteration15000 的同一 `last_checkpoint.pth` 内 Student/Teacher。FP32，TF32 off，两者 eval。Student 参数保留 requires_grad，Teacher requires_grad=False，目标 detach/no_grad。`autograd.grad` 不写 `.grad`。8L/16U，每病例2个固定112×112×80裁块，seed142。A 的八个标注病例参与过 Baseline 训练，本阶段不能声称独立泛化验证。未读取 U label，不用 test.list 做选择、调参或评价。

九阶段参数映射完全复用 A。解码器 D4/D3/D2 选择上采样分支反卷积的输出通道轴1，参数梯度描述的是 skip 相加前的该分支，不是完整阶段特征的梯度。D1 用最终 feature-producing 卷积，不用分割头权重。未提取特征图梯度或 Gradient×Activation；消融操作作用于特征张量，与保存的参数梯度含义不同。

## 有标签目标及体积平均

在**完整 GT 原始体素网格**上用欧氏距离生成内外带，再按 A 的固定坐标裁剪。宽度1/2/3原始体素，主分析预先固定为2。不会沿裁块面重新制造 GT 边界。

`LB = 0.5 mean_inner(CE) + 0.5 mean_outer(CE)`，两侧分别归一化。原任务排版中的 outer 前减号按正号执行：减去 outer CE 会鼓励外侧分类错误，违背平衡分类目标。

粗尺度以 avg_pool3d 做体积平均，保存前景占有率 `q=Avg(y)`、区域权重 `w=Avg(mask)` 及联合质量 `q_region=Avg(mask*y)`。单侧损失为 `-sum[q_region log(p_fg)+(w-q_region) log(p_bg)]/sum(w)`。这保留区域条件下的软前景占有率 `q_region/w`，而不是乘用两个分别平均量；后者会在同一 coarse cell 中把非带内体素标签混入该侧目标。本定义严格等价于 coarse logits nearest 上采样后计算原始带内 CE，不会 nearest 下采样 GT 后再膨胀。每侧 native mass、coarse support/ESS、全体积前景占有率在带内的加权统计均保存。

R 对照在同一 crop 中按类别随机无放回抽取，内前景/外背景数分别严格匹配 B；记录与真实带的重叠比例。I 对照取距真实边界大于带宽的前景内部与远背景，同样严格匹配计数，不能重复采样填满。两侧至少8体素、每尺度 ESS≥2；不足显式无效，不得补零。

## 无标签目标

仅 Teacher 最终头概率 `pT` 定义候选带：阈值0.5构造 level set，其内外欧氏距离≤w，排除 crop 边缘4体素。三维差分 `G` 在内部用中心差分、端点用单侧差分，每轴 spacing 为 native_size/head_size，因此单位为原图体素。候选权重为 `band * ||G(pT)|| / max(band * ||G(pT)||)`，stop-gradient。各尺度平均池化权重，并计算

`LU_edge_s = sum w_s * mean_xyz[(G(pS_s)-G(stopgrad(pT_s)))²] / sum w_s`。

不叠加普通 MT 项。每头使用相应 Teacher 概率；候选带来自最终头。宽度2另外将有效内部位置的原始权重整体置换，保留原始支持数、权重质量及直方图，作为空间定位负对照。粗尺度平均后，置换对照的 ESS/权重分布可以不同，须一并解释。

预先固定无效规则：概率跨度<1e-4，原始或对应尺度最大梯度<1e-6，无前景或无背景、level set少于16体素、有效权重支持少于16、带覆盖有效内部网格>0.5，或尺度 ESS<2。Teacher 信心、梯度幅值或预测一致性均不能直接等同于真实轮廓准确性。

## 梯度统计及 LossNorm

原始参数梯度是主要证据。四个头各自独立反传，不把不同病例空间坐标的特征梯度逐点比较。沿用 A 的阈值：`max(1e-12,1e-6*该参数当前观测最大通道范数)`。不可达、过小、精确零、非有限、目标无效分开记录，余弦为 NA。

同病例2裁块先在相同参数坐标平均，再计算 L×U 病例对有符号余弦。可靠通道至少75%病例对有效。九阶段×四头记录原始范数、同病例重复、跨病例稳定性及跨头余弦/SVD有效秩。A 的 U/L病例群不同，笛卡尔病例对没有配准含义。

A 原训练 `Lsup=mean4[(CE+Dice)/2]`、`Lmt=mean4[MSE(pS,pT)]`，每头归一化贡献分别为 `g_s/(4 max(stopgrad(Lsup),eps))` 和 `0.3 g_s/(4 max(stopgrad(Lmt),eps))`。本阶段未给 B 单独施加 LossNorm。次要输出只用**原 A 对应裁块的分母**缩放 B，作为控制性重新加权，不是新 B 训练公式。正标量不改同观测余弦，但可能改变先归一化后跨裁块平均的方向。损失量纲不同，不能用 B/A 原始梯度大小直接宣布优劣。

## 通道计划和最终预测干预

预先选 E3/E4/D3/D2/D1 详细分析，主 K=2。A 病例列表偶数索引4L/8U用于发现，奇数索引4L/8U为 probe；参数统计仍保留全部 A 病例。通道排名不读取 probe 的 GT 或预测效果。发现阶段最终 D1 头的 CHAP 式分数为 `cos(mean gL_c,mean gU_c)*||mean gU_c||/max_c||mean gU_c||`，两域有效覆盖各≥75%。正分从高到低，不根据效果翻转方向。

对4L/8U病例均值范数取 log，再分别对 L/U 的每通道病例响应中心化、标准化，拼接后 k=4 KMeans；选平均分最高且至少两通道的簇，再取其中前两名。它是描述性相关性分组，不是协同证据。100次发现病例 bootstrap 检查分数/Top2稳定性。每阶段8组固定随机K=2集合，从 A/B 共同有效通道中选取。

在4个 probe L病例的两个固定crop上，逐通道扫描及K=2联合组消融，评价**最终 D1 分割输出**的 hard/soft Dice、全crop CE、平衡 GT boundary CE、1体素容忍的轮廓 F1 及预测改变比例。仅作裁块评价，未运行完整体积推理或正式评估协议。轮廓F1排除crop外缘2体素，使用6邻接轮廓、3³容忍邻域。

区域 B/R/I 使用完全相同 native 计数；将 union mask 体积平均到阶段大小。阶段/crop singleton L2预算 `δ=0.1*min_region median_nonzero_channels||feature_c*mask_region||`。每次选择 alpha=δ/selected_activation_norm，K=2 预算√2δ；alpha>0.5/零激活显式不可匹配。不同组及不同区域实际扰动 L2匹配。联合消融与两个 matched singleton 用**同一个 pair alpha**，保证联合扰动等于两个坐标扰动之和。所有干预 batch=4，第4行为无干预对照，检查 B4/B1 数值误差及 hooks 清理。

E3/E4在block输出处干预；D3/D2在下游 block 的输入处干预 postskip stage，确保进入最终预测路径；D1在最终 out_conv 输入处干预。只拦辅助头不会影响最终预测，已有自动测试覆盖。

同时报告 `z_joint-z_single1-z_single2+z_base` 和 CE 同类非加性。D1 接线性1×1头，logits可加但 CE仍可能非加，不能用联合损失变化证明通道协同。通道组与随机组比较必须匹配K和实际L2。结果按4个病例聚合再1000次 bootstrap；裁块和8个随机集合不能充当额外独立病例。全部为多重探索性比较，无事后选择“显著”的阶段/带宽。

补充解释：报告还描述 `|Δboundary CE|` 和分数与 hard prediction change 的相关性，作为 signed CE 之外的影响幅值检查；它们不改变通道选择、主带宽或干预预算。D1 的 pointwise 最终头导致非重叠内部干预无法改变边界带 CE，这是结构预期；边界区干预对边界指标的直接支持重叠也不能单独证明边界专属性。共享病例bootstrap索引保证同一通道集合在不同方法别名下产生完全相同区间。
