# Stage 1B：完整数值报告

主结果采用原图宽度2；宽度1/3是预先声明的敏感性分析。全部为冻结eval、FP32、同一iteration15000配对checkpoint；病例和crop与A一致。详细定义见[DESIGN](../../DESIGN.md)，研究结论见[FINDINGS](../../FINDINGS.md)。

## 九阶段×四头：A/B1/B2/B3 有符号 L/U 参数梯度余弦

格值为可靠通道（≥75%有效病例对）的病例对平均余弦中位数。先每病例两个crop梯度平均，再比较8L×16U病例；不可达为NA。

### A

| stage | D4 | D3 | D2 | D1 |
| --- | --- | --- | --- | --- |
| E1 | 0.0223 | 0.1551 | 0.2641 | 0.2307 |
| E2 | 0.0506 | 0.1205 | 0.1136 | 0.1669 |
| E3 | 0.0632 | 0.2589 | 0.1554 | 0.1746 |
| E4 | 0.1346 | 0.1498 | 0.0986 | 0.0979 |
| E5 | 0.1019 | 0.2005 | 0.1474 | 0.1643 |
| D4 | 0.1090 | 0.1247 | 0.0813 | 0.0933 |
| D3 | NA | 0.3337 | 0.2186 | 0.2441 |
| D2 | NA | NA | 0.4195 | 0.4202 |
| D1 | NA | NA | NA | 0.4545 |

### B1

| stage | D4 | D3 | D2 | D1 |
| --- | --- | --- | --- | --- |
| E1 | -0.1560 | 0.0397 | 0.2197 | 0.5767 |
| E2 | -0.0160 | -0.0182 | 0.0191 | 0.2842 |
| E3 | 0.0367 | 0.2817 | 0.0108 | 0.2490 |
| E4 | 0.2100 | 0.0693 | 0.0936 | 0.1630 |
| E5 | 0.2413 | 0.1687 | 0.1610 | 0.2689 |
| D4 | 0.1620 | 0.0776 | 0.0977 | 0.1472 |
| D3 | NA | 0.3904 | 0.1825 | 0.3225 |
| D2 | NA | NA | 0.4730 | 0.5489 |
| D1 | NA | NA | NA | 0.7219 |

### B2

| stage | D4 | D3 | D2 | D1 |
| --- | --- | --- | --- | --- |
| E1 | 0.0399 | 0.1409 | 0.2548 | 0.2241 |
| E2 | 0.0461 | 0.1402 | 0.1384 | 0.1970 |
| E3 | 0.0646 | 0.2659 | 0.1657 | 0.1818 |
| E4 | 0.1444 | 0.1576 | 0.1051 | 0.0975 |
| E5 | 0.1122 | 0.2008 | 0.1515 | 0.1662 |
| D4 | 0.1099 | 0.1281 | 0.0799 | 0.0830 |
| D3 | NA | 0.3236 | 0.2224 | 0.2424 |
| D2 | NA | NA | 0.4334 | 0.4467 |
| D1 | NA | NA | NA | 0.5596 |

### B3

| stage | D4 | D3 | D2 | D1 |
| --- | --- | --- | --- | --- |
| E1 | -0.1436 | 0.0414 | 0.1903 | 0.5761 |
| E2 | -0.0148 | -0.0167 | 0.0096 | 0.3188 |
| E3 | 0.0360 | 0.2937 | 0.0590 | 0.2704 |
| E4 | 0.2615 | 0.0788 | 0.1128 | 0.1717 |
| E5 | 0.2643 | 0.1728 | 0.1768 | 0.2705 |
| D4 | 0.1517 | 0.0883 | 0.1064 | 0.1368 |
| D3 | NA | 0.3653 | 0.1952 | 0.3139 |
| D2 | NA | NA | 0.5133 | 0.5581 |
| D1 | NA | NA | NA | 0.9375 |

## 有效性矩阵（B3主宽度）

| stage | D4 | D3 | D2 | D1 |
| --- | --- | --- | --- | --- |
| E1 | 16/16 | 16/16 | 16/16 | 16/16 |
| E2 | 32/32 | 32/32 | 32/32 | 32/32 |
| E3 | 64/64 | 64/64 | 64/64 | 64/64 |
| E4 | 128/128 | 128/128 | 128/128 | 128/128 |
| E5 | 256/256 | 256/256 | 256/256 | 256/256 |
| D4 | 128/128 | 128/128 | 128/128 | 128/128 |
| D3 | graph NA | 64/64 | 64/64 | 64/64 |
| D2 | graph NA | graph NA | 32/32 | 32/32 |
| D1 | graph NA | graph NA | graph NA | 16/16 |

逐crop范数阈值和状态见observation_norms.csv；病例均值可靠不代表每个crop均可靠。完整17组对照矩阵保存在method_scale_summary.csv。

## 同病例目标/区域对照梯度方向（最终D1头）

| stage | boundary_vs_seg_w2 | edge_vs_MT_w2 | boundary_vs_random_w2 | boundary_vs_core_w2 | edge_vs_shuffled |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.9970 | 0.9989 | 0.9950 | 0.9235 | 0.9997 |
| E2 | 0.9644 | 0.9806 | 0.9556 | 0.6906 | 0.9935 |
| E3 | 0.9687 | 0.9598 | 0.9412 | 0.6202 | 0.9856 |
| E4 | 0.9691 | 0.9538 | 0.9466 | 0.7233 | 0.9840 |
| E5 | 0.9684 | 0.9698 | 0.9647 | 0.8019 | 0.9876 |
| D4 | 0.9638 | 0.9589 | 0.9558 | 0.7803 | 0.9796 |
| D3 | 0.9656 | 0.9500 | 0.9297 | 0.6118 | 0.9751 |
| D2 | 0.9790 | 0.9783 | 0.9677 | 0.6984 | 0.9927 |
| D1 | 0.9977 | 0.9984 | 0.9955 | 0.9790 | 0.9996 |

## 重复裁块及跨病例方向稳定性

| stage | method | within L | within U | between L | between U |
| --- | --- | --- | --- | --- | --- |
| E1 | A | 0.9472 | 0.9945 | 0.7012 | 0.9705 |
| E1 | B3_w2 | 0.9739 | 0.9949 | 0.9226 | 0.9755 |
| E2 | A | 0.6177 | 0.8496 | 0.1420 | 0.5858 |
| E2 | B3_w2 | 0.7064 | 0.8928 | 0.2913 | 0.6584 |
| E3 | A | 0.5547 | 0.7310 | 0.1580 | 0.4948 |
| E3 | B3_w2 | 0.6381 | 0.7953 | 0.2540 | 0.5281 |
| E4 | A | 0.3579 | 0.5978 | 0.0771 | 0.3798 |
| E4 | B3_w2 | 0.4250 | 0.6761 | 0.1527 | 0.4355 |
| E5 | A | 0.1763 | 0.5108 | 0.1239 | 0.4813 |
| E5 | B3_w2 | 0.2507 | 0.5361 | 0.2493 | 0.4865 |
| D4 | A | 0.2120 | 0.4838 | 0.0541 | 0.3071 |
| D4 | B3_w2 | 0.2603 | 0.5234 | 0.1300 | 0.3450 |
| D3 | A | 0.3861 | 0.6718 | 0.1425 | 0.6069 |
| D3 | B3_w2 | 0.4665 | 0.7084 | 0.2715 | 0.6104 |
| D2 | A | 0.8039 | 0.9288 | 0.5324 | 0.8683 |
| D2 | B3_w2 | 0.8696 | 0.9443 | 0.7137 | 0.8758 |
| D1 | A | 0.9864 | 0.9982 | 0.9299 | 0.9626 |
| D1 | B3_w2 | 0.9958 | 0.9971 | 0.9708 | 0.9696 |

跨尺度相似、SVD有效秩与抵消比见channel_multiscale.csv/cross_scale_summary.csv/multiscale_summary.csv；这些是描述梯度几何，不能证明协同。

## 三带宽、四尺度目标覆盖

| domain | objective | head | valid/total | inner ESS | outer ESS | U ESS | U band fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L | band_w1 | D4 | 16/16 | 288.5214 | 302.5261 | NA | NA |
| L | band_w1 | D3 | 16/16 | 1186.5360 | 1244.5328 | NA | NA |
| L | band_w1 | D2 | 16/16 | 4611.2343 | 4870.3655 | NA | NA |
| L | band_w1 | D1 | 16/16 | 15931.5000 | 16629.0000 | NA | NA |
| L | random_w1 | D4 | 16/16 | 340.6174 | 1543.2590 | NA | NA |
| L | random_w1 | D3 | 16/16 | 2199.2083 | 7472.5125 | NA | NA |
| L | random_w1 | D2 | 16/16 | 9185.9686 | 14656.5018 | NA | NA |
| L | random_w1 | D1 | 16/16 | 15931.5000 | 16629.0000 | NA | NA |
| L | core_w1 | D4 | 16/16 | 303.4120 | 1512.7692 | NA | NA |
| L | core_w1 | D3 | 16/16 | 1984.4146 | 7419.5509 | NA | NA |
| L | core_w1 | D2 | 16/16 | 8749.0968 | 14511.7085 | NA | NA |
| L | core_w1 | D1 | 16/16 | 15931.5000 | 16629.0000 | NA | NA |
| L | band_w2 | D4 | 16/16 | 285.5634 | 319.9706 | NA | NA |
| L | band_w2 | D3 | 16/16 | 1249.8567 | 1378.9616 | NA | NA |
| L | band_w2 | D2 | 16/16 | 5731.8395 | 6404.4663 | NA | NA |
| L | band_w2 | D1 | 16/16 | 31610.0000 | 35049.5000 | NA | NA |
| L | random_w2 | D4 | 16/16 | 342.0285 | 1628.1057 | NA | NA |
| L | random_w2 | D3 | 16/16 | 2345.5616 | 9721.4980 | NA | NA |
| L | random_w2 | D2 | 16/16 | 12832.5528 | 27175.3489 | NA | NA |
| L | random_w2 | D1 | 16/16 | 31610.0000 | 35049.5000 | NA | NA |
| L | core_w2 | D4 | 16/16 | 276.0718 | 1575.8351 | NA | NA |
| L | core_w2 | D3 | 16/16 | 1914.0995 | 9431.8422 | NA | NA |
| L | core_w2 | D2 | 16/16 | 11114.6742 | 26942.6025 | NA | NA |
| L | core_w2 | D1 | 16/16 | 31610.0000 | 35049.5000 | NA | NA |
| L | band_w3 | D4 | 16/16 | 287.1741 | 344.2430 | NA | NA |
| L | band_w3 | D3 | 16/16 | 1358.5072 | 1608.6813 | NA | NA |
| L | band_w3 | D2 | 16/16 | 7616.1440 | 9138.1222 | NA | NA |
| L | band_w3 | D1 | 16/16 | 49309.0000 | 59069.0000 | NA | NA |
| L | random_w3 | D4 | 16/16 | 342.3733 | 1660.1324 | NA | NA |
| L | random_w3 | D3 | 16/16 | 2412.9095 | 10959.4210 | NA | NA |
| L | random_w3 | D2 | 16/16 | 15058.5273 | 39459.2237 | NA | NA |
| L | random_w3 | D1 | 16/16 | 49309.0000 | 59069.0000 | NA | NA |
| L | core_w3 | D4 | 16/16 | 243.7805 | 1573.9772 | NA | NA |
| L | core_w3 | D3 | 16/16 | 1707.5159 | 10330.4470 | NA | NA |
| L | core_w3 | D2 | 16/16 | 11305.7614 | 38596.2033 | NA | NA |
| L | core_w3 | D1 | 16/16 | 49309.0000 | 59069.0000 | NA | NA |
| U | edge_w1 | D4 | 32/32 | NA | NA | 295.0378 | 0.0398 |
| U | edge_w1 | D3 | 32/32 | NA | NA | 1207.6155 | 0.0398 |
| U | edge_w1 | D2 | 32/32 | NA | NA | 5316.0241 | 0.0398 |
| U | edge_w1 | D1 | 32/32 | NA | NA | 29232.1200 | 0.0398 |
| U | edge_w2 | D4 | 32/32 | NA | NA | 297.5203 | 0.0812 |
| U | edge_w2 | D3 | 32/32 | NA | NA | 1238.6845 | 0.0812 |
| U | edge_w2 | D2 | 32/32 | NA | NA | 5699.9603 | 0.0812 |
| U | edge_w2 | D1 | 32/32 | NA | NA | 33149.6579 | 0.0812 |
| U | shuffle_w2 | D4 | 32/32 | NA | NA | 1654.4224 | 0.0812 |
| U | shuffle_w2 | D3 | 32/32 | NA | NA | 9031.6060 | 0.0812 |
| U | shuffle_w2 | D2 | 32/32 | NA | NA | 25641.1488 | 0.0812 |
| U | shuffle_w2 | D1 | 32/32 | NA | NA | 33149.6579 | 0.0812 |
| U | edge_w3 | D4 | 32/32 | NA | NA | 297.9475 | 0.1309 |
| U | edge_w3 | D3 | 32/32 | NA | NA | 1244.4203 | 0.1309 |
| U | edge_w3 | D2 | 32/32 | NA | NA | 5717.7772 | 0.1309 |
| U | edge_w3 | D1 | 32/32 | NA | NA | 33266.6903 | 0.1309 |

## 通道排名与稳定性

| stage | A vs | score Spearman | Top2 Jaccard |
| --- | --- | --- | --- |
| E3 | B1_w2 | 0.8835 | 1.0000 |
| E3 | B2_w2 | 0.7208 | 0.3333 |
| E3 | B3_w2 | 0.6861 | 0.3333 |
| E4 | B1_w2 | 0.8325 | 1.0000 |
| E4 | B2_w2 | 0.6766 | 0.3333 |
| E4 | B3_w2 | 0.5822 | 0.3333 |
| D3 | B1_w2 | 0.9592 | 0.3333 |
| D3 | B2_w2 | 0.8294 | 0.0000 |
| D3 | B3_w2 | 0.7825 | 0.0000 |
| D2 | B1_w2 | 0.9714 | 1.0000 |
| D2 | B2_w2 | 0.9659 | 1.0000 |
| D2 | B3_w2 | 0.9520 | 1.0000 |
| D1 | B1_w2 | 0.9912 | 1.0000 |
| D1 | B2_w2 | 0.9147 | 0.3333 |
| D1 | B3_w2 | 0.9235 | 0.3333 |

| stage | method | bootstrap rho median | Top2 Jaccard mean |
| --- | --- | --- | --- |
| E3 | A | 0.7203 | 0.4767 |
| E3 | B1 | 0.8089 | 0.4967 |
| E3 | B2 | 0.7908 | 0.4733 |
| E3 | B3 | 0.8359 | 0.6300 |
| E4 | A | 0.5171 | 0.3067 |
| E4 | B1 | 0.5942 | 0.3000 |
| E4 | B2 | 0.5995 | 0.3100 |
| E4 | B3 | 0.6828 | 0.4067 |
| D3 | A | 0.8313 | 0.1733 |
| D3 | B1 | 0.8635 | 0.2633 |
| D3 | B2 | 0.8250 | 0.4000 |
| D3 | B3 | 0.8785 | 0.6067 |
| D2 | A | 0.9434 | 0.6267 |
| D2 | B1 | 0.9758 | 0.7300 |
| D2 | B2 | 0.9578 | 0.6633 |
| D2 | B3 | 0.9723 | 0.7600 |
| D1 | A | 0.9735 | 0.8033 |
| D1 | B1 | 0.9912 | 0.9000 |
| D1 | B2 | 0.9912 | 0.9133 |
| D1 | B3 | 0.9941 | 0.9600 |

主K=2；宽度1/3敏感性见ranking_width_sensitivity.csv。聚类信息和实际通道ID见cluster_exploration.json及channel_plan.json。聚类只用发现病例。

## 最终分割预测：边界区域K=2联合消融

以下是边界区域消融造成的最终GT boundary CE变化，减去同K同实际L2随机集合均值。正值表示更大边界CE影响，不等于训练性能提升。按4个probe病例先平均crop，再1000次病例bootstrap。探索性区间不作多重检验显著性结论。

| stage | kind | method | Δboundary CE - random | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| E3 | cluster | A | -6.826e-05 | -2.349e-04 | 9.841e-05 |
| E3 | cluster | B1 | 4.488e-06 | -1.877e-04 | 1.835e-04 |
| E3 | cluster | B2 | -8.154e-04 | -0.0013 | -2.925e-04 |
| E3 | cluster | B3 | -8.154e-04 | -0.0013 | -2.925e-04 |
| E3 | ranked | A | 4.488e-06 | -1.877e-04 | 1.835e-04 |
| E3 | ranked | B1 | 4.488e-06 | -1.877e-04 | 1.835e-04 |
| E3 | ranked | B2 | -6.221e-04 | -0.0010 | -2.106e-04 |
| E3 | ranked | B3 | -6.221e-04 | -0.0010 | -2.106e-04 |
| E4 | cluster | A | -1.754e-05 | -4.585e-05 | 8.785e-06 |
| E4 | cluster | B1 | -1.499e-04 | -2.740e-04 | -3.441e-05 |
| E4 | cluster | B2 | -3.702e-05 | -6.237e-05 | -5.197e-06 |
| E4 | cluster | B3 | -2.989e-06 | -1.881e-05 | 8.530e-06 |
| E4 | ranked | A | -2.974e-06 | -5.973e-05 | 4.412e-05 |
| E4 | ranked | B1 | -2.974e-06 | -5.973e-05 | 4.412e-05 |
| E4 | ranked | B2 | 1.727e-05 | -1.248e-05 | 5.907e-05 |
| E4 | ranked | B3 | 3.986e-05 | 1.226e-05 | 6.664e-05 |
| D3 | cluster | A | -4.979e-04 | -0.0010 | 3.998e-05 |
| D3 | cluster | B1 | -4.979e-04 | -0.0010 | 3.998e-05 |
| D3 | cluster | B2 | 5.474e-04 | -1.971e-04 | 0.0013 |
| D3 | cluster | B3 | 1.930e-04 | -1.002e-04 | 4.661e-04 |
| D3 | ranked | A | -4.979e-04 | -0.0010 | 3.998e-05 |
| D3 | ranked | B1 | -0.0011 | -0.0016 | -3.831e-04 |
| D3 | ranked | B2 | 4.808e-05 | -1.473e-04 | 1.751e-04 |
| D3 | ranked | B3 | 6.715e-04 | 2.042e-05 | 0.0011 |
| D2 | cluster | A | 0.0042 | -3.406e-06 | 0.0083 |
| D2 | cluster | B1 | 0.0042 | -3.406e-06 | 0.0083 |
| D2 | cluster | B2 | 0.0042 | -3.406e-06 | 0.0083 |
| D2 | cluster | B3 | 0.0042 | -3.406e-06 | 0.0083 |
| D2 | ranked | A | 0.0042 | -3.406e-06 | 0.0083 |
| D2 | ranked | B1 | 0.0042 | -3.406e-06 | 0.0083 |
| D2 | ranked | B2 | 0.0042 | -3.406e-06 | 0.0083 |
| D2 | ranked | B3 | 0.0042 | -3.406e-06 | 0.0083 |
| D1 | cluster | A | -0.0024 | -0.0031 | -0.0013 |
| D1 | cluster | B1 | -0.0027 | -0.0041 | -0.0013 |
| D1 | cluster | B2 | -0.0023 | -0.0032 | -0.0013 |
| D1 | cluster | B3 | -0.0023 | -0.0032 | -0.0013 |
| D1 | ranked | A | -0.0024 | -0.0031 | -0.0013 |
| D1 | ranked | B1 | -0.0024 | -0.0031 | -0.0013 |
| D1 | ranked | B2 | -0.0023 | -0.0032 | -0.0013 |
| D1 | ranked | B3 | -0.0023 | -0.0032 | -0.0013 |

最终预测指标变化（主排名组，B区域，病例平均；正Dice/F1为增加，正CE为增加）：

| stage | method | full CE | boundary CE | hard Dice | soft Dice | contour F1 | changed fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E3 | A | -2.678e-05 | -3.915e-04 | 6.117e-06 | 6.035e-07 | 1.457e-05 | 1.183e-05 |
| E3 | B1 | -2.678e-05 | -3.915e-04 | 6.117e-06 | 6.035e-07 | 1.457e-05 | 1.183e-05 |
| E3 | B2 | -7.438e-05 | -0.0010 | 1.006e-05 | 6.810e-06 | 1.045e-04 | 2.068e-05 |
| E3 | B3 | -7.438e-05 | -0.0010 | 1.006e-05 | 6.810e-06 | 1.045e-04 | 2.068e-05 |
| E4 | A | 8.566e-07 | -2.079e-05 | -1.267e-07 | 1.416e-07 | 2.272e-06 | 1.993e-06 |
| E4 | B1 | 8.566e-07 | -2.079e-05 | -1.267e-07 | 1.416e-07 | 2.272e-06 | 1.993e-06 |
| E4 | B2 | -1.165e-06 | -5.439e-07 | 2.652e-06 | 1.155e-06 | 2.325e-05 | 3.612e-06 |
| E4 | B3 | 2.157e-06 | 2.205e-05 | 2.980e-08 | -6.407e-07 | 1.350e-05 | 3.239e-06 |
| D3 | A | -4.231e-05 | -7.007e-04 | -4.180e-06 | 1.043e-06 | -1.007e-05 | 2.005e-05 |
| D3 | B1 | -9.024e-05 | -0.0013 | 1.222e-06 | 8.740e-06 | 4.783e-05 | 2.715e-05 |
| D3 | B2 | -1.599e-05 | -1.548e-04 | 9.678e-06 | 1.136e-05 | -4.470e-08 | 1.769e-05 |
| D3 | B3 | 2.209e-05 | 4.686e-04 | 1.054e-05 | 5.551e-06 | 3.095e-05 | 2.616e-05 |
| D2 | A | 1.012e-04 | 0.0026 | 4.005e-05 | 2.023e-05 | 1.023e-04 | 2.078e-04 |
| D2 | B1 | 1.012e-04 | 0.0026 | 4.005e-05 | 2.023e-05 | 1.023e-04 | 2.078e-04 |
| D2 | B2 | 1.012e-04 | 0.0026 | 4.005e-05 | 2.023e-05 | 1.023e-04 | 2.078e-04 |
| D2 | B3 | 1.012e-04 | 0.0026 | 4.005e-05 | 2.023e-05 | 1.023e-04 | 2.078e-04 |
| D1 | A | -6.395e-04 | -0.0111 | 5.051e-06 | -8.494e-06 | 3.272e-05 | 5.356e-05 |
| D1 | B1 | -6.395e-04 | -0.0111 | 5.051e-06 | -8.494e-06 | 3.272e-05 | 5.356e-05 |
| D1 | B2 | -6.294e-04 | -0.0109 | 9.611e-06 | -8.702e-06 | 4.573e-05 | 3.413e-05 |
| D1 | B3 | -6.294e-04 | -0.0109 | 9.611e-06 | -8.702e-06 | 4.573e-05 | 3.413e-05 |

## B相对A的同病例消融影响差异

同K、同实际L2的B方法联合组Δboundary CE减去A联合组。正值表示B选中组的衰减产生更大边界CE变化，不是B训练后的性能收益。

| stage | kind | method | B-A | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| E3 | cluster | B1 | 7.274e-05 | -2.099e-04 | 3.651e-04 |
| E3 | cluster | B2 | -7.471e-04 | -0.0012 | -1.966e-04 |
| E3 | cluster | B3 | -7.471e-04 | -0.0012 | -1.966e-04 |
| E3 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| E3 | ranked | B2 | -6.266e-04 | -0.0010 | -1.814e-04 |
| E3 | ranked | B3 | -6.266e-04 | -0.0010 | -1.814e-04 |
| E4 | cluster | B1 | -1.324e-04 | -2.299e-04 | -3.947e-05 |
| E4 | cluster | B2 | -1.947e-05 | -6.212e-05 | 1.336e-05 |
| E4 | cluster | B3 | 1.455e-05 | -1.298e-05 | 4.745e-05 |
| E4 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| E4 | ranked | B2 | 2.024e-05 | -4.543e-05 | 6.746e-05 |
| E4 | ranked | B3 | 4.284e-05 | -3.479e-06 | 8.915e-05 |
| D3 | cluster | B1 | 0.0000 | 0.0000 | 0.0000 |
| D3 | cluster | B2 | 0.0010 | -7.795e-05 | 0.0022 |
| D3 | cluster | B3 | 6.908e-04 | -8.071e-05 | 0.0013 |
| D3 | ranked | B1 | -6.139e-04 | -0.0010 | -1.828e-04 |
| D3 | ranked | B2 | 5.459e-04 | -1.872e-04 | 0.0012 |
| D3 | ranked | B3 | 0.0012 | -4.096e-05 | 0.0021 |
| D2 | cluster | B1 | 0.0000 | 0.0000 | 0.0000 |
| D2 | cluster | B2 | 0.0000 | 0.0000 | 0.0000 |
| D2 | cluster | B3 | 0.0000 | 0.0000 | 0.0000 |
| D2 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| D2 | ranked | B2 | 0.0000 | 0.0000 | 0.0000 |
| D2 | ranked | B3 | 0.0000 | 0.0000 | 0.0000 |
| D1 | cluster | B1 | -2.839e-04 | -9.975e-04 | 8.672e-05 |
| D1 | cluster | B2 | 1.588e-04 | -1.468e-04 | 4.650e-04 |
| D1 | cluster | B3 | 1.588e-04 | -1.468e-04 | 4.650e-04 |
| D1 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| D1 | ranked | B2 | 1.588e-04 | -1.468e-04 | 4.650e-04 |
| D1 | ranked | B3 | 1.588e-04 | -1.468e-04 | 4.650e-04 |

补充影响幅值（|ΔCE|，不改变预先选定的通道）：

| stage | kind | method | |ΔCE| B-A | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| E3 | cluster | B1 | -1.109e-04 | -3.653e-04 | 1.021e-04 |
| E3 | cluster | B2 | 6.960e-04 | 2.048e-04 | 0.0011 |
| E3 | cluster | B3 | 6.960e-04 | 2.048e-04 | 0.0011 |
| E3 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| E3 | ranked | B2 | 6.418e-04 | 2.270e-04 | 0.0010 |
| E3 | ranked | B3 | 6.418e-04 | 2.270e-04 | 0.0010 |
| E4 | cluster | B1 | 1.206e-04 | 1.910e-05 | 2.299e-04 |
| E4 | cluster | B2 | 2.154e-05 | -1.699e-05 | 6.315e-05 |
| E4 | cluster | B3 | 1.891e-05 | -7.432e-06 | 4.512e-05 |
| E4 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| E4 | ranked | B2 | -4.730e-05 | -9.516e-05 | -3.368e-06 |
| E4 | ranked | B3 | -4.395e-05 | -1.008e-04 | 7.786e-07 |
| D3 | cluster | B1 | 0.0000 | 0.0000 | 0.0000 |
| D3 | cluster | B2 | -3.208e-05 | -2.462e-04 | 1.534e-04 |
| D3 | cluster | B3 | -6.910e-04 | -0.0010 | -2.362e-04 |
| D3 | ranked | B1 | 5.229e-04 | 8.941e-07 | 0.0010 |
| D3 | ranked | B2 | -5.874e-04 | -0.0010 | -1.427e-04 |
| D3 | ranked | B3 | -1.741e-04 | -3.504e-04 | 4.423e-05 |
| D2 | cluster | B1 | 0.0000 | 0.0000 | 0.0000 |
| D2 | cluster | B2 | 0.0000 | 0.0000 | 0.0000 |
| D2 | cluster | B3 | 0.0000 | 0.0000 | 0.0000 |
| D2 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| D2 | ranked | B2 | 0.0000 | 0.0000 | 0.0000 |
| D2 | ranked | B3 | 0.0000 | 0.0000 | 0.0000 |
| D1 | cluster | B1 | 2.839e-04 | -8.672e-05 | 9.975e-04 |
| D1 | cluster | B2 | -1.588e-04 | -4.650e-04 | 1.468e-04 |
| D1 | cluster | B3 | -1.588e-04 | -4.650e-04 | 1.468e-04 |
| D1 | ranked | B1 | 0.0000 | 0.0000 | 0.0000 |
| D1 | ranked | B2 | -1.588e-04 | -4.650e-04 | 1.468e-04 |
| D1 | ranked | B3 | -1.588e-04 | -4.650e-04 | 1.468e-04 |

## 真实边界 vs 区域负对照的因果探测

表为[(真实带干预−区域干预)的boundary CE影响]−随机集合的同类差分。单个正差分不建立边界专属性；还需跨病例重复并优于A。D1的1×1最终头是逐位置映射，非重叠内部区域干预在结构上不能改变边界带CE；不能把这种直接支持不重叠当作边界通道证据。

| stage | method | region control | difference | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| E3 | A | R | 1.971e-05 | -1.116e-04 | 1.693e-04 |
| E3 | A | I | 2.968e-05 | -1.501e-04 | 2.389e-04 |
| E3 | B1 | R | 1.971e-05 | -1.116e-04 | 1.693e-04 |
| E3 | B1 | I | 2.968e-05 | -1.501e-04 | 2.389e-04 |
| E3 | B2 | R | -1.060e-04 | -4.099e-04 | 7.456e-05 |
| E3 | B2 | I | -2.476e-04 | -6.057e-04 | 5.076e-07 |
| E3 | B3 | R | -1.060e-04 | -4.099e-04 | 7.456e-05 |
| E3 | B3 | I | -2.476e-04 | -6.057e-04 | 5.076e-07 |
| E4 | A | R | -2.019e-05 | -2.656e-05 | -1.001e-05 |
| E4 | A | I | -1.938e-05 | -3.157e-05 | -7.192e-06 |
| E4 | B1 | R | -2.019e-05 | -2.656e-05 | -1.001e-05 |
| E4 | B1 | I | -1.938e-05 | -3.157e-05 | -7.192e-06 |
| E4 | B2 | R | 1.027e-04 | 1.447e-05 | 1.909e-04 |
| E4 | B2 | I | 1.126e-04 | 1.729e-05 | 2.078e-04 |
| E4 | B3 | R | -4.233e-05 | -1.042e-04 | 1.952e-05 |
| E4 | B3 | I | -3.941e-05 | -1.113e-04 | 3.250e-05 |
| D3 | A | R | 2.022e-04 | 1.324e-04 | 2.724e-04 |
| D3 | A | I | 9.438e-05 | -9.449e-05 | 2.833e-04 |
| D3 | B1 | R | -2.479e-04 | -4.487e-04 | -4.711e-05 |
| D3 | B1 | I | -5.089e-04 | -7.953e-04 | -1.425e-04 |
| D3 | B2 | R | -1.985e-04 | -3.011e-04 | -9.598e-05 |
| D3 | B2 | I | -2.014e-04 | -2.803e-04 | -1.226e-04 |
| D3 | B3 | R | 8.824e-05 | -5.420e-06 | 1.703e-04 |
| D3 | B3 | I | 1.955e-04 | -2.366e-05 | 3.633e-04 |
| D2 | A | R | 0.0021 | -7.344e-04 | 0.0050 |
| D2 | A | I | 0.0030 | -5.552e-04 | 0.0065 |
| D2 | B1 | R | 0.0021 | -7.344e-04 | 0.0050 |
| D2 | B1 | I | 0.0030 | -5.552e-04 | 0.0065 |
| D2 | B2 | R | 0.0021 | -7.344e-04 | 0.0050 |
| D2 | B2 | I | 0.0030 | -5.552e-04 | 0.0065 |
| D2 | B3 | R | 0.0021 | -7.344e-04 | 0.0050 |
| D2 | B3 | I | 0.0030 | -5.552e-04 | 0.0065 |
| D1 | A | R | -0.0021 | -0.0026 | -0.0012 |
| D1 | A | I | -0.0024 | -0.0031 | -0.0013 |
| D1 | B1 | R | -0.0021 | -0.0026 | -0.0012 |
| D1 | B1 | I | -0.0024 | -0.0031 | -0.0013 |
| D1 | B2 | R | -0.0020 | -0.0027 | -0.0012 |
| D1 | B2 | I | -0.0023 | -0.0032 | -0.0013 |
| D1 | B3 | R | -0.0020 | -0.0027 | -0.0012 |
| D1 | B3 | I | -0.0023 | -0.0032 | -0.0013 |

## 全部单通道分数对最终边界CE影响的预测关系

发现集分数与probe病例逐通道消融影响的Spearman，先在各病例计算、再汇总4病例。正分不预设绝对梯度重要性；不事后取绝对值或翻转方向。

| stage | method | region contrast | rho mean | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| E3 | A | B | -0.0543 | -0.1374 | 0.0288 |
| E3 | A | B-R | 0.0233 | -0.0374 | 0.0841 |
| E3 | B1 | B | -0.0618 | -0.1110 | -0.0127 |
| E3 | B1 | B-R | -0.0189 | -0.1089 | 0.1098 |
| E3 | B2 | B | -0.1530 | -0.2220 | -0.0890 |
| E3 | B2 | B-R | -0.0105 | -0.0337 | 0.0202 |
| E3 | B3 | B | -0.1282 | -0.1710 | -0.0856 |
| E3 | B3 | B-R | 0.0020 | -0.0348 | 0.0299 |
| E4 | A | B | -0.1134 | -0.2036 | -0.0233 |
| E4 | A | B-R | -0.0493 | -0.1265 | 0.0089 |
| E4 | B1 | B | -0.1281 | -0.1333 | -0.1235 |
| E4 | B1 | B-R | -0.0986 | -0.1491 | -0.0396 |
| E4 | B2 | B | -0.1124 | -0.1929 | -0.0320 |
| E4 | B2 | B-R | 0.0229 | -0.0315 | 0.0773 |
| E4 | B3 | B | -0.1377 | -0.2100 | -0.0594 |
| E4 | B3 | B-R | 0.0177 | -0.0572 | 0.0952 |
| D3 | A | B | -0.1718 | -0.2973 | -0.0371 |
| D3 | A | B-R | 0.0296 | -0.0570 | 0.0853 |
| D3 | B1 | B | -0.1345 | -0.2454 | -0.0393 |
| D3 | B1 | B-R | 0.0891 | -0.0030 | 0.1529 |
| D3 | B2 | B | -0.0571 | -0.1265 | 0.0123 |
| D3 | B2 | B-R | 0.0416 | -0.0379 | 0.1211 |
| D3 | B3 | B | 0.0083 | -0.0950 | 0.1117 |
| D3 | B3 | B-R | 0.1161 | 0.0209 | 0.2113 |
| D2 | A | B | -0.3764 | -0.6088 | -0.2245 |
| D2 | A | B-R | -0.4483 | -0.6606 | -0.2361 |
| D2 | B1 | B | -0.2871 | -0.5701 | -0.1124 |
| D2 | B1 | B-R | -0.3562 | -0.6017 | -0.1107 |
| D2 | B2 | B | -0.3220 | -0.6255 | -0.1263 |
| D2 | B2 | B-R | -0.4095 | -0.6703 | -0.1488 |
| D2 | B3 | B | -0.2947 | -0.6058 | -0.1041 |
| D2 | B3 | B-R | -0.3686 | -0.6305 | -0.1067 |
| D1 | A | B | -0.8366 | -0.8834 | -0.7735 |
| D1 | A | B-R | -0.8331 | -0.8765 | -0.7713 |
| D1 | B1 | B | -0.8564 | -0.8981 | -0.7998 |
| D1 | B1 | B-R | -0.8493 | -0.8904 | -0.7926 |
| D1 | B2 | B | -0.8211 | -0.8766 | -0.7676 |
| D1 | B2 | B-R | -0.8272 | -0.8868 | -0.7676 |
| D1 | B3 | B | -0.8380 | -0.8864 | -0.7897 |
| D1 | B3 | B-R | -0.8412 | -0.8926 | -0.7897 |

补充描述性影响幅值与hard预测改变相关性，不用于重新选择通道：

| stage | method | endpoint | rho mean | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| E3 | A | changed_prediction_fraction | -0.0291 | -0.1944 | 0.1363 |
| E3 | A | abs_boundary_ce_delta | 0.1109 | -0.0850 | 0.2304 |
| E3 | B1 | changed_prediction_fraction | -0.0119 | -0.1909 | 0.1671 |
| E3 | B1 | abs_boundary_ce_delta | 0.0941 | -0.1257 | 0.2596 |
| E3 | B2 | changed_prediction_fraction | -0.0391 | -0.2406 | 0.1625 |
| E3 | B2 | abs_boundary_ce_delta | 0.1659 | -0.1143 | 0.3975 |
| E3 | B3 | changed_prediction_fraction | -0.0320 | -0.2544 | 0.1903 |
| E3 | B3 | abs_boundary_ce_delta | 0.1328 | -0.1300 | 0.3602 |
| E4 | A | changed_prediction_fraction | 0.0148 | -0.0582 | 0.0905 |
| E4 | A | abs_boundary_ce_delta | 0.0703 | 0.0085 | 0.1472 |
| E4 | B1 | changed_prediction_fraction | 0.0237 | -0.0676 | 0.1012 |
| E4 | B1 | abs_boundary_ce_delta | 0.0795 | 0.0423 | 0.1166 |
| E4 | B2 | changed_prediction_fraction | 0.1634 | 0.1483 | 0.1769 |
| E4 | B2 | abs_boundary_ce_delta | 0.1625 | 0.0805 | 0.2329 |
| E4 | B3 | changed_prediction_fraction | 0.1592 | 0.1349 | 0.1857 |
| E4 | B3 | abs_boundary_ce_delta | 0.1830 | 0.1118 | 0.2355 |
| D3 | A | changed_prediction_fraction | 0.0434 | -0.0297 | 0.1040 |
| D3 | A | abs_boundary_ce_delta | 0.0518 | -0.0474 | 0.1650 |
| D3 | B1 | changed_prediction_fraction | 0.0932 | 0.0148 | 0.1672 |
| D3 | B1 | abs_boundary_ce_delta | 0.0950 | -0.0042 | 0.2154 |
| D3 | B2 | changed_prediction_fraction | 0.0460 | 0.0024 | 0.0746 |
| D3 | B2 | abs_boundary_ce_delta | -0.0421 | -0.0939 | 0.0041 |
| D3 | B3 | changed_prediction_fraction | 0.1079 | 0.0552 | 0.1385 |
| D3 | B3 | abs_boundary_ce_delta | -0.0079 | -0.0520 | 0.0242 |
| D2 | A | changed_prediction_fraction | 0.6613 | 0.5868 | 0.7322 |
| D2 | A | abs_boundary_ce_delta | 0.7363 | 0.5154 | 0.8565 |
| D2 | B1 | changed_prediction_fraction | 0.6958 | 0.6219 | 0.7695 |
| D2 | B1 | abs_boundary_ce_delta | 0.6789 | 0.4173 | 0.8343 |
| D2 | B2 | changed_prediction_fraction | 0.6372 | 0.5643 | 0.7027 |
| D2 | B2 | abs_boundary_ce_delta | 0.6930 | 0.4521 | 0.8356 |
| D2 | B3 | changed_prediction_fraction | 0.6461 | 0.5712 | 0.7211 |
| D2 | B3 | abs_boundary_ce_delta | 0.6748 | 0.4106 | 0.8290 |
| D1 | A | changed_prediction_fraction | 0.4198 | 0.3903 | 0.4432 |
| D1 | A | abs_boundary_ce_delta | 0.8366 | 0.7735 | 0.8834 |
| D1 | B1 | changed_prediction_fraction | 0.4679 | 0.4428 | 0.4830 |
| D1 | B1 | abs_boundary_ce_delta | 0.8564 | 0.7998 | 0.8981 |
| D1 | B2 | changed_prediction_fraction | 0.3111 | 0.3046 | 0.3190 |
| D1 | B2 | abs_boundary_ce_delta | 0.8211 | 0.7676 | 0.8766 |
| D1 | B3 | changed_prediction_fraction | 0.3399 | 0.3327 | 0.3471 |
| D1 | B3 | abs_boundary_ce_delta | 0.8380 | 0.7897 | 0.8864 |

## 联合非加性

匹配singleton使用pair相同alpha；logit交互是zjoint−zsingle1−zsingle2+zbase的RMS。D1最终1×1头是线性映射，其CE非加性不代表通道协同。其他阶段的logit非加性同样需与随机组比较。

| stage | kind | method | metric | case mean |
| --- | --- | --- | --- | --- |
| E3 | cluster | A | boundary_ce_interaction | -8.754e-07 |
| E3 | cluster | A | logit_interaction_rms | 1.704e-05 |
| E3 | cluster | B1 | boundary_ce_interaction | -3.390e-07 |
| E3 | cluster | B1 | logit_interaction_rms | 1.389e-05 |
| E3 | cluster | B2 | boundary_ce_interaction | 1.118e-08 |
| E3 | cluster | B2 | logit_interaction_rms | 2.002e-05 |
| E3 | cluster | B3 | boundary_ce_interaction | 1.118e-08 |
| E3 | cluster | B3 | logit_interaction_rms | 2.002e-05 |
| E3 | ranked | A | boundary_ce_interaction | -3.390e-07 |
| E3 | ranked | A | logit_interaction_rms | 1.389e-05 |
| E3 | ranked | B1 | boundary_ce_interaction | -3.390e-07 |
| E3 | ranked | B1 | logit_interaction_rms | 1.389e-05 |
| E3 | ranked | B2 | boundary_ce_interaction | -3.725e-07 |
| E3 | ranked | B2 | logit_interaction_rms | 1.896e-05 |
| E3 | ranked | B3 | boundary_ce_interaction | -3.725e-07 |
| E3 | ranked | B3 | logit_interaction_rms | 1.896e-05 |
| E3 | random | random_mean | boundary_ce_interaction | -3.171e-07 |
| E3 | random | random_mean | logit_interaction_rms | 1.389e-05 |
| E4 | cluster | A | boundary_ce_interaction | -9.313e-08 |
| E4 | cluster | A | logit_interaction_rms | 4.024e-06 |
| E4 | cluster | B1 | boundary_ce_interaction | 1.267e-07 |
| E4 | cluster | B1 | logit_interaction_rms | 4.315e-06 |
| E4 | cluster | B2 | boundary_ce_interaction | -2.235e-08 |
| E4 | cluster | B2 | logit_interaction_rms | 3.268e-06 |
| E4 | cluster | B3 | boundary_ce_interaction | -3.725e-08 |
| E4 | cluster | B3 | logit_interaction_rms | 3.522e-06 |
| E4 | ranked | A | boundary_ce_interaction | -4.098e-08 |
| E4 | ranked | A | logit_interaction_rms | 3.066e-06 |
| E4 | ranked | B1 | boundary_ce_interaction | -4.098e-08 |
| E4 | ranked | B1 | logit_interaction_rms | 3.066e-06 |
| E4 | ranked | B2 | boundary_ce_interaction | -9.313e-08 |
| E4 | ranked | B2 | logit_interaction_rms | 3.400e-06 |
| E4 | ranked | B3 | boundary_ce_interaction | 1.863e-08 |
| E4 | ranked | B3 | logit_interaction_rms | 3.514e-06 |
| E4 | random | random_mean | boundary_ce_interaction | -4.703e-08 |
| E4 | random | random_mean | logit_interaction_rms | 3.347e-06 |
| D3 | cluster | A | boundary_ce_interaction | -4.917e-07 |
| D3 | cluster | A | logit_interaction_rms | 1.762e-05 |
| D3 | cluster | B1 | boundary_ce_interaction | -4.917e-07 |
| D3 | cluster | B1 | logit_interaction_rms | 1.762e-05 |
| D3 | cluster | B2 | boundary_ce_interaction | -1.501e-06 |
| D3 | cluster | B2 | logit_interaction_rms | 2.139e-05 |
| D3 | cluster | B3 | boundary_ce_interaction | 2.906e-07 |
| D3 | cluster | B3 | logit_interaction_rms | 3.087e-05 |
| D3 | ranked | A | boundary_ce_interaction | -4.917e-07 |
| D3 | ranked | A | logit_interaction_rms | 1.762e-05 |
| D3 | ranked | B1 | boundary_ce_interaction | -7.749e-07 |
| D3 | ranked | B1 | logit_interaction_rms | 2.859e-05 |
| D3 | ranked | B2 | boundary_ce_interaction | 1.416e-07 |
| D3 | ranked | B2 | logit_interaction_rms | 3.002e-05 |
| D3 | ranked | B3 | boundary_ce_interaction | -2.757e-07 |
| D3 | ranked | B3 | logit_interaction_rms | 3.350e-05 |
| D3 | random | random_mean | boundary_ce_interaction | -4.494e-07 |
| D3 | random | random_mean | logit_interaction_rms | 2.437e-05 |
| D2 | cluster | A | boundary_ce_interaction | 9.834e-05 |
| D2 | cluster | A | logit_interaction_rms | 2.091e-04 |
| D2 | cluster | B1 | boundary_ce_interaction | 9.834e-05 |
| D2 | cluster | B1 | logit_interaction_rms | 2.091e-04 |
| D2 | cluster | B2 | boundary_ce_interaction | 9.834e-05 |
| D2 | cluster | B2 | logit_interaction_rms | 2.091e-04 |
| D2 | cluster | B3 | boundary_ce_interaction | 9.834e-05 |
| D2 | cluster | B3 | logit_interaction_rms | 2.091e-04 |
| D2 | ranked | A | boundary_ce_interaction | 9.834e-05 |
| D2 | ranked | A | logit_interaction_rms | 2.091e-04 |
| D2 | ranked | B1 | boundary_ce_interaction | 9.834e-05 |
| D2 | ranked | B1 | logit_interaction_rms | 2.091e-04 |
| D2 | ranked | B2 | boundary_ce_interaction | 9.834e-05 |
| D2 | ranked | B2 | logit_interaction_rms | 2.091e-04 |
| D2 | ranked | B3 | boundary_ce_interaction | 9.834e-05 |
| D2 | ranked | B3 | logit_interaction_rms | 2.091e-04 |
| D2 | random | random_mean | boundary_ce_interaction | -3.446e-08 |
| D2 | random | random_mean | logit_interaction_rms | 4.343e-05 |
| D1 | cluster | A | boundary_ce_interaction | 1.322e-05 |
| D1 | cluster | A | logit_interaction_rms | 1.046e-07 |
| D1 | cluster | B1 | boundary_ce_interaction | 1.718e-05 |
| D1 | cluster | B1 | logit_interaction_rms | 1.086e-07 |
| D1 | cluster | B2 | boundary_ce_interaction | 8.401e-06 |
| D1 | cluster | B2 | logit_interaction_rms | 8.095e-08 |
| D1 | cluster | B3 | boundary_ce_interaction | 8.401e-06 |
| D1 | cluster | B3 | logit_interaction_rms | 8.095e-08 |
| D1 | ranked | A | boundary_ce_interaction | 1.322e-05 |
| D1 | ranked | A | logit_interaction_rms | 1.046e-07 |
| D1 | ranked | B1 | boundary_ce_interaction | 1.322e-05 |
| D1 | ranked | B1 | logit_interaction_rms | 1.046e-07 |
| D1 | ranked | B2 | boundary_ce_interaction | 8.401e-06 |
| D1 | ranked | B2 | logit_interaction_rms | 8.095e-08 |
| D1 | ranked | B3 | boundary_ce_interaction | 8.401e-06 |
| D1 | ranked | B3 | logit_interaction_rms | 8.095e-08 |
| D1 | random | random_mean | boundary_ce_interaction | 5.384e-06 |
| D1 | random | random_mean | logit_interaction_rms | 4.714e-08 |

## LossNorm与执行

B原始梯度没有独立LossNorm。次要A-denominator贡献和余弦仅用相同crop原A归一化因子。loss_scaling.json保存原损失与固定A因子范围；expanded_channel_relations.csv保留重新加权前后方向。量纲不同，不按原始范数直接断言B优于A。

### extraction_results.json

```json
{
  "status": "PASS",
  "smoke": false,
  "invariants": {
    "student_state_unchanged": true,
    "teacher_state_unchanged": true,
    "grad_slots_unchanged": true,
    "teacher_frozen": true,
    "both_eval": true,
    "torch_rng_unchanged": true,
    "cuda_rng_unchanged": true,
    "unlabeled_image_only": true,
    "data_stats_unchanged": true
  },
  "objective_scale_observations": 1088,
  "raw_observations": 48,
  "optimizer_steps": 0,
  "checkpoint_writes": 0,
  "elapsed_seconds": 171.29296098393388,
  "peak_allocated_mib": 2102.8974609375
}
```

### analysis_results.json

```json
{
  "status": "PASS",
  "method_stage_scale_cells": 612,
  "channel_relation_rows": 47056,
  "intervention_groups": 80,
  "elapsed_seconds": 282.6258435321506
}
```

### ablation_results.json

```json
{
  "status": "PASS",
  "smoke": false,
  "checks": {
    "student_unchanged": true,
    "teacher_unchanged": true,
    "no_parameter_grads": true,
    "cpu_rng_unchanged": true,
    "cuda_rng_unchanged": true,
    "eval": true,
    "only_labeled_probe_cases": true,
    "no_discovery_overlap": true,
    "no_unlabeled_access": true,
    "hooks_removed": true,
    "batch_baseline_stable": true
  },
  "labeled_cases": 4,
  "crops": 8,
  "singleton_rows": 7296,
  "group_rows": 1920,
  "batch_vs_single_max_logit_error": 0.0,
  "elapsed_seconds": 272.9753223969601,
  "peak_mib": 1761.482421875,
  "target": "Final D1 logits/predictions; no auxiliary-head outcomes",
  "optimizer_steps": 0,
  "checkpoint_writes": 0
}
```

### summary_results.json

```json
{
  "status": "PASS",
  "probe_cases": 4,
  "singleton_invalid": 1,
  "group_invalid": 0,
  "bootstrap": "1000 resamples of 4 probe cases; crops/groups remain within case",
  "supplemental_endpoints": "Absolute boundary CE delta and score-vs-prediction-change are descriptive sensitivity endpoints, not selection criteria",
  "all_results_exploratory": true,
  "not_independent_generalization": true
}
```

### verification_results.json

```json
{
  "status": "PASS",
  "checks": {
    "checkpoint_and_cases_match_A": true,
    "parameter_mapping_exact_A": true,
    "1088_unique_objective_scale_records": true,
    "native_class_region_mass_conserved": true,
    "valid_terms_have_effective_coverage": true,
    "controls_exact_native_voxel_and_class_counts": true,
    "raw_parameter_gradient_shapes_correct": true,
    "reachable_valid_gradients_finite": true,
    "unreachable_or_invalid_nan_not_fake_zero": true,
    "612_method_stage_head_cells": true,
    "matrix_graph_NA_matches_real_reachability": true,
    "independently_recomputed_LU_cells_match": true,
    "extraction_results.json_pass": true,
    "extraction_results.json_no_updates": true,
    "ablation_results.json_pass": true,
    "ablation_results.json_no_updates": true,
    "U_image_only": true,
    "joint_K_exactly2": true,
    "group_l2_matched": true,
    "component_l2_conserved": true,
    "singleton_l2_matched": true,
    "alpha_cap_respected": true,
    "probe_disjoint_from_discovery": true,
    "only_probe_cases_ablated": true,
    "each_detailed_stage_has_final_prediction_effect": true,
    "historical_files_sha256_unchanged": true,
    "unit_tests_pass": true,
    "stage0_stage1a_core_regressions_pass": true
  },
  "protection": {
    "status": "PASS",
    "checked_files": 500,
    "changed": []
  },
  "gradient_observation_statuses": {
    "valid": 752699,
    "unreachable": 47872,
    "exact_zero": 7,
    "below_threshold": 190
  },
  "finite_raw_gradient_elements": 2906840064,
  "paired_cosine_independent_spotchecks": [
    {
      "stage": "E3",
      "boundary_vs_seg_same_observation_median": 0.9749673632997444
    },
    {
      "stage": "D1",
      "boundary_vs_seg_same_observation_median": 0.9991624756365443
    }
  ],
  "independently_recomputed_LU_cells": [
    {
      "stage": "E3",
      "method": "A",
      "saved": 0.17464790158825294,
      "independent": 0.174647901588253,
      "match": true
    },
    {
      "stage": "E3",
      "method": "B3_w2",
      "saved": 0.27036353833053384,
      "independent": 0.27036353833053384,
      "match": true
    },
    {
      "stage": "D1",
      "method": "A",
      "saved": 0.45449455597451716,
      "independent": 0.4544945559745172,
      "match": true
    },
    {
      "stage": "D1",
      "method": "B3_w2",
      "saved": 0.9375353411454443,
      "independent": 0.9375353411454446,
      "match": true
    }
  ],
  "limitations": [
    "HDF5 size/mtime checked by extraction; contents not hashed/read for protection.",
    "U ground truth is never opened; no U boundary accuracy claim.",
    "No feature-map gradients generated: all saved gradient vectors are convolution parameter coordinates."
  ]
}
```

## 解释限制

所有L病例参与过Baseline训练，probe只是未用于通道选择，不能称独立泛化。仅1个checkpoint/1个数据划分、2个crop/病例、4个probe L病例、8组随机K=2集合、单一干预预算。没有读取U GT，因此不能宣称Teacher边界正确或U边界准确率改善。没有增强逆变换、完整体积干预或新损失训练。全部发现应先在独立标注验证病例、多checkpoint及多标注seed复核。
