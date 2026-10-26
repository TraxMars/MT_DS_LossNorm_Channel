"""Summarize full kernel gradients, shared-coordinate case stability and profiles."""

import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from core import SCALES, STAGES, gradient_validity, signed_cosine
from stats_tools import (bootstrap_pair_mean, case_means, describe, masked_mean,
                         observation_patterns, pairwise_cosine, profile_pca_and_clusters)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def cv(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1) / a.mean()) if len(a) > 1 and a.mean() > 0 else None


def within_case_cosines(array, scale_index, repeats, absolute_floor, relative_floor):
    result = []
    for repeat_a, repeat_b in itertools.combinations(range(repeats), 2):
        scores, _, _, _ = signed_cosine(array[repeat_a::repeats, scale_index],
                                         array[repeat_b::repeats, scale_index], absolute_floor, relative_floor)
        result.append(scores.T)
    return np.concatenate(result, axis=1)


def pool_between_case(case_vectors, absolute_floor, relative_floor):
    values, _, _, _ = pairwise_cosine(case_vectors, case_vectors, absolute_floor, relative_floor)
    a, b = np.triu_indices(len(case_vectors), 1)
    return values[:, a, b]


def heatmap(ax, values, title, vmin, vmax, cmap="coolwarm"):
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#d8d8d8")
    im = ax.imshow(np.ma.masked_invalid(np.array(values, dtype=float)), vmin=vmin, vmax=vmax,
                   cmap=colors, aspect="auto")
    ax.set_xticks(range(4), SCALES)
    ax.set_yticks(range(9), STAGES)
    ax.set_title(title, fontsize=10)
    for a in range(9):
        for b in range(4):
            value = values[a][b]
            if value is not None and np.isfinite(value):
                ax.text(b, a, f"{value:.2f}", ha="center", va="center", fontsize=7)
    return im


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_signed_d1(channel_rows, directory):
    rows = sorted((row for row in channel_rows if row["stage"]=="D1" and row["scale"]=="D1"),
                  key=lambda row:int(row["channel"]))
    fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
    colors = {"positive":"#c45b22", "negative":"#2765a0", "uncertain":"#777777"}
    for row in rows:
        channel = int(row["channel"])
        ax.vlines(channel,float(row["lu_case_bootstrap_ci_lower"]),float(row["lu_case_bootstrap_ci_upper"]),
                  color=colors[row["stable_sign"]],linewidth=1.6)
        ax.scatter(channel,float(row["lu_raw_mean"]),color=colors[row["stable_sign"]],s=32,zorder=3)
    ax.axhline(0,color="black",linewidth=.8,linestyle="--")
    ax.set_xticks(range(16))
    ax.set_ylim(-1.05,1.05)
    ax.set_xlabel("D1 output-channel index")
    ax.set_ylabel("Signed L/U kernel-gradient cosine")
    ax.set_title("D1 head / D1 parameters: case-pair mean and 95% case-bootstrap interval")
    save_figure(fig,directory,"d1_signed_channel_cosines")


def format_number(value):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.3f}"


def report_markdown(output, design, summary, pca_results, extraction, elapsed):
    rows = {row["stage"]+row["scale"]: row for row in summary}
    with (output / "per_observation_channels.csv").open() as handle:
        norm_counts = Counter(row["status"] for row in csv.DictReader(handle))
    text = ["# 阶段 1A：冻结模型多尺度参数梯度诊断", "",
            f"LA 10%，同一 last_checkpoint 中的 Student/Teacher，iteration={design['iteration']}；"
            f"{len(design['labeled_ids'])} 个标注病例、{len(design['unlabeled_ids'])} 个无标签病例，每病例 {design['repeats']} 个完整 112×112×80 裁块。", "",
            "两个模型始终 eval。Student 保留 autograd，但不创建 optimizer；Teacher 冻结并使用 no_grad。"
            "无标签数据只读取 image，逐键访问日志和状态保护检查见本目录 JSON。", "",
            "## 九阶段 × 四尺度：有符号 L/U 余弦", "",
            "每格为具有至少 75% 有效 L/U 病例对的通道，其病例对平均余弦的通道中位数。"
            "L/U 是不同病例群体，比较的是同一权重坐标，病例对没有解剖对应或空间配准含义。NA 不代表零相似度。", "",
            "| 阶段 | D4 | D3 | D2 | D1 |", "| --- | --- | --- | --- | --- |"]
    for stage in STAGES:
        text.append("| "+stage+" | "+" | ".join(format_number(rows[stage+s].get("lu_mean_channel_median")) for s in SCALES)+" |")
    text += ["", "## 有效性与病例稳定性", "",
             "| 阶段/尺度 | 有效病例对比例 | 可靠通道/总通道 | 同病例重复 L/U 中位余弦 | 跨病例 L/U 中位余弦 | L/U 正/负稳定符号通道 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in summary:
        if not row["graph_reachable"]:
            continue
        text.append(f"| {row['stage']}/{row['scale']} | {row['valid_pair_fraction']:.3f} | "
                    f"{row['reliable_channels']}/{row['channels']} | "
                    f"{format_number(row['within_L_median'])}/{format_number(row['within_U_median'])} | "
                    f"{format_number(row['between_L_median'])}/{format_number(row['between_U_median'])} | "
                    f"{row['stable_positive_channels']}/{row['stable_negative_channels']} |")
    reachable = [r for r in summary if r["graph_reachable"] and r.get("within_U_median") is not None]
    ordered = sorted(reachable, key=lambda r:r["within_U_median"], reverse=True)
    text += ["", "按无标签同病例重复方向的一致性排序，较高的阶段/尺度为："+
             "、".join(f"{r['stage']}/{r['scale']} ({r['within_U_median']:.3f})" for r in ordered[:5])+"。",
             "较低的为："+"、".join(f"{r['stage']}/{r['scale']} ({r['within_U_median']:.3f})" for r in ordered[-5:])+"。", "",
             "以上是当前病例样本与裁块设计下的描述性排序，不是通道选择规则。完整跨病例、同病例、"
             "病例 bootstrap 区间、低梯度排除原因见 per_channel.csv 和 validity_matrices.json。"
             "8×16 的病例对共享病例，区间使用分别重采样 L/U 病例的 200 次 bootstrap，不能按独立 128 对解释。", "",
             "图不可达固定为：D3←D4、D2←D4/D3、D1←D4/D3/D2 的头损失到对应阶段参数。"
             "其中箭头指损失反向到参数；全部编码器及 D4 对四个头均可达。"
             "可达但为零、非有限或低于 max(1e-12,1e-6×当前参数各通道最大范数) 的梯度，均不生成有效余弦。", "",
             f"逐裁块范数观测：有效 {norm_counts['valid']} 个、结构不可达 {norm_counts['unreachable']} 个、"
             f"低于阈值 {norm_counts['below_threshold']} 个、精确零 {norm_counts['exact_zero']} 个、非有限 {norm_counts['nonfinite']} 个。"
             "病例均值达到覆盖标准不等于每个原始裁块观测都有效；低于阈值的观测保留原始范数，但其相似度已屏蔽。", "",
             "## 跨尺度、PCA 与分组", "",
             "cross_scale_summary.csv/图记录同一病例同一阶段参数上不同头梯度的有符号余弦。"
             "per_channel_multiscale.csv 记录梯度方向 Gram 矩阵 SVD 的有效秩及梯度求和抵消比。", "",
             "通道分组仅使用 L/U 各自按病例平均的跨尺度梯度范数占比，并要求两域至少 75% 裁块全尺度梯度有效。"
             "不把图不可达项补零后聚类。norm-share 各自和为 1，因此 PCA 秩上限天然为 2×(可达头数−1)。", "",
             "| 阶段 | QC 有效通道 | PCA 实测秩/代数上限 | PC1/PC1+PC2 方差 | k=2 病例 bootstrap ARI 均值 | 状态 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for stage, result in pca_results.items():
        variance = result.get("explained_variance_ratio", [])
        first = variance[0] if variance else None
        two = sum(variance[:2]) if variance else None
        text.append(f"| {stage} | {len(result['eligible_channels'])} | {result.get('rank','NA')}/"
                    f"{result['algebraic_rank_upper_bound']} | {format_number(first)}/{format_number(two)} | "
                    f"{format_number(result.get('k2_case_bootstrap_ARI',{}).get('mean'))} | {result['status']} |")
    text += ["", "k=2/3/4 均为探索性候选；k=2 的病例 bootstrap 为 100 次，报告 ARI，不把它变成训练时固定分组。"
             "不同通道的相似、相反或低维范数响应只能描述梯度几何，不能证明通道协同或互补。"
             "D1 只有一个可达头，占比恒为 1，PCA/聚类无解释空间。", "",
             "## 原始梯度与 LossNorm", "",
             "原始项 g_s=∂L_s/∂W。原监督目标为四尺度平均 Lsup，训练中该头贡献为 "
             "g_s/[4·max(stopgrad(Lsup),eps)]；MT 还乘 0.3。没有对单尺度各自做 L_s/L_s.detach()。"
             "实际聚合梯度与四独立梯度的重构误差记录于 gradient_reconstruction.json；"
             "跨头梯度相加与一次反传存在浮点加法顺序差异，按全向量相对 L2 误差 ≤1e-4（或绝对 L2≤1e-9）验证。", "",
             "同一裁块、同一尺度的正标量归一化不会改变有效余弦，但改变梯度幅值。"
             "跨裁块先归一化再平均会重新加权各裁块，因此病例均值的 raw/normalized 方向可能不同。"
             "两种 L/U 余弦和差值均保留在 per_channel.csv。norm_scaling.json 记录幅值缩放范围。", "",
             "## 能否称为边界相关梯度", "",
             "**不能。当前梯度只来自普通区域分割 CE+Dice 与 Student–Teacher 概率 MSE，"
             "没有显式边界目标，也没有测量它与真实轮廓或距离场的关联。**"
             "预测不确定性、梯度冲突或低维分组不等于边界证据。", "",
             "后续应先定义独立的可证伪诊断：仅在有标签病例从 GT 构造窄带轮廓或有符号距离场，"
             "预先确定各尺度边界的重采样和带宽，再定义边界平衡 BCE+Dice 或距离回归目标。"
             "届时对同一病例、同一权重坐标比较边界目标与区域/MT 目标的梯度，并使用等体积随机带、"
             "平移/打乱轮廓、区域内部与背景目标作对照，检查差异是否在留出病例、多个裁块、模型迭代及标注种子上重复。"
             "只有关联诊断成立，才值得另外设计干预/消融来检验协同或性能因果作用；本阶段不实现这些目标或训练。", "",
             "## 验证与限制", "",
             f"提取 {extraction['observations']} 个裁块，耗时 {extraction['elapsed_seconds']:.3f} 秒；"
             f"峰值已分配显存 {extraction['peak_allocated_mib']:.1f} MiB。"
             f"{extraction['protected_files']} 个既有文件哈希保护通过；模型参数、BN buffers、checkpoint、.grad 槽及 eval RNG 均未变。", "",
             "batch_size=1 是病例级诊断；Dice 在 batch 内全局求和，不能把本结果宣称为还原原 batch_size=2 的每一步训练梯度。"
             "eval 禁用 Dropout 且使用保存的 BN 统计，结论也不等于 train-mode 梯度关系。"
             "仅单一 LA 标注划分、迭代及预选 16/72 个无标签病例，8 个标注病例内无独立验证集。"
             "两个裁块不足以穷尽每个病例空间差异；没有做跨病例特征图坐标逐点比较。", "",
             f"统计与作图耗时 {elapsed:.3f} 秒。输出图提供 PNG/PDF，完整原始参数梯度保存在 raw/ 的 NPY 文件。"]
    (output / "REPORT.md").write_text("\n".join(text)+"\n")


def summarize(output, bootstrap_draws=200):
    started = time.monotonic()
    extraction = json.loads((output / "extraction_results.json").read_text())
    if extraction["status"] != "PASS":
        raise ValueError("Only a completed successful extraction can be summarized")
    design = json.loads((output / "design.json").read_text())
    observations = json.loads((output / "observations.json").read_text())
    mapping = json.loads((output / "parameter_mapping.json").read_text())
    repeats, absolute, relative = design["repeats"], design["absolute_gradient_floor"], design["relative_gradient_floor"]
    factors = {domain: np.array([row["normalized_scale_factor"] for row in observations[domain]]) for domain in ("L", "U")}
    channel_rows, summaries, cross_rows, multiscale_rows, profile_rows = [], [], [], [], []
    pca_results, pca_plots = {}, {}
    matrix = {key: np.full((9,4), np.nan) for key in ("reachable", "valid_fraction", "lu", "within_L", "within_U", "between_L", "between_U", "norm_L", "norm_U")}
    cross_matrices = {domain: np.full((9,4,4), np.nan) for domain in ("L", "U")}
    with (output / "lu_case_pairs.csv").open("w", newline="") as pair_handle, \
         (output / "per_observation_cross_scale.csv").open("w", newline="") as cross_handle:
        pair_writer = csv.DictWriter(pair_handle, lineterminator="\n", fieldnames=("stage","scale","channel","labeled_case","unlabeled_case",
                                                             "raw_cosine","normalized_cosine","valid","L_status","U_status"))
        pair_writer.writeheader()
        cross_writer = csv.DictWriter(cross_handle, lineterminator="\n", fieldnames=("domain","case","repeat","stage","channel","scale_a","scale_b","cosine","status"))
        cross_writer.writeheader()
        for stage_index, stage in enumerate(STAGES):
            spec = mapping[stage]
            channels = spec["channels"]
            reachable = [SCALES.index(scale) for scale in spec["reachable_scales"]]
            arrays = {domain: np.load(output / "raw" / f"{domain}_{stage}.npy", mmap_mode="r") for domain in ("L","U")}
            case_raw = {domain: case_means(array, repeats) for domain, array in arrays.items()}
            case_normalized = {domain: case_means(array, repeats, factors[domain]) for domain, array in arrays.items()}
            profiles_by_case, coverage = {}, {}
            for domain, array in arrays.items():
                fractions, cross, rank, cancellation = observation_patterns(array, reachable, absolute, relative)
                profiles_by_case[domain] = masked_mean(fractions.reshape(-1,repeats,channels,len(reachable)), axis=1)
                coverage[domain] = np.isfinite(fractions).all(axis=2).mean(axis=0)
                for channel in range(channels):
                    multiscale_rows.append(dict(domain=domain,stage=stage,channel=channel,
                                                complete_valid_observations=int(np.isfinite(rank[:,channel]).sum()),
                                                total_observations=len(array), reachable_heads=len(reachable),
                                                direction_effective_rank_median=describe(rank[:,channel])["median"],
                                                cancellation_ratio_median=describe(cancellation[:,channel])["median"]))
                for a in range(4):
                    for b in range(4):
                        channel_medians = np.array([describe(cross[:,channel,a,b])["median"]
                                                    if describe(cross[:,channel,a,b])["n"] else np.nan for channel in range(channels)])
                        cross_matrices[domain][stage_index,a,b] = describe(channel_medians)["median"] \
                            if describe(channel_medians)["n"] else np.nan
                        if a < b:
                            description = describe(channel_medians)
                            cross_rows.append(dict(domain=domain,stage=stage,scale_a=SCALES[a],scale_b=SCALES[b],
                                                   graph_reachable=a in reachable and b in reachable,
                                                   valid_channels=description["n"],cosine_channel_median=description["median"],
                                                   cosine_channel_q10=description["q10"],cosine_channel_q90=description["q90"]))
                            for obs, observation in enumerate(observations[domain]):
                                for channel in range(channels):
                                    value = cross[obs,channel,a,b]
                                    cross_writer.writerow(clean(dict(domain=domain,case=observation["case"],repeat=observation["repeat"],
                                                                      stage=stage,channel=channel,scale_a=SCALES[a],scale_b=SCALES[b],cosine=value,
                                                                      status="valid" if np.isfinite(value) else (
                                                                          "unreachable" if a not in reachable or b not in reachable else "zero_tiny_or_nonfinite"))))
            result, profiles, scores, labels = profile_pca_and_clusters(
                profiles_by_case["L"],profiles_by_case["U"],coverage["L"],coverage["U"],seed=design["seed"])
            pca_results[stage] = result
            pca_plots[stage] = (scores, labels)
            eligible = set(result["eligible_channels"])
            lookup = {channel:index for index,channel in enumerate(result["eligible_channels"])}
            for channel in range(channels):
                row = dict(stage=stage,channel=channel,coverage_L=coverage["L"][channel],coverage_U=coverage["U"][channel],
                           eligible_for_descriptive_pca=channel in eligible,
                           k2_cluster=int(labels[lookup[channel]]) if labels is not None and channel in eligible else None)
                for index, scale_index in enumerate(reachable):
                    row[f"L_share_{SCALES[scale_index]}"] = profiles[channel,index]
                    row[f"U_share_{SCALES[scale_index]}"] = profiles[channel,len(reachable)+index]
                profile_rows.append(row)
            for scale_index, scale in enumerate(SCALES):
                is_reachable = scale_index in reachable
                matrix["reachable"][stage_index,scale_index] = int(is_reachable)
                if not is_reachable:
                    summaries.append(dict(stage=stage,scale=scale,graph_reachable=False,channels=channels,reason="graph_unreachable"))
                    channel_rows.extend(dict(stage=stage,scale=scale,channel=c,status="graph_unreachable") for c in range(channels))
                    continue
                left, right = case_raw["L"][:,scale_index], case_raw["U"][:,scale_index]
                cosines, valid_pairs, left_status, right_status = pairwise_cosine(left,right,absolute,relative)
                normalized_cosines, _, _, _ = pairwise_cosine(case_normalized["L"][:,scale_index],
                                                              case_normalized["U"][:,scale_index],absolute,relative)
                intervals = bootstrap_pair_mean(cosines,seed=design["seed"],draws=bootstrap_draws)
                within = {domain: within_case_cosines(array,scale_index,repeats,absolute,relative) for domain,array in arrays.items()}
                between = {domain: pool_between_case(case_raw[domain][:,scale_index],absolute,relative) for domain in ("L","U")}
                norms = {domain: gradient_validity(case_raw[domain][:,scale_index],absolute,relative)[0] for domain in ("L","U")}
                local = []
                for channel in range(channels):
                    description = describe(cosines[channel])
                    normalized_description = describe(normalized_cosines[channel])
                    valid_fraction = float(valid_pairs[channel].mean())
                    reliable = valid_fraction >= .75
                    sign = "uncertain"
                    if reliable and intervals[0,channel] > 0:
                        sign = "positive"
                    elif reliable and intervals[1,channel] < 0:
                        sign = "negative"
                    status = "valid" if reliable else ("no_valid_L_gradient" if not (left_status[:,channel]=="valid").any()
                                                        else "no_valid_U_gradient" if not (right_status[:,channel]=="valid").any()
                                                        else "insufficient_valid_pair_coverage")
                    row = dict(stage=stage,scale=scale,channel=channel,status=status,valid_pair_fraction=valid_fraction,
                               valid_pairs=description["n"],total_pairs=cosines.shape[1]*cosines.shape[2],
                               lu_raw_mean=description["mean"],lu_raw_median=description["median"],
                               lu_positive_fraction=description["positive_fraction"],
                               lu_case_bootstrap_ci_lower=intervals[0,channel],lu_case_bootstrap_ci_upper=intervals[1,channel],
                               stable_sign=sign,lu_normalized_mean=normalized_description["mean"],
                               raw_normalized_mean_abs_difference=(abs(description["mean"]-normalized_description["mean"])
                                                                  if description["mean"] is not None and normalized_description["mean"] is not None else None))
                    for domain in ("L","U"):
                        row[f"raw_norm_median_{domain}"] = describe(norms[domain][:,channel])["median"]
                        row[f"raw_norm_case_CV_{domain}"] = cv(norms[domain][:,channel])
                        row[f"within_{domain}_median"] = describe(within[domain][channel])["median"]
                        row[f"between_{domain}_median"] = describe(between[domain][channel])["median"]
                        row[f"zero_case_count_{domain}"] = int(np.sum((left_status if domain=="L" else right_status)[:,channel]=="exact_zero"))
                        row[f"tiny_case_count_{domain}"] = int(np.sum((left_status if domain=="L" else right_status)[:,channel]=="below_threshold"))
                    local.append(row)
                    for a, labeled_case in enumerate(design["labeled_ids"]):
                        for b, unlabeled_case in enumerate(design["unlabeled_ids"]):
                            pair_writer.writerow(clean(dict(stage=stage,scale=scale,channel=channel,labeled_case=labeled_case,
                                                            unlabeled_case=unlabeled_case,raw_cosine=cosines[channel,a,b],
                                                            normalized_cosine=normalized_cosines[channel,a,b],valid=bool(valid_pairs[channel,a,b]),
                                                            L_status=left_status[a,channel],U_status=right_status[b,channel])))
                channel_rows.extend(local)
                reliable_rows = [row for row in local if row["status"]=="valid"]
                def median_key(key):
                    return describe([row[key] if row[key] is not None else np.nan for row in reliable_rows])["median"]
                summary = dict(stage=stage,scale=scale,graph_reachable=True,channels=channels,
                               valid_pair_fraction=float(valid_pairs.mean()),reliable_channels=len(reliable_rows),
                               lu_mean_channel_median=median_key("lu_raw_mean"),
                               within_L_median=median_key("within_L_median"),within_U_median=median_key("within_U_median"),
                               between_L_median=median_key("between_L_median"),between_U_median=median_key("between_U_median"),
                               raw_norm_median_L=median_key("raw_norm_median_L"),raw_norm_median_U=median_key("raw_norm_median_U"),
                               stable_positive_channels=sum(row["stable_sign"]=="positive" for row in reliable_rows),
                               stable_negative_channels=sum(row["stable_sign"]=="negative" for row in reliable_rows),
                               raw_normalized_cosine_difference_median=median_key("raw_normalized_mean_abs_difference"))
                summaries.append(summary)
                for key in ("within_L","within_U","between_L","between_U"):
                    matrix[key][stage_index,scale_index] = summary[key+"_median"]
                matrix["valid_fraction"][stage_index,scale_index] = summary["valid_pair_fraction"]
                matrix["lu"][stage_index,scale_index] = summary["lu_mean_channel_median"]
                for domain in ("L","U"):
                    value = summary["raw_norm_median_"+domain]
                    matrix["norm_"+domain][stage_index,scale_index] = np.log10(value) if value is not None and value>0 else np.nan
            print(f"SUMMARIZED {stage}: PCA {result['status']}; valid channels {len(result['eligible_channels'])}/{channels}",flush=True)
            del arrays,case_raw,case_normalized
    write_csv(output/"per_channel.csv",channel_rows)
    write_csv(output/"scale_summary.csv",summaries)
    write_csv(output/"cross_scale_summary.csv",cross_rows)
    write_csv(output/"per_channel_multiscale.csv",multiscale_rows)
    write_csv(output/"channel_profiles.csv",profile_rows)
    write_json(output/"pca_clusters.json",pca_results)
    write_json(output/"validity_matrices.json",dict(stages=STAGES,scales=SCALES,matrices=matrix,
                                                   invalid_policy="null/gray means unreachable or no reliable score; never imputed as zero"))
    scaling = {domain:dict(normalized_scale_factor=describe(factors[domain]),
                           relative_to_raw_mean_contribution=describe(factors[domain]*4),
                           aggregate_loss=describe([row["aggregate_loss"] for row in observations[domain]])) for domain in ("L","U")}
    write_json(output/"norm_scaling.json",scaling)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,6),constrained_layout=True)
    for ax,key,title,limits,cmap in zip(axes,("reachable","valid_fraction","lu"),
                                        ("Graph reachability","Valid L/U case-pair fraction","Median signed L/U cosine"),
                                        ((0,1),(0,1),(-1,1)),("Blues","Blues","coolwarm")):
        im=heatmap(ax,matrix[key],title,*limits,cmap=cmap);fig.colorbar(im,ax=ax,shrink=.7)
    save_figure(fig,figures,"validity_and_lu")
    fig,axes=plt.subplots(2,2,figsize=(10,11),constrained_layout=True)
    for ax,key in zip(axes.flat,("within_L","within_U","between_L","between_U")):
        im=heatmap(ax,matrix[key],key.replace("_"," ")+" signed cosine",-1,1);fig.colorbar(im,ax=ax,shrink=.8)
    save_figure(fig,figures,"case_stability")
    fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
    for ax,stage in zip(axes.flat,STAGES):
        scores,labels=pca_plots[stage];result=pca_results[stage]
        if scores is None:
            ax.text(.5,.5,result["status"],ha="center",va="center",transform=ax.transAxes,fontsize=8)
        else:
            ax.scatter(scores[:,0],scores[:,1],c=labels,cmap="viridis",s=12,alpha=.8)
        ax.set_title(f"{stage}: {len(result['eligible_channels'])} channels; rank {result.get('rank','NA')}")
        ax.set_xlabel("PC1");ax.set_ylabel("PC2")
    save_figure(fig,figures,"channel_profile_pca")
    fig,axes=plt.subplots(9,2,figsize=(8,25),constrained_layout=True)
    for index,stage in enumerate(STAGES):
        for col,domain in enumerate(("L","U")):
            ax=axes[index,col];colors=plt.get_cmap("coolwarm").copy();colors.set_bad("#d8d8d8")
            ax.imshow(np.ma.masked_invalid(cross_matrices[domain][index]),vmin=-1,vmax=1,cmap=colors)
            ax.set_xticks(range(4),SCALES);ax.set_yticks(range(4),SCALES);ax.set_title(f"{stage} {domain}")
    save_figure(fig,figures,"cross_scale_cosines")
    plot_signed_d1(channel_rows,figures)
    elapsed=time.monotonic()-started
    report_markdown(output,design,summaries,pca_results,extraction,elapsed)
    result=dict(status="PASS",elapsed_seconds=elapsed,stage_scale_cells=36,structurally_reachable_cells=sum(row["graph_reachable"] for row in summaries),
                per_channel_rows=len(channel_rows),bootstrap_case_draws=bootstrap_draws,profile_clustering_bootstrap_draws=100,
                causal_claims=False,feature_map_coordinate_comparisons=False)
    write_json(output/"summary_results.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    output=args.run.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent/"runs"):
        parser.error("Run must be in this analysis directory")
    with threadpool_limits(limits=4):
        result=summarize(output)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
