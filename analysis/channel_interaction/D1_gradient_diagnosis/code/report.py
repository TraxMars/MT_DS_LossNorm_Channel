"""Produce separate dataset findings and a cross-dataset comparison from verified data."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import ROOT,SPECS,sha256,write_json

def fmt(value,digits=3): return 'NA' if value is None else f'{value:.{digits}f}'
def pct(value):return 'NA' if value is None else f'{value*100:.1f}%'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def csvrows(path):return list(csv.DictReader(path.open()))
def lookup(rows,**keys):return next(r for r in rows if all(r[k]==v for k,v in keys.items()))

def dataset_report(ds,run,result):
    base=run/ds
    channels=csvrows(base/'per_channel.csv');mini=csvrows(base/'mini_channel_stability.csv')
    singles=[r for r in result['scale_summary'] if r['policy']=='raw']
    rows=[]
    for r in singles:
        mt=lookup(result['mini_summary'],mode='train',stage=r['stage'],head=r['head'],policy='raw')
        rows.append([r['stage']+'/'+r['head'],fmt(r['median_channel_lu_mean']),pct(r['negative_pair_fraction']),
                     f"{r['negative_interval_channels']}/{r['channels']}",pct(mt['negative_fraction']),pct(mt['valid_fraction'])])
    text=f'# {ds} 10%：D1-A 第一轮发现\n\n'
    text+=f"使用本数据集 iteration=15000 原始配对 Student/Teacher。{SPECS[ds]['labeled']} 个 L、16 个 U，每病例两裁块；patch={SPECS[ds]['patch']}。原始输出、范数、有效性、逐病例组合及 mini-batch 全表保存在本目录。\n\n"
    text+=table(['参数/头','eval 通道均值中位数','eval 负余弦比例','负区间通道数','train mini 负比例','train mini 有效覆盖'],rows)
    text+='\n“负区间”是 200 次分别重采样 L/U 病例得到的未校正探索性区间；负比例只以有效余弦为分母。病例组合不是独立样本。三个 mini 病例组×两次裁块/Dropout 的结果只作描述。\n\n'
    if ds=='LA':
        text+='## 原 Stage1A 四个预指定通道\n\n'
        rows=[]
        for r in result['LA_predeclared_channels']:
            c=r['channel'];m=lookup(mini,mode='train',stage='D1',head='D1',policy='raw',channel=str(c))
            rows.append([c,fmt(r['lu_mean']),f"[{fmt(r['bootstrap_low'])}, {fmt(r['bootstrap_high'])}]",
                         pct(r['negative_pair_fraction']),f"{m['negative_units']}/{m['valid_units']}"])
        text+=table(['通道（从 0 起）','eval raw L/U 均值','病例 bootstrap 95% 区间','eval 负比例','train 负/有效单位'],rows)
        mixed=[];negative=[]
        for c in (0,4,10,14):
            m=lookup(mini,mode='train',stage='D1',head='D1',policy='raw',channel=str(c))
            n,v=int(m['negative_units']),int(m['valid_units'])
            if 0<n<v:mixed.append(str(c))
            if n==v and v>0:negative.append(f'{c}（{n}/{v}）')
        text+=f"\ntrain mini 中通道 {', '.join(mixed)} 出现正负符号切换；通道 {', '.join(negative)} 的有效单位全为负，但最多仅六个重叠单位且覆盖不同。frozen eval 的四通道结论不能直接当作 train-mode 固定通道规则。\n"
        rep=result['historical_replay']
        text+=f"\n与历史 GPU FP32 原始梯度逐病例/裁块复核：最大 relative L2={rep['max_relative_l2']:.3e}，中位数={rep['median_relative_l2']:.3e}。本轮 CPU 后端，未宣称逐 bit 复现。\n\n"
    else:
        text+='\nPancreas 仅按自身参数坐标分析完整通道。上述区间通道计数不能与 LA 的 0、4、10、14 编号建立对应。\n\n'
    text+='## 四尺度与重复性\n\n'
    text+=table(['参数/头','域','同病例两裁块中位余弦','不同病例中位余弦'],[
        [r['stage']+'/'+r['head'],r['role'],fmt(r['within_crop_median']),fmt(r['between_case_median'])]
        for r in result['case_crop_stability']])
    text+='\nE3 跨头方向（同观测、同参数核坐标）：\n\n'
    text+=table(['域','头对','中位余弦','负比例'],[[r['role'],r['head_a']+'/'+r['head_b'],fmt(r['median_cosine']),pct(r['negative_fraction'])] for r in result['cross_head_summary']])
    text+='\nD1 参数只可达最终 D1 头，其他三头是结构不可达，保存为缺失；不能据此认为通道没有价值。E3 参数可达四个头，跨头关系与跨病例 L/U 关系分别报告。E3/D1 的每通道参数向量长度分别为 1728/432，不直接比较不同层的绝对范数或匹配通道编号。\n\n'
    text+='## Raw、LossNorm 与抵消\n\n'
    text+=table(['参数/头','病例 eval raw rho','病例 eval effective rho','train raw rho','train effective rho'],[
        [r['stage']+'/'+r['head'],fmt(r['sum_to_norms_ratio_median']),
         fmt(lookup(result['scale_summary'],stage=r['stage'],head=r['head'],policy='lossnorm')['sum_to_norms_ratio_median']),
         fmt(lookup(result['mini_summary'],mode='train',stage=r['stage'],head=r['head'],policy='raw')['median_sum_to_norms_ratio']),
         fmt(lookup(result['mini_summary'],mode='train',stage=r['stage'],head=r['head'],policy='lossnorm')['median_sum_to_norms_ratio'])]
        for r in singles])
    text+=f"\nrho=||gL+gU||/(||gL||+||gU||)，越小表示抵消越强；抵消程度为 1-rho。raw 使用原目标中的 U 权重 .3，effective 使用整体四头目标 LossNorm。两裁块先归一化再平均所产生的 L/U 余弦均值最大差为 {fmt(result['raw_lossnorm_case_mean_max_abs_difference'])}。同观测正标量缩放保留余弦，但会改变幅值平衡及抵消；病例平均向量会受重加权影响。完整范数与 U/L 比例在 observation_norms.csv、mini_relations.csv。\n\n"
    text+='## 验证、局限和继续条件\n\n'
    text+=f"统一数值策略：FP32 首测触发原重构阈值时，恢复 BN/RNG 并以 FP64 重放整个 L/U 单元，仍通过原严格检查。此数据集有 {result['fp64_fallback_units']} 个单元触发重放，正式保存的 {result['fp64_observations']} 个观测使用 FP64；其余为 FP32。触发原因与失败尝试的状态恢复见 restoration_checks.json。该策略不依据余弦符号或数据集效果，不修改公式或阈值。\n\n"
    verify=json.loads((run/'verification.json').read_text())['datasets'][ds]
    text+=f"原始/有效梯度重构最大 relative L2 为 {verify['max_raw_relative_reconstruction']:.3e}/{verify['max_lossnorm_relative_reconstruction']:.3e}；{verify['invalid_channels']} 个可达逐观测通道因零值/过小而无效，已排除余弦。BN、RNG、参数、模式、grad 均恢复；train 首组重放误差为 0；U GT 读取和 optimizer step 均为 0。\n\n"
    text+='沿用原 checker：仅在绝对 L2 误差 >1e-9 且相对 L2 >1e-4 时拒绝。Pancreas 有两条极小 U 梯度的相对误差超过 1e-4，但最大系数误差仅约 1.9e-12 / 3.2e-12，且对应的原始 D1 十六通道全部低于有效性阈值；未将它们当作可靠逐裁块方向。详见 [数值边缘记录](../../../NUMERICAL_EDGE_CASES.md)。\n\n'
    text+='仅单 checkpoint、单划分、16 个 U、每病例两裁块。mini-batch 只含三个病例组，未分离 BN 与 Dropout；LA mini 使用 6/8 个 L。部分单元的 FP64 重放会引入微小舍入差异，mode_comparison.csv 同时记录两侧精度。原 test.list 已用于 Baseline 在线选模，不能称为独立验证集。没有训练性能或干预终点，没有多重比较校正。\n\n'
    text+='**负余弦不能解释为有害通道。**它描述当前两个目标在共享参数坐标上的方向差异，不能证明删除/交换会改善性能，也不能证明协同、互补或边界语义。满足继续扩大冻结诊断的工程条件；本轮证据不足以据此固定通道筛选、开始 POS 或通道交换。\n'
    (base/'FINDINGS.md').write_text(text)

def figures(run,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks=[('D1','D1'),('E3','D4'),('E3','D3'),('E3','D2'),('E3','D1')]
    data=[];labels=[];coverage=[]
    for ds in SPECS:
        r=comparison[ds]
        for mode in ('case eval','mini eval','mini train'):
            row=[];cov=[]
            for stage,head in tasks:
                if mode=='case eval':
                    x=lookup(r['scale_summary'],stage=stage,head=head,policy='raw')
                    row.append(x['negative_pair_fraction']);cov.append(x['valid_pair_fraction'])
                else:
                    x=lookup(r['mini_summary'],mode=mode.split()[1],stage=stage,head=head,policy='raw')
                    row.append(x['negative_fraction']);cov.append(x['valid_fraction'])
            data.append(row);coverage.append(cov);labels.append(ds+' '+mode)
    fig,ax=plt.subplots(figsize=(10,5.5));im=ax.imshow(np.array(data,dtype=float),vmin=0,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(5),[a+'/'+b for a,b in tasks]);ax.set_yticks(range(6),labels)
    for i in range(6):
        for j in range(5):
            v=data[i][j];q=coverage[i][j]
            label='NA' if v is None else f'{v:.1%}'
            ax.text(j,i,f'{label}\ncoverage {q:.1%}',ha='center',va='center',fontsize=9,
                    color='white' if v is not None and abs(v-.5)>.3 else 'black')
    ax.set_title('Signed L/U gradient conflict among valid channels\nCase-pair and mini-batch denominators differ; no channel correspondence across datasets')
    fig.colorbar(im,ax=ax,label='fraction cosine < 0');fig.tight_layout()
    path=run/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/'gradient_conflict.png',dpi=180);fig.savefig(path/'gradient_conflict.pdf');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    assert run.is_relative_to(ROOT/'D1_gradient_diagnosis/results')
    assert json.loads((run/'verification.json').read_text())['status']=='PASS'
    comparison=json.loads((run/'comparison.json').read_text())
    for ds in SPECS:dataset_report(ds,run,comparison[ds])
    text='# D1-A 跨数据集比较\n\n'
    text+='两套独立模型使用各自原始配对 checkpoint、原划分与原 patch；同一套梯度公式、阈值和统计代码。比较的是参数/监督头层面的分布，不按独立模型的通道编号配对。\n\n'
    rows=[]
    for ds,r in comparison.items():
        for x in r['scale_summary']:
            if x['policy']!='raw':continue
            m=lookup(r['mini_summary'],mode='train',stage=x['stage'],head=x['head'],policy='raw')
            rows.append([ds,x['stage']+'/'+x['head'],fmt(x['median_channel_lu_mean']),pct(x['negative_pair_fraction']),pct(x['valid_pair_fraction']),
                         f"{x['negative_interval_channels']}/{x['channels']}",pct(m['negative_fraction']),pct(m['valid_fraction'])])
    text+=table(['数据集','参数/头','eval 通道均值中位数','eval 负比例','eval 有效覆盖','负区间通道数','train 负比例','train 有效覆盖'],rows)
    text+='\n![梯度关系与有效覆盖](results/'+run.name+'/figures/gradient_conflict.png)\n\n'
    text+='## 相同点\n\n'
    text+='两套使用相同的 FP32 首测、原阈值失败后整个 L/U 单元 FP64 重放规则，逐观测记录精度；未按数据集效果调整。首次 LA train 的数值失败及配对精度校准保留在独立目录，见 [数值校准](NUMERICAL_CALIBRATION.md)。\n\n'
    text+='两套均通过四独立头到原始总梯度、LossNorm 有效梯度的重构；D1 只可达最终头，E3 可达四头。同一 mini 单元 raw/effective 正缩放不改变有效余弦；跨裁块先归一化再平均可能改变病例关系。两套均显式排除零值/近零值，并恢复 train-mode BN 与 RNG。\n\n'
    text+='E3 上较细的 D2/D1 头方向在两套中均比 D4/D1 更一致；这是梯度几何描述，不是协同或互补的证明：\n\n'
    rows=[]
    for ds,r in comparison.items():
        for role in ('L','U'):
            fine=lookup(r['cross_head_summary'],role=role,head_a='D2',head_b='D1')
            coarse=lookup(r['cross_head_summary'],role=role,head_a='D4',head_b='D1')
            rows.append([ds,role,fmt(fine['median_cosine']),fmt(coarse['median_cosine'])])
    text+=table(['数据集','域','E3 上 D2/D1 中位余弦','E3 上 D4/D1 中位余弦'],rows)+'\n'
    text+='## 差异与适用范围\n\n'
    for stage,head in [('D1','D1'),('E3','D4'),('E3','D3'),('E3','D2'),('E3','D1')]:
        a=lookup(comparison['LA']['scale_summary'],stage=stage,head=head,policy='raw')
        b=lookup(comparison['Pancreas']['scale_summary'],stage=stage,head=head,policy='raw')
        text+=f"- {stage}/{head} 的 eval 负比例：LA {pct(a['negative_pair_fraction'])}，Pancreas {pct(b['negative_pair_fraction'])}；负区间通道数分别为 {a['negative_interval_channels']} 与 {b['negative_interval_channels']}。这些是各自模型的分布摘要，不能跨模型对应具体通道。\n"
    text+='\n若某个头/指标只有一套数据支持，结论只属于该数据集；即使两套都有负余弦，也只支持方向差异的存在，不能证明某个固定负通道集合普遍成立。各自的跨头方向和病例/裁块稳定性详见下列分数据集报告。\n\n'
    text+='D1 的病例/裁块稳定性也存在明显差异：\n\n'
    text+=table(['数据集','域','同病例两裁块余弦中位数','不同病例余弦中位数'],[
        [ds,x['role'],fmt(x['within_crop_median']),fmt(x['between_case_median'])]
        for ds,r in comparison.items() for x in r['case_crop_stability'] if x['stage']=='D1'])+'\n'
    text+='LA 的监督 D1 方向在这些病例/裁块中较稳定；Pancreas 的监督 D1 跨病例中位关系接近零，不能外推 LA 的稳定性。LA 的预指定通道 0、4 在 train mini 中出现符号切换；10、14 的负观测也受到小样本和有效覆盖限制。\n\n'
    text+='- [LA 发现](results/'+run.name+'/LA/FINDINGS.md)\n- [Pancreas 发现](results/'+run.name+'/Pancreas/FINDINGS.md)\n\n'
    text+='## 局限与继续条件\n\n'
    text+='仅一个迭代/种子/划分；U 子集、两个裁块和少量重叠 mini 单元，未做独立训练、性能终点或因果干预。train 模式同时启用 BN 与 Dropout，且每单元恢复 checkpoint BN，不能代表连续训练轨迹。LA 与 Pancreas 的有效覆盖必须与负比例一起阅读。原 test.list 用于在线选模，不是独立留出验证。\n\n'
    text+='两套均满足后续扩大冻结梯度诊断的工程继续条件；不满足将负余弦解释为有害通道、固定筛选方法或证明跨数据集交互规则有效的科学条件。本轮未执行 POS、通道交换、参数更新或 optimizer.step，历史文件由 D0 前后 SHA 保护。\n'
    text+='\n最终历史核查：原有 3,063 个文件的 SHA 全部一致。Boundary 项目并发新增 99 个 C15 文件，触发原“无新增文件”严格检查；原失败记录与并发说明均保留在 D0/results/round1_after_D1/。这些新增文件未用于或写入本轮诊断。\n'
    figures(run,comparison)
    (ROOT/'D1_gradient_diagnosis/CROSS_DATASET_REPORT.md').write_text(text)
    finding='# D1-A 第一轮 FINDINGS\n\n'
    finding+='已完成 LA 10% 与 Pancreas 10% 的 frozen eval、匹配 eval mini-batch 与真实 2L+2U train-mode 梯度诊断。原始结果分别保存，统一比较见 [跨数据集报告](CROSS_DATASET_REPORT.md)。\n\n'
    for ds,r in comparison.items():
        d=lookup(r['scale_summary'],stage='D1',head='D1',policy='raw')
        finding+=f"- **{ds}**：D1/D1 eval 有效病例组合的负余弦比例为 {pct(d['negative_pair_fraction'])}，有 {d['negative_interval_channels']}/{d['channels']} 个通道的探索性病例区间位于负侧。详见 [{ds} FINDINGS](results/{run.name}/{ds}/FINDINGS.md)。\n"
    finding+='\n**负余弦不能解释为有害通道。**两套独立模型不按通道编号匹配；没有基于任一数据集修改筛选或交互规则。单数据集成立不代表跨数据集普遍性。\n\n'
    finding+='两套工程验证通过；本轮可继续扩大冻结诊断，证据不足以直接固定通道筛选或启动 POS/交换。完整限制、共同点、差异和每数据集继续条件均在独立报告中列明。\n'
    (ROOT/'D1_gradient_diagnosis/FINDINGS.md').write_text(finding)
    write_json(run/'report_source.json',dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)))

if __name__=='__main__':main()
