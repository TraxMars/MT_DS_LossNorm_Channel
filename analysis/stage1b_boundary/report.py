"""Standalone scientific figures and numerical Stage1B report from saved outputs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import require_stage1b_run,STAGES,SCALES,write_json

METHODS=['A','B1_w2','B2_w2','B3_w2']
DETAIL=['E3','E4','D3','D2','D1']


def read(path):return list(csv.DictReader(path.open()))
def value(row,key):return float(row[key]) if row.get(key) else np.nan

def table(headers,rows):
    fmt=lambda x:'NA' if x is None or isinstance(x,(float,np.floating)) and not np.isfinite(x) else (f'{x:.3e}' if 0<abs(x)<.001 else f'{x:.4f}') if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])

def save(fig,directory,name):
    fig.savefig(directory/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def report(run):
    run=require_stage1b_run(run)
    ab=run/'ablations';figdir=run/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    cells=read(run/'method_scale_summary.csv');lookup={(r['stage'],r['scale'],r['method']):r for r in cells}
    paired=read(run/'paired_target_comparisons.csv');coverage=read(run/'coverage_summary.csv')
    ranks=read(run/'ranking_agreement.csv');stability=json.loads((run/'rank_stability.json').read_text())
    contrasts=read(ab/'group_vs_random.csv');summary=read(ab/'group_summary.csv');specificity=read(ab/'group_specificity.csv');predict=read(ab/'score_predictiveness.csv');versus_A=read(ab/'group_vs_A.csv')
    fig,axs=plt.subplots(1,4,figsize=(14,6),sharey=True,layout='constrained')
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dddddd')
    for method,ax in zip(METHODS,axs):
        matrix=np.array([[value(lookup[(stage,scale,method)],'L_U_cosine_median') for scale in SCALES] for stage in STAGES])
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap=cmap,aspect='auto');ax.set_title(method.replace('_w2',''))
        ax.set_xticks(range(4),SCALES);ax.set_yticks(range(9),STAGES)
        for a in range(9):
            for b in range(4):ax.text(b,a,'NA' if not np.isfinite(matrix[a,b]) else f'{matrix[a,b]:.2f}',ha='center',va='center',fontsize=9)
        ax.set_xlabel('Loss head')
    fig.colorbar(im,ax=axs,shrink=.7,label='Signed L/U parameter-gradient cosine')
    fig.suptitle('Width 2; paired A/B; 8 L and 16 U cases, two crops per case')
    save(fig,figdir,'gradient_matrices')
    comparisons=['boundary_vs_seg_w2','edge_vs_MT_w2','boundary_vs_random_w2','boundary_vs_core_w2','edge_vs_shuffled']
    labels=['GT band / segmentation','Teacher edge / MT','GT band / random region','GT band / interior region','Teacher edge / shuffled weights']
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for comparison,label in zip(comparisons,labels):
        yy=[value(next(r for r in paired if r['stage']==s and r['scale']=='D1' and r['comparison']==comparison),'paired_cosine_median') for s in STAGES]
        axs[0].plot(range(9),yy,marker='o',label=label)
    axs[0].set_xticks(range(9),STAGES);axs[0].set_ylim(0,1.02);axs[0].set_ylabel('Same-case parameter-gradient cosine');axs[0].set_title('Target and spatial negative controls, final loss head')
    axs[0].legend(fontsize=7,loc='lower left')
    for method,label in [('A','A'),('B3_w2','B3'),('R_w2','Random-label-region / edge'),('I_w2','Interior-label-region / edge'),('teacher_shuffle_w2','GT band / shuffled edge')]:
        axs[1].plot(range(9),[value(lookup[(s,'D1',method)],'L_U_cosine_median') for s in STAGES],marker='o',label=label)
    axs[1].set_xticks(range(9),STAGES);axs[1].set_ylabel('L/U cosine');axs[1].set_title('High L/U alignment is not boundary specificity');axs[1].legend(fontsize=7)
    save(fig,figdir,'negative_controls')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in ('B1_w2','B2_w2','B3_w2'):
        axs[0].plot(DETAIL,[value(next(r for r in ranks if r['stage']==s and r['method_a']=='A' and r['method_b']==method),'spearman') for s in DETAIL],marker='o',label=method.split('_')[0])
    axs[0].set_ylabel('Discovery score Spearman vs A');axs[0].set_ylim(0,1.03);axs[0].legend()
    for method in ('A','B1','B2','B3'):
        axs[1].plot(DETAIL,[next(r for r in stability if r['stage']==s and r['method']==method)['top2_bootstrap_jaccard']['mean'] for s in DETAIL],marker='o',label=method)
    axs[1].set_ylabel('Mean Top2 Jaccard, case bootstrap');axs[1].set_ylim(0,1.03);axs[1].legend()
    fig.suptitle('Discovery split only: four L and eight U cases')
    save(fig,figdir,'channel_rankings')
    fig,axs=plt.subplots(1,5,figsize=(16,4),sharey=True,layout='constrained')
    for ax,stage in zip(axs,DETAIL):
        for offset,kind in [(-.12,'ranked'),(.12,'cluster')]:
            rs=[next(r for r in contrasts if r['stage']==stage and r['region']=='B' and r['kind']==kind and r['method']==m and r['metric']=='boundary_ce_delta') for m in ('A','B1','B2','B3')]
            means=np.array([value(r,'mean') for r in rs]);low=np.array([value(r,'low') for r in rs]);high=np.array([value(r,'high') for r in rs])
            ax.errorbar(np.arange(4)+offset,means,yerr=[means-low,high-means],fmt='o',capsize=2,label=kind)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(4),['A','B1','B2','B3']);ax.set_title(stage)
    axs[0].set_ylabel('GT boundary CE change minus random-set mean')
    axs[-1].legend(fontsize=8);fig.suptitle('Energy-matched K=2; boundary-region intervention; 95% case bootstrap, n=4')
    save(fig,figdir,'group_probe_vs_random')
    fig,axs=plt.subplots(1,5,figsize=(16,4),sharey=True,layout='constrained')
    for ax,stage in zip(axs,DETAIL):
        for offset,control in [(-.12,'R'),(.12,'I')]:
            rs=[next(r for r in specificity if r['stage']==stage and r['kind']=='ranked' and r['method']==m and r['metric']=='boundary_ce_delta' and r['control']==control) for m in ('A','B1','B2','B3')]
            means=np.array([value(r,'mean') for r in rs]);low=np.array([value(r,'low') for r in rs]);high=np.array([value(r,'high') for r in rs])
            ax.errorbar(np.arange(4)+offset,means,yerr=[means-low,high-means],fmt='o',capsize=2,label='vs '+control)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(4),['A','B1','B2','B3']);ax.set_title(stage)
    axs[0].set_ylabel('(Boundary - control) CE effect minus random sets')
    axs[-1].legend(fontsize=8);fig.suptitle('Spatial specificity contrast, ranked groups; 95% case bootstrap, n=4')
    save(fig,figdir,'spatial_specificity')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    raw=json.loads((run/'loss_coverage.json').read_text())
    for width in (1,2,3):
        vals=[r['stats']['native_band_fraction'] for r in raw if r['objective']==f'edge_w{width}' and r['scale']=='D1']
        axs[0].boxplot(vals,positions=[width],widths=.5)
    axs[0].set_xticks([1,2,3],['1','2','3']);axs[0].set_xlabel('Native width (voxels)');axs[0].set_ylabel('Teacher band / valid native volume');axs[0].set_title('32 U crops; no U ground truth read')
    for domain,obj,label in [('L','band_w2','GT inner'),('L','band_w2','GT outer'),('U','edge_w2','Teacher edge')]:
        key='effective_cells' if domain=='U' else ('inner_' if label.endswith('inner') else 'outer_')+'effective_cells'
        yy=[np.median([r['stats'][key] for r in raw if r['objective']==obj and r['scale']==scale]) for scale in SCALES]
        axs[1].plot(SCALES,yy,marker='o',label=label)
    axs[1].set_yscale('log');axs[1].set_ylabel('Median effective weighted cells (ESS)');axs[1].set_title('Coarse boundary coverage; width 2');axs[1].legend()
    save(fig,figdir,'coverage')
    lines=['# Stage 1B：完整数值报告','','主结果采用原图宽度2；宽度1/3是预先声明的敏感性分析。全部为冻结eval、FP32、同一iteration15000配对checkpoint；病例和crop与A一致。详细定义见[DESIGN](../../DESIGN.md)，研究结论见[FINDINGS](../../FINDINGS.md)。','',
           '## 九阶段×四头：A/B1/B2/B3 有符号 L/U 参数梯度余弦','',
           '格值为可靠通道（≥75%有效病例对）的病例对平均余弦中位数。先每病例两个crop梯度平均，再比较8L×16U病例；不可达为NA。']
    for method in METHODS:
        lines+=['','### '+method.replace('_w2',''),'',table(['stage',*SCALES],[[stage]+[value(lookup[(stage,scale,method)],'L_U_cosine_median') for scale in SCALES] for stage in STAGES])]
    lines+=['','## 有效性矩阵（B3主宽度）','',table(['stage',*SCALES],[[s]+[(lookup[(s,sc,'B3_w2')].get('reliable_channels','')+'/'+lookup[(s,sc,'B3_w2')].get('channels','')) if lookup[(s,sc,'B3_w2')]['graph_reachable']=='True' else 'graph NA' for sc in SCALES] for s in STAGES]),'',
            '逐crop范数阈值和状态见observation_norms.csv；病例均值可靠不代表每个crop均可靠。完整17组对照矩阵保存在method_scale_summary.csv。','',
            '## 同病例目标/区域对照梯度方向（最终D1头）','',table(['stage']+comparisons,[[s]+[value(next(r for r in paired if r['stage']==s and r['scale']=='D1' and r['comparison']==c),'paired_cosine_median') for c in comparisons] for s in STAGES]),'',
            '## 重复裁块及跨病例方向稳定性','',table(['stage','method','within L','within U','between L','between U'],[[s,m]+[value(lookup[(s,'D1',m)],k) for k in ('within_L_median','within_U_median','between_L_median','between_U_median')] for s in STAGES for m in ('A','B3_w2')]),'',
            '跨尺度相似、SVD有效秩与抵消比见channel_multiscale.csv/cross_scale_summary.csv/multiscale_summary.csv；这些是描述梯度几何，不能证明协同。','',
            '## 三带宽、四尺度目标覆盖','',table(['domain','objective','head','valid/total','inner ESS','outer ESS','U ESS','U band fraction'],[[r['domain'],r['objective'],r['scale'],r['valid']+'/'+r['total']]+[value(r,k+'_median') for k in ('inner_effective_cells','outer_effective_cells','effective_cells','native_band_fraction')] for r in coverage]),'',
            '## 通道排名与稳定性','',table(['stage','A vs','score Spearman','Top2 Jaccard'],[[r['stage'],r['method_b'],value(r,'spearman'),value(r,'top2_jaccard')] for r in ranks if r['method_a']=='A']),'',
            table(['stage','method','bootstrap rho median','Top2 Jaccard mean'],[[r['stage'],r['method'],r['score_bootstrap_spearman']['median'],r['top2_bootstrap_jaccard']['mean']] for r in stability]),'',
            '主K=2；宽度1/3敏感性见ranking_width_sensitivity.csv。聚类信息和实际通道ID见cluster_exploration.json及channel_plan.json。聚类只用发现病例。','',
            '## 最终分割预测：边界区域K=2联合消融','',
            '以下是边界区域消融造成的最终GT boundary CE变化，减去同K同实际L2随机集合均值。正值表示更大边界CE影响，不等于训练性能提升。按4个probe病例先平均crop，再1000次病例bootstrap。探索性区间不作多重检验显著性结论。','',
            table(['stage','kind','method','Δboundary CE - random','95% low','95% high'],[[r['stage'],r['kind'],r['method'],value(r,'mean'),value(r,'low'),value(r,'high')] for r in contrasts if r['region']=='B' and r['metric']=='boundary_ce_delta']), '',
            '最终预测指标变化（主排名组，B区域，病例平均；正Dice/F1为增加，正CE为增加）：','',
            table(['stage','method','full CE','boundary CE','hard Dice','soft Dice','contour F1','changed fraction'],[[stage,method]+[value(next(r for r in summary if r['stage']==stage and r['method']==method and r['kind']=='ranked' and r['region']=='B' and r['metric']==metric),'mean') for metric in ('ce_delta','boundary_ce_delta','dice_delta','soft_dice_delta','boundary_f1_delta','changed_prediction_fraction')] for stage in DETAIL for method in ('A','B1','B2','B3')]), '',
            '## B相对A的同病例消融影响差异','',
            '同K、同实际L2的B方法联合组Δboundary CE减去A联合组。正值表示B选中组的衰减产生更大边界CE变化，不是B训练后的性能收益。','',
            table(['stage','kind','method','B-A','95% low','95% high'],[[r['stage'],r['kind'],r['method'],value(r,'mean'),value(r,'low'),value(r,'high')] for r in versus_A if r['region']=='B' and r['metric']=='boundary_ce_delta']), '',
            '补充影响幅值（|ΔCE|，不改变预先选定的通道）：','',
            table(['stage','kind','method','|ΔCE| B-A','95% low','95% high'],[[r['stage'],r['kind'],r['method'],value(r,'mean'),value(r,'low'),value(r,'high')] for r in versus_A if r['region']=='B' and r['metric']=='abs_boundary_ce_delta']), '',
            '## 真实边界 vs 区域负对照的因果探测','',
            '表为[(真实带干预−区域干预)的boundary CE影响]−随机集合的同类差分。单个正差分不建立边界专属性；还需跨病例重复并优于A。D1的1×1最终头是逐位置映射，非重叠内部区域干预在结构上不能改变边界带CE；不能把这种直接支持不重叠当作边界通道证据。','',
            table(['stage','method','region control','difference','95% low','95% high'],[[r['stage'],r['method'],r['control'],value(r,'mean'),value(r,'low'),value(r,'high')] for r in specificity if r['kind']=='ranked' and r['metric']=='boundary_ce_delta']), '',
            '## 全部单通道分数对最终边界CE影响的预测关系','',
            '发现集分数与probe病例逐通道消融影响的Spearman，先在各病例计算、再汇总4病例。正分不预设绝对梯度重要性；不事后取绝对值或翻转方向。','',
            table(['stage','method','region contrast','rho mean','95% low','95% high'],[[r['stage'],r['method'],r['region_contrast'],value(r,'mean'),value(r,'low'),value(r,'high')] for r in predict if r['metric']=='boundary_ce_delta' and r['region_contrast'] in ('B','B-R')]),'',
            '补充描述性影响幅值与hard预测改变相关性，不用于重新选择通道：','',
            table(['stage','method','endpoint','rho mean','95% low','95% high'],[[r['stage'],r['method'],r['metric'],value(r,'mean'),value(r,'low'),value(r,'high')] for r in predict if r['metric'] in ('abs_boundary_ce_delta','changed_prediction_fraction') and r['region_contrast']=='B']), '',
            '## 联合非加性','',
            '匹配singleton使用pair相同alpha；logit交互是zjoint−zsingle1−zsingle2+zbase的RMS。D1最终1×1头是线性映射，其CE非加性不代表通道协同。其他阶段的logit非加性同样需与随机组比较。','',
            table(['stage','kind','method','metric','case mean'],[[r['stage'],r['kind'],r['method'],r['metric'],value(r,'mean')] for r in summary if r['region']=='B' and r['metric'] in ('boundary_ce_interaction','logit_interaction_rms')]), '',
            '## LossNorm与执行','',
            'B原始梯度没有独立LossNorm。次要A-denominator贡献和余弦仅用相同crop原A归一化因子。loss_scaling.json保存原损失与固定A因子范围；expanded_channel_relations.csv保留重新加权前后方向。量纲不同，不按原始范数直接断言B优于A。','']
    for filename,folder in [('extraction_results.json',run),('analysis_results.json',run),('ablation_results.json',ab),('summary_results.json',ab),('verification_results.json',run)]:
        content=json.loads((folder/filename).read_text());lines+=['### '+filename,'','```json',json.dumps(content,indent=2,ensure_ascii=False),'```','']
    lines+=['## 解释限制','','所有L病例参与过Baseline训练，probe只是未用于通道选择，不能称独立泛化。仅1个checkpoint/1个数据划分、2个crop/病例、4个probe L病例、8组随机K=2集合、单一干预预算。没有读取U GT，因此不能宣称Teacher边界正确或U边界准确率改善。没有增强逆变换、完整体积干预或新损失训练。全部发现应先在独立标注验证病例、多checkpoint及多标注seed复核。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(run/'report_results.json',dict(status='PASS',figures=6,formats=['PNG','PDF'],report='REPORT.md'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run.resolve())
