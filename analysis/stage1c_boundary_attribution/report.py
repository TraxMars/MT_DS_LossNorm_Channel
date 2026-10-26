"""Generate C0-only reports and standalone numerical fidelity figures."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from support import ROOT,STAGES,DOSES,output_path,write_json


def read(path):return list(csv.DictReader(path.open()))
def num(row,key):return float(row[key]) if row.get(key) else np.nan

def fmt(x):
    if x is None or isinstance(x,(float,np.floating)) and not np.isfinite(x):return 'NA'
    if isinstance(x,(float,np.floating)):return f'{x:.3e}' if 0<abs(x)<.001 else f'{x:.5f}'
    return str(x)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(fmt(x) for x in row)+' |' for row in rows])

def save(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def report(run):
    run=output_path(run)
    execution=json.loads((run/'execution_results.json').read_text());verification=json.loads((run/'verification_results.json').read_text())
    summary=json.loads((run/'summary_results.json').read_text());gates=json.loads((run/'fidelity_gates.json').read_text())
    tests=json.loads((ROOT/'TEST_RESULTS.json').read_text());regression=json.loads((ROOT/'REGRESSION_RESULTS.json').read_text())
    assert all(r['status']=='PASS' for r in (execution,verification,summary,tests,regression))
    assert json.loads((ROOT/'REMOTE_RECHECK.json').read_text())['status']=='PASS'
    rows=read(run/'deltas.csv');fidelity=read(run/'fidelity_by_stage_dose.csv');metrics=read(run/'intervention_metrics.csv')
    geometry=json.loads((run/'geometry.json').read_text());nodes=json.loads((run/'node_identity.json').read_text());noise=json.loads((run/'noise_calibration.json').read_text())
    folder=run/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for gate,label,color in zip(gates,['L transition, margin 1','U pair-gap consistency'],['#246ca5','#b75727']):
        stats=gate['dose_statistics'];xx=[r['dose'] for r in stats]
        axs[0].plot(xx,[r['median_relative_residual'] for r in stats],marker='o',color=color,label=label)
        axs[1].plot(xx,[r['median_absolute_residual'] for r in stats],marker='o',color=color,label=label)
        axs[1].axhline(stats[0]['median_epsilon'],color=color,linestyle=':',lw=1)
    for ax in axs:ax.set_xscale('log');ax.set_xticks(DOSES,[str(d) for d in DOSES]);ax.set_xlabel('Fixed relative energy budget');ax.legend(fontsize=8)
    axs[0].set_ylabel('Median |actual - predicted| / |predicted|');axs[0].set_ylim(bottom=0)
    axs[1].set_ylabel('Median absolute residual');axs[1].set_yscale('log')
    fig.suptitle(f'C0 fixed identifiable cells: L n={gates[0]["eligible_cells"]}, U n={gates[1]["eligible_cells"]}; dotted lines = epsilon')
    save(fig,folder,'dose_fidelity')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,(domain,obj) in zip(axs,[('L','transition_m1'),('U','u_transition')]):
        rr=[r for r in rows if r['domain']==domain and r['objective']==obj and r['status']=='valid']
        bound=max(max(abs(num(r,'delta_hat')) for r in rr),max(abs(num(r,'delta_actual')) for r in rr))*1.05
        for dose,color in zip(DOSES,['#2678b2','#e68a27','#7654a0']):
            chosen=[r for r in rr if float(r['dose'])==dose]
            ax.scatter([num(r,'delta_hat') for r in chosen],[num(r,'delta_actual') for r in chosen],s=15,alpha=.7,color=color,label=str(dose))
        uncertain=[r for r in rr if r['actual_direction']=='indeterminate']
        ax.scatter([num(r,'delta_hat') for r in uncertain],[num(r,'delta_actual') for r in uncertain],s=35,facecolors='none',edgecolors='black',label='indeterminate')
        ax.plot([-bound,bound],[-bound,bound],color='black',lw=.8,linestyle='--');ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound)
        ax.set_title(domain+' primary objective, all fixed cells');ax.set_xlabel('Signed first-order prediction');ax.set_ylabel('Actual loss difference');ax.legend(fontsize=7)
    save(fig,folder,'predicted_actual')
    fig,axs=plt.subplots(1,2,figsize=(11,3.7),sharey=True,layout='constrained')
    for ax,(domain,obj) in zip(axs,[('L','transition_m1'),('U','u_transition')]):
        for offset,dose in [(-.23,.01),(0.,.03),(.23,.1)]:
            values=[num(next(r for r in fidelity if r['domain']==domain and r['objective']==obj and r['stage']==s and float(r['dose'])==dose),'actual_indeterminate_fraction') for s in STAGES]
            ax.bar(np.arange(5)+offset,values,width=.22,label=str(dose))
        ax.set_xticks(range(5),STAGES);ax.set_title(domain+' primary objective');ax.set_ylabel('Indeterminate fraction');ax.set_ylim(0,.35);ax.legend(fontsize=8)
    fig.suptitle('Predeclared uncertainty interval; no forced direction labels')
    save(fig,folder,'direction_uncertainty')
    test_lines=['# C0 测试与完整性报告','','最终判定：**PASS**。实际结果见TEST_RESULTS.json、REGRESSION_RESULTS.json和最终run的verification_results.json。','',
       table(['检查','实际结果'],[['C0 CPU合同测试',f'{tests["tests_run"]}/{tests["tests_run"]} PASS，{tests["elapsed_seconds"]:.3f}s'],
           *[[r['suite'],f'{r["tests_run"]}/{r["tests_run"]} PASS，{r["elapsed_seconds"]:.3f}s'] for r in regression['suites'] if 'tests_run' in r],
           ['verify_core.py','exit0：上游输出、Teacher梯度隔离、目标/EMA、LossNorm和滑窗检查通过'],
           ['最终GPU smoke',f'4病例/8crop，{execution["elapsed_seconds"]:.3f}s，峰值{execution["peak_allocated_mib"]:.1f}MiB'],
           ['历史SHA256',f'{verification["protection"]["checked_files"]}文件一致，changed=[]'],
           ['主目标归因独立重算',f'360个Delta_hat，最大绝对差{verification["independent_feature_prediction_max_absolute_error"]:.3e}'],
           ['A/B参数梯度兼容',f'80个原向量，最大相对L2差{verification["maximum_AB_parameter_relative_error"]:.3e}']]),'',
       '合计52项unittest（C0 21+历史31），另有core验证。历史suite只作为模块导入，报告写到C0，未运行会覆盖Stage1A/1B旧报告的main。','',
       '## 覆盖内容','','native位置对方向/6邻接/有效边缘；确定性采样；空边界；排序目标平移不变性；最终D1 tensor限制；翻转逆坐标；粗尺度坍缩/占有率；端点置换marginal；Teacher stop-gradient/U GT键拒绝；五个真实计算节点和skip融合；Stage1B干预等价；D3/D2辅助头梯度反例；特征/参数梯度独立与已有.grad保持；小扰动正负方向；净GxA/逐点绝对GxA区别；数值区间；固定剂量/alpha失败；实际张量减法L2与舍入误差；输出路径保护。','',
       '## 独立最终检查','',table(['检查','结果'],[[k,v] for k,v in verification['checks'].items()]),'',
       '## 保留的失败和修正','','首轮20项有2个测试夹具失败（TEST_ATTEMPT_01.json）：对两个类别加共同logit偏移，softmax不变；随机模型单个通道可能死ReLU，不能据零响应否定路径。改为只扰动前景logit和已知激活的合成网络后20项通过（TEST_ATTEMPT_02.json）；增加实际L2测试后21项通过。真实checkpoint和选定通道未改变。','',
       '第一次真实smoke runs/c0_la10_fixed保留，其L2字段测量的是公式扰动向量。最终runs/c0_la10_checked按实际F_before-F_after测量，同时保存intended_l2和subtraction_roundoff_l2，重跑同一8crop。前向削弱公式、病例、采样、剂量和方向门槛未改变。','',
       '初次独立验证的检查和JSON已PASS，但CLI打印numpy bool时exit1；输出改用既有clean序列化后，对最终run重做验证并exit0。可选远程完工复核首个响应未解析到SHA，不视为远程变化；重试commits/main仍为锁定SHA。记录见EXECUTION_NOTES.json和REMOTE_RECHECK*。','',
       '## 状态和数据保护','','Student/Teacher完整state_dict（参数与BN）精确不变、Teacher frozen/no_grad、均为eval、.grad均为None，CPU/CUDA/Python/numpy RNG不变，无残留hooks。原checkpoint strict=True加载，SHA不变。U loader只读image，GT被U目标接口拒绝，U的GT准确性指标为不可用。独立验证只重读原2L的GT验证方向；HDF5只读且size/mtime不变，不哈希U容器以避免读取标签字节。生产源码、历史Stage0/1A/1B、Results/summary/checkpoint/划分未改。执行源文件及DESIGN与source_at_run中的预先哈希一致。']
    (ROOT/'TEST_REPORT.md').write_text('\n'.join(test_lines)+'\n')
    dose_rows=[[g['domain'],r['dose'],r['eligible'],r['median_relative_residual'],r['median_absolute_residual'],r['median_epsilon']] for g in gates for r in g['dose_statistics']]
    loss_groups=defaultdict(list)
    for r in noise:loss_groups[(r['domain'],r['objective'])].append(r)
    loss_rows=[[d,obj,float(np.median([r['base'] for r in rr])),min(r['epsilon'] for r in rr),max(r['epsilon'] for r in rr),max(r['null_max_difference'] for r in rr)] for (d,obj),rr in loss_groups.items()]
    energy_error=max(abs(num(r,'actual_l2')-num(r,'target_l2'))/num(r,'target_l2') for r in metrics if r['status']=='valid')
    rounding=max(num(r,'subtraction_roundoff_l2')/num(r,'target_l2') for r in metrics if r['status']=='valid')
    primary=[r for r in fidelity if r['objective'] in ('transition_m1','u_transition')]
    lines=['# Stage1C C0：最终 smoke 报告','','**C0通过。仅证明几何、节点、干预和局部数值归因可用，不证明可靠边界通道。没有执行C1/C2/C3或正式训练。**','',
       '最终运行：`runs/c0_la10_checked/`。预先设计见[DESIGN](DESIGN.md)，测试与失败记录见[TEST_REPORT](TEST_REPORT.md)。','',
       '## 起点核查','','本地07a0b90f840dbf26776e2f9df9077be47dd90fa3，远程main锁定5f6303a4ffde996a8200c294a47d35062f54e8b6。共有290个文件Git blob全部一致，本地另有22个历史图件；不能称两个完整Git树相同。公开artifact manifest相同，本地Stage1B manifest的210文件SHA全部核实；完工时远程HEAD仍一致。711个历史文件最终SHA全部不变。','',
       '## 固定范围与几何','',f'原LA10 iteration15000配对Student/Teacher，eval，FP32网络/反向、TF32 off、FP64损失运算。发现集2L+2U，每病例原有2crop，共8crop。最终运行{execution["elapsed_seconds"]:.3f}s，峰值{execution["peak_allocated_mib"]:.1f}MiB；未进入probe。所有病例曾参与Baseline训练，不能称独立泛化验证。','',
       table(['域','病例','crop','候选面数','采样对数','端点M覆盖','状态'],[[r['domain'],r['case'],r['repeat'],r['stats']['candidate_faces'],r['stats']['selected_pairs'],r['pair_endpoint_mask_fraction'],r['status']] for r in geometry]),'',
       'L为GT前景/背景6邻接对的softplus排序，主margin1.0，0.5/2.0为固定敏感性；独立参照普通CE和原宽2边界CE。U为Teacher候选位置对的概率差MSE，参照MT和edge_w2。没有将目标相加或独立LossNorm。2048对可共享端点，不是2048个独立样本。真实数据只用最终D1损失，不在粗头强制制造细边界；CPU验证粗尺度坍缩/占有率与逆坐标。','',
       'U只验证候选几何/概率关系、同crop整数坐标与stop-gradient，**没有验证Teacher真实边界正确性或增强可靠性**，可靠性状态为not_assessed_in_C0，不能直接用于训练。','',
       '## 五阶段同节点合同','',table(['阶段','真实干预模块/位置','shape','固定通道'],[[s,next(r for r in nodes if r['stage']==s)['module']+'/'+next(r for r in nodes if r['stage']==s)['site'],next(r for r in nodes if r['stage']==s)['shape'],[0,next(r for r in nodes if r['stage']==s)['shape'][1]//2,next(r for r in nodes if r['stage']==s)['shape'][1]-1]] for s in STAGES]),'',
       '40个stage/crop的捕获节点与九阶段接口tensor为同一对象；D3/D2 skip融合等式精确成立。主目标对接口和实际节点的GPU特征梯度逐项相同。D3/D2在融合后下游路径处削弱，辅助头本身loss不参与C0评分；CPU反例覆盖这个区别。特征梯度和反卷积/卷积参数梯度分开保存、分开解释。','',
       '## 三剂量一阶近似','',f'360个单通道干预、1440个独立目标差分；全部能量匹配，无alpha超限失败。实际F_before-F_after的L2相对目标最大差{energy_error:.3e}，实际向量与公式向量的舍入L2相对预算最大{rounding:.3e}。alpha<=0.5，M/alpha/位置对/Teacher目标均冻结，不对这些选择过程求导。','',
       table(['域','剂量','共同可辨识观测','median相对残差','median绝对残差','median epsilon'],dose_rows),'',
       '共同观测按预设条件：三档有效、最小档abs(Delta_hat)>5epsilon、alpha<=0.05。L 50/60、U 56/60；这是数值可辨识筛选，不是未来通道选择。L在0.03到0.10的相对残差不严格单调，不能宣称每档/每通道都满足理想二阶缩放；两域最小档均比最大档更准确并通过预先门槛。','',
       table(['域','主目标','最小档可判定实际变化数','符号准确率','C0局部门槛'],[[g['domain'],g['objective'],g['resolved_small_sign_cases'],g['small_dose_sign_accuracy'],g['status']] for g in gates]),'',
       '符号准确率只针对超出不确定区间的真实变化，不把不可判定样本算正确。局部预测使用本crop本loss梯度，不能冒称不依赖标签的通道发现，也没有重新排名CHAP通道或证明优于CHAP。','',
       '## 预定数值不确定区间','',
       '每crop/目标非零干预前有20个null观测；epsilon=max(1e-9,1e-6*max(abs(L0),1e-3),5*最大null差)。区间闭合为[-epsilon,+epsilon]。实测null差约0至1e-16，epsilon主要来自保守精度下限，不应误称观察到同等幅度的随机噪声；门槛从未按消融结果调整。','',
       table(['域','目标','基线loss中位数','epsilon最小','epsilon最大','null最大差'],loss_rows),'',
       f'全部1440条有效loss差分：负向{summary["actual_directions"]["negative"]}、正向{summary["actual_directions"]["positive"]}、方向不可判定{summary["actual_directions"]["indeterminate"]}（{100*summary["indeterminate_fraction"]:.4f}%）。没有强制划分正负。真实crop均有有效位置对；空边界/无效通道/近零效应由CPU合同测试和显式状态覆盖。','',
       table(['域','阶段','剂量','signed rho','净幅值rho','符号准确率','balanced准确率','多数方向基线','不可判定/有效'],[[r['domain'],r['stage'],num(r,'dose'),num(r,'spearman'),num(r,'absolute_net_strength_spearman'),num(r,'sign_accuracy'),num(r,'balanced_sign_accuracy'),num(r,'majority_direction_baseline'),r['actual_indeterminate']+'/'+r['valid']] for r in primary]),'',
       '所有参照目标和逐点绝对GxA幅值相关性另存fidelity_by_stage_dose.csv；|sum(gFM)|与sum(|gFM|)含义不同，不作替换或翻转符号。rho基于有限数值观测，含方向不可判定者；符号准确率另排除实际不可判定者，门槛另用共同可辨识集。不能由rho推断弱信号有可靠方向或边界语义。','',
       '## 最终预测与A/B兼容','',
       '每个L干预还计算最终D1整体CE、真实宽2边界CE、hard/soft Dice、1体素容忍轮廓F1、预测改变比例，见intervention_metrics.csv；FP64主CE差与FP32传统指标分开命名。U没有GT，准确性指标明确不可用，只记录预测改变与一致性目标，没有读取U标签。未进行完整体积独立评价。','',
       f'8crop原A四头loss、原B四头loss均重放；80个原A/B最终头到五阶段参数梯度兼容，最大相对L2差{verification["maximum_AB_parameter_relative_error"]:.3e}。C0同公式FP64损失与原FP32的舍入差逐项保留，不伪称bitwise一致。','',
       '原始主目标feature activation/gradient/M保存40组，独立卷积参数梯度160组，另有8个native pair档案。特征/参数统计各9728行。大型raw数组留在本地独立目录并纳入SHA清单，Git忽略；独立numpy重算360个主目标Delta_hat最大绝对差8.674e-19。','',
       f'![剂量与残差]({run.relative_to(ROOT).as_posix()}/figures/dose_fidelity.png)','',
       '## 解释范围与停止点','','C0通过只确认归因实现和局部近似工具可用。没有执行C1的困难度匹配边界专属性、C2的Teacher增强可靠性、C3的通道组或任何正式训练。不能把同目标Taylor准确性叫作可靠边界通道，当前仍无足够证据训练新的边界监督模型。本次到C0停止。']
    (ROOT/'SMOKE_REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(run/'report_results.json',dict(status='PASS',figures=3,formats=['PNG','PDF'],deliverables=['DESIGN.md','TEST_REPORT.md','SMOKE_REPORT.md']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run)
