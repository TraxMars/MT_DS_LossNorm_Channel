"""Case-level summaries of discovery scores and energy-matched probe outcomes."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from common import require_stage1b_run,write_csv,write_json,fixed_rng,describe


def read_csv(path):return list(csv.DictReader(Path(path).open()))
def mean(values):return float(np.mean(values)) if values else None

def ci(values,*seed):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if len(x)==0:return dict(mean=None,low=None,high=None,n_cases=0)
    # Shared resampling indices keep all paired contrasts comparable and give
    # identical intervals to identical channel sets, regardless of method name.
    indices=fixed_rng(142,'probe_case_bootstrap',len(x)).integers(0,len(x),size=(1000,len(x)))
    samples=x[indices].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),n_cases=len(x))


def rho(x,y):
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<3 or np.std(x[good])==0 or np.std(y[good])==0:return None
    return float(spearmanr(x[good],y[good]).statistic)


def summarize(run,ablation):
    run,ablation=require_stage1b_run(run),require_stage1b_run(ablation)
    result=json.loads((ablation/'ablation_results.json').read_text());assert result['status']=='PASS' and not result['smoke']
    single=read_csv(ablation/'singletons.csv');groups=read_csv(ablation/'groups.csv');ranks=read_csv(run/'rankings.csv')
    # Supplementary influence magnitude, kept separate from the signed primary
    # CE endpoint. No channel selections are changed after viewing these values.
    for row in single+groups:
        if row.get('boundary_ce_delta'):row['abs_boundary_ce_delta']=str(abs(float(row['boundary_ce_delta'])))
    stages=list(dict.fromkeys(r['stage'] for r in single));cases=list(dict.fromkeys(r['case'] for r in single))
    metrics=['ce_delta','boundary_ce_delta','dice_delta','soft_dice_delta','boundary_f1_delta','changed_prediction_fraction','abs_boundary_ce_delta']
    score={(s,m):{int(r['channel']):float(r['score']) for r in ranks if r['stage']==s and r['method']==m and r['score']} for s in stages for m in ('A','B1','B2','B3')}
    observations=defaultdict(list)
    for row in single:
        if row['status']!='valid':continue
        for metric in metrics:
            if row.get(metric):observations[(row['stage'],row['region'],int(row['channel']),row['case'],metric)].append(float(row[metric]))
    effects={key:mean(values) for key,values in observations.items()}
    correlations=[]
    for stage in stages:
        for method in ('A','B1','B2','B3'):
            for contrast in ('B','R','I','B-R','B-I'):
                for metric in metrics:
                    values=[]
                    for case in cases:
                        xx,yy=[],[]
                        for channel,s in score[(stage,method)].items():
                            b=effects.get((stage,'B',channel,case,metric))
                            value=effects.get((stage,contrast,channel,case,metric))
                            if '-' in contrast:
                                control=effects.get((stage,contrast[-1],channel,case,metric))
                                value=b-control if b is not None and control is not None else None
                            if value is not None:xx.append(s);yy.append(value)
                        r=rho(np.array(xx),np.array(yy))
                        if r is not None:values.append(r)
                    correlations.append(dict(stage=stage,method=method,region_contrast=contrast,metric=metric,
                                             **ci(values,stage,method,contrast,metric)))
    write_csv(ablation/'score_predictiveness.csv',correlations)
    # Each case contributes its mean over crops; random groups are averaged within
    # that case, not treated as eight new independent experimental subjects.
    obs=defaultdict(list)
    for row in groups:
        if row['status']!='valid':continue
        for metric in metrics+['ce_interaction','boundary_ce_interaction','logit_interaction_rms','logit_joint_delta_rms']:
            if row.get(metric):obs[(row['stage'],row['region'],row['kind'],row['method'],row['case'],metric)].append(float(row[metric]))
    group_means={key:mean(values) for key,values in obs.items()}
    random=defaultdict(list)
    for (s,reg,kind,method,case,metric),value in group_means.items():
        if kind=='random':random[(s,reg,case,metric)].append(value)
    random={key:mean(value) for key,value in random.items()}
    summary,contrasts,specificity=[],[],[]
    methods=sorted({(r['kind'],r['method']) for r in groups if r['kind']!='random'})
    for stage in stages:
        for region in ('B','R','I'):
            for kind,method in methods+[('random','random_mean')]:
                for metric in metrics+['ce_interaction','boundary_ce_interaction','logit_interaction_rms','logit_joint_delta_rms']:
                    values=[];diff=[]
                    for case in cases:
                        r=random.get((stage,region,case,metric))
                        x=r if kind=='random' else group_means.get((stage,region,kind,method,case,metric))
                        if x is not None:values.append(x)
                        if kind!='random' and x is not None and r is not None:diff.append(x-r)
                    summary.append(dict(stage=stage,region=region,kind=kind,method=method,metric=metric,**ci(values,stage,region,kind,method,metric)))
                    if kind!='random':contrasts.append(dict(stage=stage,region=region,kind=kind,method=method,metric=metric,comparison='minus_same_K_energy_random',**ci(diff,stage,region,kind,method,metric,'contrast')))
        for kind,method in methods:
            for control in ('R','I'):
                for metric in ('boundary_ce_delta','boundary_f1_delta','ce_delta','abs_boundary_ce_delta','changed_prediction_fraction'):
                    values=[]
                    for case in cases:
                        b=group_means.get((stage,'B',kind,method,case,metric));c=group_means.get((stage,control,kind,method,case,metric))
                        rb=random.get((stage,'B',case,metric));rc=random.get((stage,control,case,metric))
                        if all(x is not None for x in (b,c,rb,rc)):values.append((b-c)-(rb-rc))
                    specificity.append(dict(stage=stage,kind=kind,method=method,control=control,metric=metric,comparison='(boundary-control) minus random-set equivalent',**ci(values,stage,kind,method,control,metric)))
    write_csv(ablation/'group_summary.csv',summary);write_csv(ablation/'group_vs_random.csv',contrasts)
    write_csv(ablation/'group_specificity.csv',specificity)
    against_A=[]
    for stage in stages:
        for region in ('B','R','I'):
            for kind,method in methods:
                if method=='A':continue
                for metric in metrics+['boundary_ce_interaction','logit_interaction_rms']:
                    differences=[]
                    for case in cases:
                        a=group_means.get((stage,region,kind,'A',case,metric))
                        b=group_means.get((stage,region,kind,method,case,metric))
                        if a is not None and b is not None:differences.append(b-a)
                    against_A.append(dict(stage=stage,region=region,kind=kind,method=method,metric=metric,
                                          comparison='minus_A_same_kind_K_energy',**ci(differences)))
    write_csv(ablation/'group_vs_A.csv',against_A)
    # Between-probe-case repeatability of singleton response rankings.
    stability=[]
    for stage in stages:
        for region in ('B','R','I'):
            channels=sorted({key[2] for key in effects if key[0]==stage})
            for metric in ('boundary_ce_delta','boundary_f1_delta','ce_delta'):
                values=[]
                for a,case in enumerate(cases):
                    for other in cases[a+1:]:
                        x=np.array([effects.get((stage,region,c,case,metric),np.nan) for c in channels]);y=np.array([effects.get((stage,region,c,other,metric),np.nan) for c in channels])
                        r=rho(x,y)
                        if r is not None:values.append(r)
                stability.append(dict(stage=stage,region=region,metric=metric,**describe(values)))
    write_csv(ablation/'singleton_case_stability.csv',stability)
    output=dict(status='PASS',probe_cases=len(cases),singleton_invalid=sum(r['status']!='valid' for r in single),
                group_invalid=sum(r['status']!='valid' for r in groups),bootstrap='1000 resamples of 4 probe cases; crops/groups remain within case',
                supplemental_endpoints='Absolute boundary CE delta and score-vs-prediction-change are descriptive sensitivity endpoints, not selection criteria',
                all_results_exploratory=True,not_independent_generalization=True)
    write_json(ablation/'summary_results.json',output)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--ablation',type=Path,required=True)
    a=p.parse_args();print(json.dumps(summarize(a.run.resolve(),a.ablation.resolve()),indent=2))
