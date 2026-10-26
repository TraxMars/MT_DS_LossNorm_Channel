"""Identical signed-gradient statistics for both datasets; no channel selection."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import PROJECT,ROOT,SPECS,clean,sha256,write_json
from diagnostic import SCALES,gradient_validity,signed_cosine
from stats_tools import bootstrap_pair_mean,describe,pairwise_cosine

def write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(clean(rows))

def mean(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return float(x.mean()) if x.size else None

def median(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None

def ratio(l,u):
    ln,ls,_=gradient_validity(l);un,us,_=gradient_validity(u)
    valid=(ls=='valid')&(us=='valid')
    out=np.full(np.broadcast_shapes(ln.shape,un.shape),np.nan)
    np.divide(np.linalg.norm(l+u,axis=-1),ln+un,out=out,where=valid)
    return np.clip(out,0,1)

def reachable(stage,index): return stage=='E3' or index==3

def summarize_dataset(ds,run):
    out=run/ds;design=json.loads((out/'design.json').read_text())
    obs=json.loads((out/'observations.json').read_text())
    arrays={o['index']:{s:np.load(out/o['raw_file'])[s].astype(np.float64) for s in ('E3','D1')} for o in obs}
    normrows=[];crossrows=[];multiscale=[]
    reconstruction=[]
    for o in obs:
        for c in o['checks']['gradient_reconstruction']: reconstruction.append(c)
        for stage in ('E3','D1'):
            values=arrays[o['index']][stage];factor=o['checks']['normalized_scale_factor']
            for h,head in enumerate(SCALES):
                if not reachable(stage,h): continue
                norms,status,floor=gradient_validity(values[h])
                for c in range(len(norms)):
                    normrows.append(dict(observation=o['index'],mode=o['mode'],unit=o['unit'],role=o['role'],
                        stage=stage,head=head,channel=c,status=status[c],validity_floor=floor[c],
                        raw_norm=norms[c],raw_mean_contribution_norm=norms[c]/4,
                        raw_weighted_contribution_norm=norms[c]*(1 if o['role']=='L' else .3)/4,
                        lossnorm_contribution_norm=norms[c]*factor,normalization_factor=factor))
            indices=[h for h in range(4) if reachable(stage,h)]
            norms,status,_=gradient_validity(values[indices])
            valid=(status=='valid').all(axis=0)
            ratios=np.full(values.shape[1],np.nan)
            np.divide(np.linalg.norm(values[indices].sum(axis=0),axis=-1),norms.sum(axis=0),out=ratios,where=valid)
            ratios=np.clip(ratios,0,1)
            for c in range(values.shape[1]):
                multiscale.append(dict(observation=o['index'],mode=o['mode'],unit=o['unit'],role=o['role'],stage=stage,
                    channel=c,status='valid' if valid[c] else 'invalid_component',reachable_heads=len(indices),
                    raw_sum_norm=np.linalg.norm(values[indices,c].sum(axis=0)),
                    lossnorm_sum_norm=np.linalg.norm(values[indices,c].sum(axis=0))*factor,
                    sum_to_norms_ratio=ratios[c],cancellation_degree=1-ratios[c]))
            for h,j in itertools.combinations(indices,2):
                cos,valid,_,_=signed_cosine(values[h],values[j])
                for c in range(values.shape[1]):
                    crossrows.append(dict(observation=o['index'],mode=o['mode'],unit=o['unit'],role=o['role'],
                        stage=stage,head_a=SCALES[h],head_b=SCALES[j],channel=c,valid=valid[c],raw_cosine=cos[c],
                        lossnorm_cosine=cos[c]))
    write_csv(out/'observation_norms.csv',normrows)
    write_csv(out/'cross_head_relations.csv',crossrows)
    write_csv(out/'multiscale_cancellation.csv',multiscale)
    perchannel=[];pairrows=[];scale_summary=[];stability=[]
    for stage in ('D1','E3'):
        means={};case_raw={};case_obs={}
        for role in ('L','U'):
            selected=[o for o in obs if o['unit']=='case' and o['role']==role]
            case_obs[role]=selected
            values=np.stack([arrays[o['index']][stage] for o in selected])
            n=len(selected)//2
            case_raw[role]=values
            means[role]={
                'raw':values.reshape(n,2,*values.shape[1:]).mean(axis=1),
                'lossnorm':(values*np.array([o['checks']['normalized_scale_factor'] for o in selected])[:,None,None,None]).reshape(n,2,*values.shape[1:]).mean(axis=1)}
        for h,head in enumerate(SCALES):
            if not reachable(stage,h):
                for c in range(means['L']['raw'].shape[2]):
                    perchannel.append(dict(stage=stage,head=head,channel=c,policy='raw',status='unreachable'))
                continue
            for role in ('L','U'):
                values=case_raw[role][:,h]
                within,valid,_,_=signed_cosine(values[0::2],values[1::2])
                between,_,_,_=pairwise_cosine(means[role]['raw'][:,h],means[role]['raw'][:,h])
                offdiag=np.triu_indices(between.shape[1],1)
                stability.append(dict(stage=stage,head=head,role=role,within_crop_median=median(within),
                                      between_case_median=median(between[:,offdiag[0],offdiag[1]]),
                                      valid_within_fraction=float(valid.mean())))
            scores_by_policy={}
            for policy in ('raw','lossnorm'):
                left,right=means['L'][policy][:,h],means['U'][policy][:,h]
                scores,valid,ls,us=pairwise_cosine(left,right)
                scores_by_policy[policy]=scores
                bounds=bootstrap_pair_mean(scores,seed=142,draws=200)
                # Raw contribution to original unnormalized objective includes 1/4 and U weight .3.
                l=left[:,None]/4 if policy=='raw' else left[:,None]
                u=right[None,:]*.3/4 if policy=='raw' else right[None,:]
                cancellation=ratio(l,u).transpose(2,0,1)
                norms_l=np.linalg.norm(left,axis=-1);norms_u=np.linalg.norm(right,axis=-1)
                for c in range(left.shape[1]):
                    observed=scores[c][valid[c]];coverage=float(valid[c].mean())
                    status='valid' if coverage>=.75 else 'low_coverage'
                    row=dict(stage=stage,head=head,channel=c,policy=policy,status=status,
                             valid_pair_fraction=coverage,valid_pairs=int(valid[c].sum()),total_pairs=valid[c].size,
                             lu_mean=mean(observed),lu_median=median(observed),
                             negative_pair_fraction=float((observed<0).mean()) if observed.size else None,
                             bootstrap_low=bounds[0,c],bootstrap_high=bounds[1,c],
                             exploratory_negative_interval=bool(status=='valid' and bounds[1,c]<0),
                             L_norm_median=median(norms_l[:,c]),U_norm_median=median(norms_u[:,c]),
                             sum_to_norms_ratio_median=median(cancellation[c]),
                             cancellation_degree_median=None if median(cancellation[c]) is None else 1-median(cancellation[c]))
                    perchannel.append(row)
                    for i in range(len(left)):
                        for j in range(len(right)):
                            pairrows.append(dict(stage=stage,head=head,channel=c,policy=policy,
                                L_case=design['labeled_ids'][i],U_case=design['unlabeled_ids'][j],
                                cosine=scores[c,i,j],L_status=ls[i,c],U_status=us[j,c],
                                sum_to_norms_ratio=cancellation[c,i,j]))
                rows=[r for r in perchannel if r['stage']==stage and r['head']==head and r['policy']==policy and r['status']!='unreachable']
                scale_summary.append(dict(stage=stage,head=head,policy=policy,channels=len(rows),
                    valid_channels=sum(r['status']=='valid' for r in rows),
                    valid_pair_fraction=float(valid.mean()),
                    median_channel_lu_mean=median([r['lu_mean'] for r in rows if r['status']=='valid']),
                    negative_pair_fraction=float(np.mean(scores[valid]<0)) if valid.any() else None,
                    negative_interval_channels=sum(r['exploratory_negative_interval'] for r in rows),
                    sum_to_norms_ratio_median=median(cancellation)))
            raw,effective=scores_by_policy['raw'],scores_by_policy['lossnorm']
            for row in perchannel:
                if row['stage']==stage and row['head']==head and row['status']!='unreachable':
                    c=row['channel'];a,b=mean(raw[c]),mean(effective[c])
                    row['raw_lossnorm_mean_difference']=None if a is None or b is None else b-a
    write_csv(out/'per_channel.csv',perchannel);write_csv(out/'lu_case_pairs.csv',pairrows)
    write_csv(out/'scale_summary.csv',scale_summary);write_csv(out/'case_crop_stability.csv',stability)
    # Matched 2L+2U units. Six overlapping/repeated units are descriptive, not six iid subjects.
    mini=[]
    for mode in ('eval','train'):
        for b in range(3):
            for repeat in range(2):
                selected={o['role']:o for o in obs if o['unit']=='mini' and o['mode']==mode and o['batch']==b and o['repeat']==repeat}
                for stage in ('D1','E3'):
                    for h,head in enumerate(SCALES):
                        if not reachable(stage,h): continue
                        for policy in ('raw','lossnorm'):
                            a=arrays[selected['L']['index']][stage][h]
                            z=arrays[selected['U']['index']][stage][h]
                            l=a/4 if policy=='raw' else a*selected['L']['checks']['normalized_scale_factor']
                            u=z*.3/4 if policy=='raw' else z*selected['U']['checks']['normalized_scale_factor']
                            scores,valid,ls,us=signed_cosine(l,u);ratios=ratio(l,u)
                            ln=np.linalg.norm(l,axis=-1);un=np.linalg.norm(u,axis=-1)
                            for c in range(len(scores)):
                                mini.append(dict(mode=mode,batch=b,repeat=repeat,stage=stage,head=head,policy=policy,channel=c,
                                    precision=selected['L'].get('gradient_precision','torch.float32'),
                                    cosine=scores[c],valid=valid[c],L_status=ls[c],U_status=us[c],L_norm=ln[c],U_norm=un[c],
                                    U_L_norm_ratio=un[c]/ln[c] if valid[c] else None,
                                    sum_to_norms_ratio=ratios[c],cancellation_degree=1-ratios[c]))
    write_csv(out/'mini_relations.csv',mini)
    mini_summary=[];mini_channels=[]
    for mode,stage,head,policy in itertools.product(('eval','train'),('D1','E3'),SCALES,('raw','lossnorm')):
        selected=[r for r in mini if r['mode']==mode and r['stage']==stage and r['head']==head and r['policy']==policy]
        if not selected:continue
        valid=[r for r in selected if r['valid']]
        mini_summary.append(dict(mode=mode,stage=stage,head=head,policy=policy,
            valid_fraction=len(valid)/len(selected),median_cosine=median([r['cosine'] for r in valid]),
            negative_fraction=mean([r['cosine']<0 for r in valid]),
            median_sum_to_norms_ratio=median([r['sum_to_norms_ratio'] for r in valid]),
            median_U_L_norm_ratio=median([r['U_L_norm_ratio'] for r in valid])))
        for c in sorted(set(r['channel'] for r in selected)):
            rows=[r for r in selected if r['channel']==c and r['valid']]
            mini_channels.append(dict(mode=mode,stage=stage,head=head,policy=policy,channel=c,valid_units=len(rows),
                negative_units=sum(r['cosine']<0 for r in rows),mean_cosine=mean([r['cosine'] for r in rows]),
                minimum= min((r['cosine'] for r in rows),default=None),maximum=max((r['cosine'] for r in rows),default=None)))
    write_csv(out/'mini_summary.csv',mini_summary);write_csv(out/'mini_channel_stability.csv',mini_channels)
    # Direct mode differences use identical cases and crop starts; BN+Dropout change together.
    indexed={(r['mode'],r['batch'],r['repeat'],r['stage'],r['head'],r['policy'],r['channel']):r for r in mini}
    differences=[]
    for key,r in indexed.items():
        if key[0]!='train':continue
        e=indexed[('eval',*key[1:])]
        differences.append(dict(batch=r['batch'],repeat=r['repeat'],stage=r['stage'],head=r['head'],policy=r['policy'],channel=r['channel'],
            eval_precision=e['precision'],train_precision=r['precision'],
            eval_cosine=e['cosine'],train_cosine=r['cosine'],cosine_difference=r['cosine']-e['cosine'] if r['valid'] and e['valid'] else None))
    write_csv(out/'mode_comparison.csv',differences)
    # Same-case/crop replay of historical LA raw gradients (backend tolerance is reported, not hidden).
    historical=None
    if ds=='LA':
        old=PROJECT/'analysis/stage1a_gradients/runs/la10_iter15000_seed142_fp32'
        errors=[]
        for role in ('L','U'):
            selected=[o for o in obs if o['unit']=='case' and o['role']==role]
            for stage in ('D1','E3'):
                original=np.load(old/'raw'/f'{role}_{stage}.npy',mmap_mode='r')
                for index,o in enumerate(selected):
                    for h,head in enumerate(SCALES):
                        if not reachable(stage,h):continue
                        a=arrays[o['index']][stage][h];z=np.asarray(original[index,h],dtype=np.float64)
                        errors.append(dict(role=role,case=o['cases'][0],repeat=o['repeat'],stage=stage,head=head,
                            relative_l2=float(np.linalg.norm(a-z)/max(np.linalg.norm(z),1e-30)),max_abs=float(np.max(np.abs(a-z)))))
        write_csv(out/'historical_gradient_replay.csv',errors)
        historical=dict(max_relative_l2=max(r['relative_l2'] for r in errors),
                        median_relative_l2=median([r['relative_l2'] for r in errors]),
                        same_checkpoint_cases_crops=True,backend='historical CUDA fp32 versus current CPU fp32')
    cross_summary=[]
    for role,a,b in itertools.product(('L','U'),SCALES,SCALES):
        rows=[r for r in crossrows if r['unit']=='case' and r['role']==role and r['head_a']==a and r['head_b']==b and r['valid']]
        if rows:cross_summary.append(dict(role=role,head_a=a,head_b=b,median_cosine=median([r['raw_cosine'] for r in rows]),
                                         negative_fraction=mean([r['raw_cosine']<0 for r in rows])))
    write_csv(out/'cross_head_summary.csv',cross_summary)
    priority=[r for r in perchannel if r['stage']=='D1' and r['head']=='D1' and r['policy']=='raw' and r['channel'] in (0,4,10,14)] if ds=='LA' else []
    result=dict(dataset=ds,status='PASS',scale_summary=scale_summary,mini_summary=mini_summary,
                case_crop_stability=stability,cross_head_summary=cross_summary,historical_replay=historical,
                LA_predeclared_channels=priority,observations=len(obs),
                fp64_fallback_units=json.loads((out/'execution.json').read_text())['fp64_fallback_units'],
                fp64_observations=sum(o.get('gradient_precision')=='torch.float64' for o in obs),
                reconstruction_max_relative_raw=max(r['raw_relative_l2_error'] for r in reconstruction),
                reconstruction_max_relative_lossnorm=max(r['normalized_relative_l2_error'] for r in reconstruction),
                invalid_observation_channels=sum(r['status']!='valid' for r in normrows),
                observation_channels=len(normrows),raw_lossnorm_case_mean_max_abs_difference=max(
                    (abs(r['raw_lossnorm_mean_difference']) for r in perchannel if r.get('raw_lossnorm_mean_difference') is not None),default=0),
                limitations=['single checkpoint/split','16 U cases per dataset','two uniform crops per case',
                    'three mini case groups and two crop/Dropout draws; no mini inferential CI',
                    'LA mini uses six of eight labeled cases','BN and Dropout effects not separated',
                    'no task outcome or causal intervention','unadjusted exploratory case bootstrap intervals'])
    write_json(out/'summary.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve()
    assert run.is_relative_to(ROOT/'D1_gradient_diagnosis/results') and json.loads((run/'execution.json').read_text())['status']=='PASS'
    assert not (run/'comparison.json').exists(),'Use a fresh summary output; never overwrite a previous completed summary'
    results={ds:summarize_dataset(ds,run) for ds in SPECS}
    write_json(run/'comparison.json',results)
    write_json(run/'summary_source.json',dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)))
    print({ds:r['observations'] for ds,r in results.items()})

if __name__=='__main__':main()
