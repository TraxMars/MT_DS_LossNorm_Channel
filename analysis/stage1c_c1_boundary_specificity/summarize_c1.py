"""Case-level, paired exploratory summaries. No channel re-selection on probe."""
from collections import defaultdict
from itertools import combinations
import json
import numpy as np
import pandas as pd
from bridge import ROOT,STAGES,PRIMARY,DOSES,b,write_json,write_csv,fixed_rng,split,direction
from measures import rho,bootstrap,cosine_rows

RUNS=ROOT/'runs'

def load(name):
    return pd.concat([pd.read_csv(RUNS/p/(name+'.csv')).assign(split=p) for p in ('discovery','probe')],ignore_index=True)

def mean_finite(x):
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None

def attribution_summary():
    actual=load('interventions');local=load('local_attribution');plan=json.loads((RUNS/'discovery/FROZEN_PLAN.json').read_text())
    rows=[]
    for (part,case,stage,support,dose),block in actual.groupby(['split','case','stage','support','dose'],sort=False):
        valid=block[block.status=='valid']
        # Two original crops required for a case/channel estimate; failures
        # remain in raw outputs and denominator, never filled with zero.
        counts=valid.groupby('channel')['index'].nunique();channels=counts[counts==2].index
        if not len(channels):continue
        case_values=valid[valid.channel.isin(channels)].groupby('channel').mean(numeric_only=True).sort_index()
        y=case_values['boundary_delta'].to_numpy();eps=case_values['epsilon'].to_numpy();ch=case_values.index
        actual_directions=np.array([direction(v,e) for v,e in zip(y,eps)])
        for objective in PRIMARY:
            for mode in ('local_GT_gradient','discovery_B_transfer','discovery_support_transfer'):
                if mode=='local_GT_gradient':
                    sub=local[(local.split==part)&(local.case==case)&(local.stage==stage)&(local.support==support)&(local.dose==dose)&(local.objective==objective)&(local.status=='valid')]
                    pred=sub.groupby('channel')['delta_hat'].mean().reindex(ch).to_numpy()
                else:
                    source='B' if mode=='discovery_B_transfer' else support
                    pred=np.array([plan['scores'][stage][source][objective][str(c)] for c in ch])*case_values['target_l2'].to_numpy()
                sign_pred=np.array([direction(v,e) for v,e in zip(pred,eps)])
                decidable=(actual_directions!='indeterminate')&(actual_directions!='undefined')
                both=decidable&(sign_pred!='indeterminate')&(sign_pred!='undefined')
                signed,status=rho(pred,y);absolute,abs_status=rho(abs(pred),abs(y))
                row=dict(split=part,case=case,stage=stage,support=support,dose=dose,objective=objective,mode=mode,
                    candidate_channels=len(plan['stages'][stage]['channels']),paired_channels=len(ch),signed_spearman=signed,correlation_status=status,
                    absolute_spearman=absolute,absolute_status=abs_status,actual_indeterminate_fraction=float((~decidable).mean()),
                    both_direction_decidable_fraction=float(both.mean()),direction_accuracy=float((sign_pred[both]==actual_directions[both]).mean()) if both.any() else None,
                    actual_absolute_mean=float(abs(y).mean()),predicted_absolute_mean=float(abs(pred).mean()))
                for metric in ('boundary_f1_delta','dice_delta','region_delta','changed_prediction_fraction'):
                    # Higher boundary CE is worse. For contour/Dice, reverse
                    # metric difference solely to express degradation, not to
                    # re-rank contributions or rescue a poor signed score.
                    effect=case_values[metric].to_numpy();degradation=-effect if metric in ('boundary_f1_delta','dice_delta') else effect
                    row[metric+'_rho_degradation'],row[metric+'_rho_status']=rho(pred,degradation)
                    row[metric+'_abs_rho'],_=rho(abs(pred),abs(effect))
                    row[metric+'_mean']=float(effect.mean())
                    row[metric+'_nonzero_fraction']=float((effect!=0).mean())
                rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'attribution_by_case.csv',index=False,lineterminator='\n')
    summaries=[];paired=[];probe=split()['probe'];ids=fixed_rng(142,'C1_case_bootstrap').integers(0,4,(2000,4))
    np.save(ROOT/'bootstrap_case_indices.npy',ids)
    metrics=['signed_spearman','absolute_spearman','direction_accuracy','actual_indeterminate_fraction','both_direction_decidable_fraction',
             'boundary_f1_delta_rho_degradation','boundary_f1_delta_abs_rho','dice_delta_rho_degradation','region_delta_rho_degradation']
    for key,block in df[df.split=='probe'].groupby(['stage','support','dose','objective','mode'],sort=False):
        base=dict(zip(['stage','support','dose','objective','mode'],key));block=block.set_index('case').reindex(probe)
        for metric in metrics:
            values=block[metric].to_numpy();record=bootstrap(values,ids)
            summaries.append(dict(base,metric=metric,available_cases=int(np.isfinite(values).sum()),descriptive_available_case_mean=mean_finite(values),**record))
    primary=df[(df.split=='probe')&(df.support=='B')&(df.dose==.03)&(df['mode']=='discovery_B_transfer')]
    for stage in STAGES:
        for other in ('region','boundary','permuted','far'):
            a=primary[(primary.stage==stage)&(primary.objective=='transition')].set_index('case').reindex(probe)
            c=primary[(primary.stage==stage)&(primary.objective==other)].set_index('case').reindex(probe)
            for metric in ('signed_spearman','absolute_spearman','direction_accuracy','boundary_f1_delta_rho_degradation'):
                v=a[metric].to_numpy()-c[metric].to_numpy()
                paired.append(dict(stage=stage,contrast='transition-minus-'+other,metric=metric,**bootstrap(v,ids)))
    write_json(ROOT/'attribution_summary.json',summaries);write_json(ROOT/'paired_case_comparisons.json',paired)
    # Faithfulness (secondary) at all fixed doses, separated from transfer.
    faith=[]
    for key,block in local[local.status=='valid'].groupby(['split','case','stage','support','dose','objective'],sort=False):
        eps=block.epsilon.to_numpy();p=block.delta_hat.to_numpy();y=block.delta_actual.to_numpy()
        rel=abs(y-p)/np.maximum(abs(p),eps)
        faith.append(dict(zip(['split','case','stage','support','dose','objective'],key)),)
        faith[-1].update(median_relative_residual=float(np.median(rel)),absolute_error=float(abs(y-p).mean()),n=len(block))
    write_csv(ROOT/'local_self_fidelity_secondary.csv',faith)
    return df,actual


def gradient_summary():
    df=load('gradient_relations');norms=load('parameter_norms');rows=[]
    primary_pairs={frozenset(('transition',n)) for n in ('region','boundary','permuted','far','transition_m0_5','transition_m2','permuted_1','permuted_2','far_matched','true_common')}
    df=df[[frozenset((a,c)) in primary_pairs for a,c in zip(df.left,df.right)]].copy()
    df['relative_l2_difference']=np.sqrt(np.maximum(0,df.left_norm**2+df.right_norm**2-2*df.left_norm*df.right_norm*df.signed_cosine))/df.left_norm
    for key,block in df.groupby(['split','case','stage','left','right','coordinate_kind'],sort=False):
        per_crop=block.groupby('index').agg(cosine=('signed_cosine','median'),relative_difference=('relative_l2_difference','median'),coverage=('signed_cosine','count'),total=('channel','count'))
        rows.append(dict(zip(['split','case','stage','left','right','coordinate_kind'],key)),)
        rows[-1].update(median_channel_cosine_mean_crops=float(per_crop.cosine.mean()),relative_l2_difference_mean_crops=float(per_crop.relative_difference.mean()),
                       valid_fraction=float(per_crop.coverage.sum()/per_crop.total.sum()),n_crops=len(per_crop))
    write_csv(ROOT/'gradient_relations_by_case.csv',rows)
    n=norms.groupby(['split','case','stage','objective','status']).size().reset_index(name='channel_crop_count');n.to_csv(ROOT/'parameter_validity.csv',index=False)
    return pd.DataFrame(rows)


def stability():
    d=load('contributions');d=d[d.objective.isin(PRIMARY)];d['unit_response']=pd.to_numeric(d.unit_response,errors='coerce')
    rows=[];parameter=[];cases=split()['discovery']+split()['probe']
    for s in STAGES:
        for objective in PRIMARY:
            for support in ('B','I'):
                sub=d[(d.stage==s)&(d.objective==objective)&(d.support==support)]
                vectors={}
                for case in cases:
                    block=sub[sub.case==case];pivot=block.pivot(index='channel',columns='repeat',values='unit_response')
                    val,status=rho(pivot[0],pivot[1]);rows.append(dict(stage=s,objective=objective,support=support,comparison='within_case_crops',left=case,right=case,
                        coordinate_kind='spatially_reduced_feature_contribution',spearman=val,status=status))
                    vectors[case]=pivot.mean(axis=1)
                for a,c in combinations(cases,2):
                    val,status=rho(vectors[a],vectors[c]);rows.append(dict(stage=s,objective=objective,support=support,comparison='between_cases',left=a,right=c,
                        coordinate_kind='spatially_reduced_feature_contribution',spearman=val,status=status))
            pvec={}
            for case in cases:
                case_sub=d[(d.case==case)&(d.stage==s)&(d.objective==objective)&(d.support=='B')]
                part=case_sub.iloc[0]['split'];indices=sorted(case_sub['index'].unique())
                arrays=[np.load(RUNS/part/'raw/parameters'/f'L_{i}_{objective}_{s}.npy') for i in indices]
                cos,_,_,status=cosine_rows(*arrays)
                for c,value in enumerate(cos):parameter.append(dict(stage=s,objective=objective,comparison='within_case_crops',left=case,right=case,channel=c,cosine=value,status=status[c]))
                pvec[case]=(arrays[0].astype(float)+arrays[1].astype(float))/2
            for a,c in combinations(cases,2):
                cos,_,_,status=cosine_rows(pvec[a],pvec[c])
                for ch,value in enumerate(cos):parameter.append(dict(stage=s,objective=objective,comparison='between_cases',left=a,right=c,channel=ch,cosine=value,status=status[ch]))
    write_csv(ROOT/'feature_contribution_stability.csv',rows);write_csv(ROOT/'parameter_direction_stability.csv',parameter)


def geometry_summary():
    rows=[];groups=[];margins=[];far=[]
    for part in ('discovery','probe'):
        records=json.loads((RUNS/part/'geometry.json').read_text())
        for r in records:
            key={k:r[k] for k in ('case','index','repeat')};key['split']=part
            for name,s in r['pairs'].items():
                row=dict(key,control=name,status=s['status'],pairs=s['count'])
                if s['count']:
                    row.update(distance_mean=s['distance']['mean'],distance_median=s['distance']['q'][2],fg_reuse=s['fg_reuse_fraction'],bg_reuse=s['bg_reuse_fraction'],
                        gap_mean=s['logit_gap']['mean'],fg_logit_mean=s['fg_logit']['mean'],bg_logit_mean=s['bg_logit']['mean'])
                    for g in s['groups']:
                        item=dict(key,control=name,**g);groups.append(item)
                    for g in s['margins']:margins.append(dict(key,control=name,**g))
                rows.append(row)
            far.append(dict(key,**r['far_matching']))
    write_csv(ROOT/'pair_geometry_summary.csv',rows);write_json(ROOT/'endpoint_classification_groups.json',groups)
    write_csv(ROOT/'endpoint_margin_correctness.csv',margins);write_json(ROOT/'far_matching_summary.json',far)


def main():
    for phase in ('smoke','discovery','probe'):assert json.loads((RUNS/phase/'execution_results.json').read_text())['status']=='PASS'
    df,actual=attribution_summary();g=gradient_summary();stability();geometry_summary()
    result=dict(status='PASS',cases=8,probe_cases=4,crops=16,primary_dose=.03,primary_margin=1.,
        actual_interventions=len(actual),invalid_interventions=int((actual.status!='valid').sum()),
        actual_boundary_direction_indeterminate_fraction=float((actual[actual.status=='valid'].boundary_direction=='indeterminate').mean()),
        probe_scoring='discovery fixed signed response per unit L2; no probe reranking',local_scoring='probe GT-gradient rows separate',
        case_bootstrap_replicates=2000,independent_generalization=False,C2_C3_executed=False)
    write_json(ROOT/'SUMMARY_STATUS.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
