"""Paired A/B gradients and discovery-only channel plans; no model inference."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from common import (require_stage1b_run,ROOT,A_RUN,SCALES,STAGES,case_means,describe,fixed_rng,gradient_validity,masked_mean,
                    observation_patterns,pairwise_cosine,signed_cosine,write_csv,write_json)
from interventions import DETAIL_STAGES


def sources(run,domain,objective,stage):
    path=A_RUN/'raw'/f'{domain}_{stage}.npy' if objective=='A' else run/'raw'/f'{domain}_{objective}_{stage}.npy'
    return np.load(path,mmap_mode='r')


def relations():
    result={'A':('A','A')}
    for w in (1,2,3):
        result.update({f'B1_w{w}':(f'band_w{w}','A'),f'B2_w{w}':('A',f'edge_w{w}'),
                       f'B3_w{w}':(f'band_w{w}',f'edge_w{w}'),f'R_w{w}':(f'random_w{w}',f'edge_w{w}'),
                       f'I_w{w}':(f'core_w{w}',f'edge_w{w}')})
    result['teacher_shuffle_w2']=('band_w2','shuffle_w2')
    return result


def finite_median(values):
    return describe(values)['median']


def rho(left,right):
    valid=np.isfinite(left)&np.isfinite(right)
    if valid.sum()<3 or np.std(left[valid])==0 or np.std(right[valid])==0:return None
    return float(spearmanr(left[valid],right[valid]).statistic)


def chap_score(left,right,absolute,relative):
    gl=left.mean(0);gu=right.mean(0)
    cosine,valid,_,_=signed_cosine(gl,gu,absolute,relative)
    ln,ls,_=gradient_validity(left,absolute,relative)
    un,us,_=gradient_validity(right,absolute,relative)
    coverage_l=(ls=='valid').mean(0);coverage_u=(us=='valid').mean(0)
    _,_,_=ln,un,valid
    nl=np.linalg.norm(gl,axis=1);nu=np.linalg.norm(gu,axis=1)
    valid &= (coverage_l>=.75)&(coverage_u>=.75)
    score=cosine*nu/max(float(np.nanmax(nu)),1e-30)
    score[~valid]=np.nan
    order=sorted(np.flatnonzero(valid),key=lambda c:(-score[c],int(c)))
    return score,cosine,nl,nu,order,coverage_l,coverage_u


def channel_plans(run,stage,design,observations):
    absolute,relative=design['gradient_absolute_floor'],design['gradient_relative_floor']
    disc={d:[i for i,o in enumerate(observations[d]) if o['case'] in design['data_split']['discovery_'+d]] for d in ('L','U')}
    rank_rows,plans,cluster_info,agreements,stability,width_rows=[],[],[],[],[],[]
    all_results={}
    for method,objs in relations().items():
        if method!='A' and not method.startswith(('B1_','B2_','B3_')):continue
        left=np.asarray(sources(run,'L',objs[0],stage)[disc['L'],3],dtype=np.float64).reshape(-1,2,*sources(run,'L',objs[0],stage).shape[2:]).mean(1)
        right=np.asarray(sources(run,'U',objs[1],stage)[disc['U'],3],dtype=np.float64).reshape(-1,2,*sources(run,'U',objs[1],stage).shape[2:]).mean(1)
        score,cosine,nl,nu,order,cl,cu=chap_score(left,right,absolute,relative)
        all_results[method]=(score,order)
        if method!='A' and not method.endswith('_w2'):continue
        name=method.split('_')[0]
        for channel in range(len(score)):
            rank_rows.append(dict(stage=stage,method=name,channel=channel,score=score[channel],signed_cosine=cosine[channel],
                                  L_mean_gradient_norm=nl[channel],U_mean_gradient_norm=nu[channel],coverage_L=cl[channel],coverage_U=cu[channel],
                                  rank=order.index(channel)+1 if channel in order else None))
        if len(order)>=2:
            plans.append(dict(stage=stage,method=name,kind='ranked',channels=[int(c) for c in order[:2]]))
        # Correlation/clustering features: log norms over discovery cases, separately
        # centered/scaled in L/U. This is not the single-head simplex of Stage1A.
        eligible=np.array(order,dtype=int)
        if len(eligible)>=4:
            blocks=[]
            for array in (left,right):
                z=np.log10(np.maximum(np.linalg.norm(array[:,eligible],axis=2).T,1e-30))
                z=(z-z.mean(1,keepdims=True))/np.maximum(z.std(1,keepdims=True),1e-8)
                blocks.append(z)
            features=np.concatenate(blocks,axis=1)
            k=min(4,len(eligible)-1)
            fit=KMeans(n_clusters=k,n_init=20,random_state=142).fit(features)
            candidates=[]
            for group in range(k):
                members=eligible[fit.labels_==group]
                if len(members)>=2:candidates.append((float(np.mean(score[members])),members))
            if candidates:
                _,members=max(candidates,key=lambda x:x[0])
                selected=sorted(members,key=lambda c:(-score[c],int(c)))[:2]
                plans.append(dict(stage=stage,method=name,kind='cluster',channels=[int(c) for c in selected]))
            cluster_info.append(dict(stage=stage,method=name,feature_space='standardized per-case log gradient norms, final D1 head, discovery only',
                                     channels=eligible.tolist(),labels=fit.labels_.tolist(),cluster_sizes=np.bincount(fit.labels_,minlength=k).tolist(),
                                     silhouette=float(silhouette_score(features,fit.labels_)) if len(np.unique(fit.labels_))>1 else None))
        generator=fixed_rng(142,stage,'rank_bootstrap')
        correlations,jaccards=[],[]
        for _ in range(100):
            a=left[generator.integers(0,len(left),len(left))]
            b=right[generator.integers(0,len(right),len(right))]
            boot,_,_,_,bo,_,_=chap_score(a,b,absolute,relative)
            r=rho(score,boot)
            if r is not None:correlations.append(r)
            if len(order)>=2 and len(bo)>=2:
                aa,bb=set(order[:2]),set(bo[:2]);jaccards.append(len(aa&bb)/len(aa|bb))
        stability.append(dict(stage=stage,method=name,score_bootstrap_spearman=describe(correlations),top2_bootstrap_jaccard=describe(jaccards)))
    primary=['A','B1_w2','B2_w2','B3_w2']
    for a in primary:
        for b in primary:
            if a>=b:continue
            x,xo=all_results[a];y,yo=all_results[b]
            union=set(xo[:2])|set(yo[:2])
            agreements.append(dict(stage=stage,method_a=a,method_b=b,spearman=rho(x,y),
                                    top2_jaccard=len(set(xo[:2])&set(yo[:2]))/len(union) if union else None))
    for method in ('B1','B2','B3'):
        main,order=all_results[method+'_w2']
        for width in (1,3):
            other,order2=all_results[f'{method}_w{width}']
            union=set(order[:2])|set(order2[:2])
            width_rows.append(dict(stage=stage,method=method,width=width,reference_width=2,spearman=rho(main,other),
                                   top2_jaccard=len(set(order[:2])&set(order2[:2]))/len(union) if union else None))
    common=set.intersection(*(set(all_results[name][1]) for name in primary))
    generator=fixed_rng(142,stage,'random_groups')
    seen=set();maximum=min(8,len(common)*(len(common)-1)//2)
    while len(seen)<maximum:
        pair=tuple(sorted(int(c) for c in generator.choice(sorted(common),2,replace=False)))
        if pair in seen:continue
        seen.add(pair);plans.append(dict(stage=stage,method=f'random_{len(seen)-1}',kind='random',channels=list(pair)))
    return rank_rows,plans,cluster_info,agreements,stability,width_rows


def analyze(run):
    run=require_stage1b_run(run)
    started=time.monotonic()
    assert json.loads((run/'extraction_results.json').read_text())['status']=='PASS'
    design=json.loads((run/'design.json').read_text())
    observations=json.loads((A_RUN/'observations.json').read_text())
    mapping=json.loads((run/'parameter_mapping.json').read_text())
    factors={d:np.array([o['normalized_scale_factor'] for o in observations[d]]) for d in ('L','U')}
    coverage=json.loads((run/'loss_coverage.json').read_text())
    grouped=defaultdict(list)
    for row in coverage:grouped[(row['domain'],row['objective'],row['scale'])].append(row)
    coverage_rows=[]
    for (d,obj,scale),rows in grouped.items():
        counts=Counter(r['status'] for r in rows)
        row=dict(domain=d,objective=obj,scale=scale,valid=sum(r['status']=='valid' for r in rows),total=len(rows),reasons=dict(counts))
        for key in ('inner_effective_cells','outer_effective_cells','effective_cells','native_inner','native_outer',
                    'native_band_fraction','candidate_levelset_voxels','scale_teacher_gradient_max','inner_occupancy_mean','outer_occupancy_mean'):
            values=[r['stats'][key] for r in rows if r['stats'].get(key) is not None]
            row[key+'_median']=finite_median(values) if values else None
        coverage_rows.append(row)
    write_csv(run/'coverage_summary.csv',coverage_rows)
    summary,paired_rows,channel_rows,cross_rows,multiscale=[],[],[],[],[]
    rank_rows,plans,clusters,agreements,rank_stability,width_sensitivity=[],[],[],[],[],[]
    for stage,spec in mapping.items():
        accessible=[SCALES.index(s) for s in spec['reachable_scales']]
        for domain,objectives in (('L',['A']+design['L_objectives']),('U',['A']+design['U_objectives'])):
            for obj in objectives:
                raw=sources(run,domain,obj,stage)
                _,cross,rank,cancel=observation_patterns(raw,accessible,design['gradient_absolute_floor'],design['gradient_relative_floor'])
                for a in range(4):
                    for b in range(a+1,4):
                        medians=[finite_median(cross[:,c,a,b]) for c in range(spec['channels'])]
                        cross_rows.append(dict(stage=stage,domain=domain,objective=obj,scale_a=SCALES[a],scale_b=SCALES[b],
                                               median_cosine=finite_median([x for x in medians if x is not None]),
                                               valid_channels=sum(x is not None for x in medians)))
                multiscale.append(dict(stage=stage,domain=domain,objective=obj,effective_rank_median=finite_median(rank),
                                       cancellation_ratio_median=finite_median(cancel)))
        for si,scale in enumerate(SCALES):
            if si not in accessible:
                summary.extend(dict(stage=stage,scale=scale,method=m,graph_reachable=False) for m in relations())
                continue
            cache={}
            @lru_cache(maxsize=4)
            def normalized(domain,obj):
                return case_means(sources(run,domain,obj,stage)[:,si:si+1],2,factors[domain])[:,0]
            def get(domain,obj):
                key=(domain,obj)
                if key not in cache:
                    raw=sources(run,domain,obj,stage)
                    mean=case_means(raw[:,si:si+1],2)[:,0]
                    within,_,_,_=signed_cosine(raw[::2,si],raw[1::2,si],design['gradient_absolute_floor'],design['gradient_relative_floor'])
                    bet,_,_,_=pairwise_cosine(mean,mean,design['gradient_absolute_floor'],design['gradient_relative_floor'])
                    a,b=np.triu_indices(len(mean),1)
                    norms,_,_=gradient_validity(mean,design['gradient_absolute_floor'],design['gradient_relative_floor'])
                    cache[key]=(mean,within.T,bet[:,a,b],norms)
                return cache[key]
            for method,(lobj,uobj) in relations().items():
                left,lwithin,lbetween,ln=get('L',lobj)
                right,uwithin,ubetween,un=get('U',uobj)
                cosines,valid,_,_=pairwise_cosine(left,right,design['gradient_absolute_floor'],design['gradient_relative_floor'])
                normcos,_,_,_=pairwise_cosine(normalized('L',lobj),normalized('U',uobj),design['gradient_absolute_floor'],design['gradient_relative_floor'])
                reliable=valid.mean((1,2))>=.75
                means=masked_mean(cosines,axis=(1,2));normalized_means=masked_mean(normcos,axis=(1,2))
                for c in range(spec['channels']):
                    channel_rows.append(dict(stage=stage,scale=scale,method=method,channel=c,
                                             status='valid' if reliable[c] else 'insufficient_valid_case_pairs',
                                             valid_pair_fraction=float(valid[c].mean()),L_U_cosine=means[c],
                                             A_denominator_cosine=normalized_means[c],raw_L_norm_median=finite_median(ln[:,c]),
                                             raw_U_norm_median=finite_median(un[:,c]),within_L_median=finite_median(lwithin[c]),
                                             within_U_median=finite_median(uwithin[c]),between_L_median=finite_median(lbetween[c]),
                                             between_U_median=finite_median(ubetween[c])))
                summary.append(dict(stage=stage,scale=scale,method=method,graph_reachable=True,channels=spec['channels'],
                                    reliable_channels=int(reliable.sum()),valid_pair_fraction=float(valid.mean()),
                                    L_U_cosine_median=finite_median(means[reliable]),
                                    A_denominator_cosine_median=finite_median(normalized_means[reliable]),
                                    within_L_median=finite_median(lwithin[reliable]),within_U_median=finite_median(uwithin[reliable]),
                                    between_L_median=finite_median(lbetween[reliable]),between_U_median=finite_median(ubetween[reliable])))
            for domain,x,y,label in [('L','band_w1','A','boundary_vs_seg_w1'),('L','band_w2','A','boundary_vs_seg_w2'),
                                      ('L','band_w3','A','boundary_vs_seg_w3'),('U','edge_w1','A','edge_vs_MT_w1'),
                                      ('U','edge_w2','A','edge_vs_MT_w2'),('U','edge_w3','A','edge_vs_MT_w3'),
                                      ('U','edge_w2','shuffle_w2','edge_vs_shuffled')]+[
                                      ('L',f'band_w{w}',f'{control}_w{w}',f'boundary_vs_{control}_w{w}') for w in (1,2,3) for control in ('random','core')]:
                a,_,_,_=get(domain,x);b,_,_,_=get(domain,y)
                values,valid,_,_=signed_cosine(a,b,design['gradient_absolute_floor'],design['gradient_relative_floor'])
                paired_rows.append(dict(stage=stage,scale=scale,domain=domain,comparison=label,
                                        valid_fraction=float(valid.mean()),paired_cosine_median=finite_median(values)))
            cache.clear();normalized.cache_clear()
        if stage in DETAIL_STAGES:
            a,b,c,d,e,f=channel_plans(run,stage,design,observations)
            rank_rows.extend(a);plans.extend(b);clusters.extend(c);agreements.extend(d);rank_stability.extend(e);width_sensitivity.extend(f)
        print(f'ANALYZED {stage}; {time.monotonic()-started:.1f}s',flush=True)
    write_csv(run/'method_scale_summary.csv',summary)
    write_csv(run/'expanded_channel_relations.csv',channel_rows)
    write_csv(run/'paired_target_comparisons.csv',paired_rows)
    write_csv(run/'cross_scale_summary.csv',cross_rows)
    write_csv(run/'multiscale_summary.csv',multiscale)
    write_csv(run/'rankings.csv',rank_rows)
    write_csv(run/'ranking_agreement.csv',agreements)
    write_csv(run/'ranking_width_sensitivity.csv',width_sensitivity)
    write_json(run/'rank_stability.json',rank_stability)
    write_json(run/'cluster_exploration.json',clusters)
    write_json(run/'channel_plan.json',dict(groups=plans,K=2,stages=DETAIL_STAGES,discovery=design['data_split'],
                                           ranking='final-head raw parameter gradients; CHAP signed cosine x relative U norm',
                                           no_test_selection=True,primary_width=2))
    write_json(run/'gradient_summary.json',dict(methods=summary,paired=paired_rows,coverage=coverage_rows))
    result=dict(status='PASS',method_stage_scale_cells=len(summary),channel_relation_rows=len(channel_rows),
                intervention_groups=len(plans),elapsed_seconds=time.monotonic()-started)
    write_json(run/'analysis_results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):print(json.dumps(analyze(args.run.resolve()),indent=2))
