"""Energy-matched final-prediction probes; frozen model, labeled probe split only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from common import (require_stage1b_run,ROOT,A_RUN,load_models,load_case_and_crops,state_snapshot,unchanged,write_json,write_csv)
from losses import distances,labeled_regions,pool_native
from interventions import attenuation,prediction_metrics,DETAIL_STAGES


def energy_budget(features,masks,fraction=.1):
    """Fix a stage/crop budget from activations, before looking at predictions."""
    energies={name:(features[0]*mask[0]).flatten(1).norm(dim=1) for name,mask in masks.items()}
    medians=[float(e[e>1e-12].median()) for e in energies.values() if bool((e>1e-12).any())]
    return fraction*min(medians) if len(medians)==len(masks) else 0.,energies


def matched_alpha(norm,budget,cap=.5):
    if not np.isfinite(norm) or norm<=1e-12:return None,'zero_or_nonfinite_activation'
    alpha=budget/norm
    if budget<=0:return None,'invalid_common_budget'
    if alpha>cap:return None,'unmatchable_budget_exceeds_alpha_cap'
    return alpha,'valid'


def tensor_metrics(output,y,regions):
    return {key:value.detach().cpu().numpy() for key,value in prediction_metrics(output,y,*regions).items()}


def run(args,output):
    begin=time.monotonic()
    run=require_stage1b_run(args.run)
    proof=json.loads((run/'analysis_results.json').read_text());assert proof['status']=='PASS'
    plan=json.loads((run/'channel_plan.json').read_text())
    design=json.loads((run/'design.json').read_text())
    observations=json.loads((A_RUN/'observations.json').read_text())
    student,teacher,a_design=load_models(args.device)
    snapshots=[state_snapshot(model) for model in (student,teacher)]
    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(args.device).clone()
    torch.cuda.reset_peak_memory_stats(args.device)
    access,rows,pair_rows,calibrations,baseline_checks=[],[],[],[],[]
    stages=DETAIL_STAGES[:1] if args.smoke else DETAIL_STAGES
    cases=design['data_split']['probe_L'][:1] if args.smoke else design['data_split']['probe_L']
    group_map=defaultdict(list)
    for group in plan['groups']:group_map[group['stage']].append(group)
    for ci,case in enumerate(cases):
        obs=[o for o in observations['L'] if o['case']==case]
        if args.smoke:obs=obs[:1]
        _,label,crops=load_case_and_crops('L',case,a_design,obs,access)
        distance_maps=distances(label)
        for obs,x,y,meta in crops:
            x,y=x.to(args.device),y.to(args.device)
            with torch.no_grad():
                native,info=labeled_regions(label,distance_maps,meta,2,design['seed'],case,obs['repeat'])
                assert info['B']['status']=='valid'
                base,features=student(x,deepsup=True,return_features=True)
                base_metrics=tensor_metrics(base[-1],y,native['B'])
                # Fixed batch size4 for every causal contrast: row3 is always control.
                xb=x.expand(4,-1,-1,-1,-1).contiguous()
                for stage in stages:
                    group='encoder' if stage[0]=='E' else 'decoder'
                    feature=features[group][stage]
                    masks={name:pool_native(a|b,feature.shape[2:],args.device)[0]
                           for name,regions in native.items() if regions is not None and info[name]['status']=='valid' for a,b in [regions]}
                    delta,energies=energy_budget(feature,masks)
                    calibrations.append(dict(case=case,repeat=obs['repeat'],stage=stage,delta_single=delta,delta_pair=np.sqrt(2)*delta,
                                             channels=feature.shape[1],regions=list(masks),label_control_info=info))
                    singleton_channels=list(range(feature.shape[1]))
                    if args.smoke:singleton_channels=singleton_channels[:3]
                    for region,mask in masks.items():
                        energies_np=energies[region].cpu().numpy()
                        for offset in range(0,len(singleton_channels),3):
                            chosen=singleton_channels[offset:offset+3]
                            sets,alphas,statuses=[],[],[]
                            for channel in chosen:
                                alpha,status=matched_alpha(float(energies_np[channel]),delta)
                                sets.append([channel]);alphas.append(alpha if alpha is not None else 0.);statuses.append(status)
                            while len(sets)<3:sets.append([]);alphas.append(0.)
                            sets.append([]);alphas.append(0.)
                            with attenuation(student,stage,sets,mask,alphas) as hook:logits=student(xb)
                            assert hook['calls']==1
                            metrics=tensor_metrics(logits,y,native['B'])
                            baseline_checks.append(float((logits[3]-base[-1][0]).abs().max()))
                            changed=(logits.argmax(1)!=logits[3:4].argmax(1)).flatten(1).float().mean(1).cpu().numpy()
                            for j,channel in enumerate(chosen):
                                record=dict(case=case,repeat=obs['repeat'],stage=stage,region=region,channel=channel,
                                            status=statuses[j],alpha=alphas[j],target_l2=delta,actual_l2=hook['actual_l2'][j])
                                if statuses[j]=='valid':
                                    assert abs(hook['actual_l2'][j]-delta)<=max(1e-5,delta*2e-5)
                                    record['changed_prediction_fraction']=changed[j]
                                    for key,value in metrics.items():record[key+'_delta']=value[j]-value[3]
                                rows.append(record)
                            del logits
                        groups=group_map[stage]
                        if args.smoke:groups=groups[:1]+[g for g in groups if g['kind']=='random'][:1]
                        grouped=defaultdict(list)
                        for g in groups:grouped[tuple(sorted(g['channels']))].append(g)
                        for channels,aliases in grouped.items():
                            assert len(channels)==2
                            pair_norm=float(np.linalg.norm(energies_np[list(channels)]))
                            budget=np.sqrt(2)*delta
                            alpha,status=matched_alpha(pair_norm,budget)
                            stats=dict(case=case,repeat=obs['repeat'],stage=stage,region=region,channels=list(channels),
                                       status=status,alpha=alpha,target_l2=budget)
                            if status=='valid':
                                sets=[list(channels),[channels[0]],[channels[1]],[]]
                                with attenuation(student,stage,sets,mask,[alpha,alpha,alpha,0.]) as hook:logits=student(xb)
                                assert hook['calls']==1
                                stats['actual_l2']=hook['actual_l2'][0]
                                stats['single_component_l2']=hook['actual_l2'][1:3]
                                assert abs(hook['actual_l2'][0]-budget)<=max(1e-5,budget*2e-5)
                                assert abs(sum(z*z for z in hook['actual_l2'][1:3])-budget**2)<=max(1e-5,budget**2*5e-5)
                                metrics=tensor_metrics(logits,y,native['B'])
                                stats['logit_interaction_rms']=float((logits[0]-logits[1]-logits[2]+logits[3]).square().mean().sqrt())
                                stats['logit_joint_delta_rms']=float((logits[0]-logits[3]).square().mean().sqrt())
                                stats['changed_prediction_fraction']=float((logits[0].argmax(0)!=logits[3].argmax(0)).float().mean())
                                for key,value in metrics.items():
                                    stats[key+'_delta']=value[0]-value[3]
                                    stats[key+'_interaction']=value[0]-value[1]-value[2]+value[3]
                                baseline_checks.append(float((logits[3]-base[-1][0]).abs().max()))
                                del logits
                            for alias in aliases:pair_rows.append(dict(stats,method=alias['method'],kind=alias['kind']))
                    del feature,masks,energies
                del base,features,xb,x,y
            write_csv(output/'singletons.csv',rows)
            write_csv(output/'groups.csv',pair_rows)
            write_json(output/'calibration.json',calibrations)
            write_json(output/'hdf5_access_log.json',access)
            print(f'ABLATION {case} crop{obs["repeat"]}; singles{len(rows)} pairs{len(pair_rows)} elapsed{time.monotonic()-begin:.1f}s',flush=True)
    checks=dict(student_unchanged=unchanged(student,snapshots[0]),teacher_unchanged=unchanged(teacher,snapshots[1]),
                no_parameter_grads=all(p.grad is None for m in (student,teacher) for p in m.parameters()),
                cpu_rng_unchanged=torch.equal(rng,torch.get_rng_state()),cuda_rng_unchanged=torch.equal(cuda_rng,torch.cuda.get_rng_state(args.device)),
                eval=all(not m.training for m in (student,teacher)),
                only_labeled_probe_cases=set(cases)<=set(design['data_split']['probe_L']),
                no_discovery_overlap=not(set(cases)&set(design['data_split']['discovery_L'])),
                no_unlabeled_access=all(r['role']!='unlabeled' for r in access),
                hooks_removed=all(not m._forward_hooks and not m._forward_pre_hooks for m in student.modules()),
                batch_baseline_stable=max(baseline_checks,default=0)<1e-4)
    assert all(checks.values()),checks
    result=dict(status='PASS',smoke=args.smoke,checks=checks,labeled_cases=len(cases),crops=len(calibrations)//len(stages),
                singleton_rows=len(rows),group_rows=len(pair_rows),batch_vs_single_max_logit_error=max(baseline_checks,default=0),
                elapsed_seconds=time.monotonic()-begin,peak_mib=torch.cuda.max_memory_allocated(args.device)/2**20,
                target='Final D1 logits/predictions; no auxiliary-head outcomes',optimizer_steps=0,checkpoint_writes=0)
    write_json(output/'ablation_results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-proof',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    assert output.is_relative_to(ROOT/'runs')
    if not args.smoke:
        assert args.smoke_proof and json.loads(args.smoke_proof.read_text())['status']=='PASS'
    output.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(run(args,output),indent=2))
    except BaseException:
        write_json(output/'ablation_results.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
