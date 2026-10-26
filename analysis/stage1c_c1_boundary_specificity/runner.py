"""C1 frozen L-only diagnostics, discovery plan frozen before probe access."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
from scipy import ndimage
import torch
from bridge import (ROOT,C0,b,b_losses,STAGES,DOSES,PRIMARY,write_json,write_csv,fixed_rng,split,output_path,
                    Capture,attenuate,adjacent_pairs,permuted_partners,calibrate_epsilon,direction,dose_alpha,node_statistics,prediction_metrics,surface6)
from control_pairs import far_controls,pair_statistics,distributions,compare_endpoints
from targets import Targets
from measures import cosine_rows


def read_csv(path):
    import csv
    return list(csv.DictReader(Path(path).open()))

def compatibility(outputs,y,regions,info,obs,mapping,features,base,targets):
    old_cov=json.loads((b.ROOT/'runs/la10_paired/loss_coverage.json').read_text())
    a,_,_=b.independent_losses(outputs,labels=y)
    np.testing.assert_allclose([float(v.detach()) for v in a],obs['losses'],atol=1e-7,rtol=1e-5)
    terms=b_losses.labeled_terms(outputs,y[0].cpu().numpy(),regions,info)
    reference=[next(r for r in old_cov if r['domain']=='L' and r['index']==obs['index'] and r['objective']=='band_w2' and r['scale']==s) for s in b.SCALES]
    np.testing.assert_allclose([float(t.loss.detach()) for t in terms],[r['raw_loss'] for r in reference],atol=1e-7,rtol=1e-5)
    np.testing.assert_allclose(float(base['boundary'][0].detach()),float(terms[-1].loss.detach()),atol=1e-7,rtol=1e-5)
    rows=[]
    for name,loss in [('A',a[-1]),('B',terms[-1].loss)]:
        gradients=torch.autograd.grad(loss,[mapping[s].parameter for s in STAGES],retain_graph=True)
        for s,g in zip(STAGES,gradients):
            value=mapping[s].channel_vectors(g).detach().cpu().numpy()
            path=b.A_RUN/'raw'/f'L_{s}.npy' if name=='A' else b.ROOT/'runs/la10_paired/raw'/f'L_band_w2_{s}.npy'
            original=np.load(path,mmap_mode='r')[obs['index'],3]
            error=np.linalg.norm((value-original).astype(float));norm=np.linalg.norm(original.astype(float));rel=error/max(norm,1e-30)
            assert error<=1e-9 or rel<=1e-4
            rows.append(dict(reference=name,stage=s,absolute_error=error,relative_error=rel))
    # Only the two C0 L cases have archived C0 reference tensors. Compare all
    # five feature nodes and the exact native pair coordinates without edits.
    c0_run=C0/'runs/c0_la10_checked'
    path=c0_run/'raw'/f'L_{obs["index"]}_pairs.npz'
    if path.exists():
        p=np.load(path);np.testing.assert_array_equal(targets.pairs['true'].plus,p['plus']);np.testing.assert_array_equal(targets.pairs['true'].minus,p['minus'])
        gs=torch.autograd.grad(base['transition'][0],[features[s] for s in STAGES],retain_graph=True)
        for s,g in zip(STAGES,gs):
            ref=np.load(c0_run/'raw/features'/f'L_{obs["index"]}_{s}.npz')['gradient']
            v=g.detach().cpu().numpy();err=float(np.linalg.norm((v-ref).astype(float)));rel=err/max(float(np.linalg.norm(ref.astype(float))),1e-30)
            assert err<=1e-9 or rel<=1e-4
            rows.append(dict(reference='C0_feature_transition',stage=s,absolute_error=err,relative_error=rel))
    return rows


def context(student,x,y,label,dist,meta,obs,device):
    with Capture(student) as capture:outputs,interface=student(x,deepsup=True,return_features=True)
    checks=capture.verify(interface)
    bands,infos=b_losses.labeled_regions(label,dist,meta,2,142,obs['case'],obs['repeat'])
    assert infos['B']['status']=='valid' and infos['I']['status']=='valid',infos
    yc=y[0].cpu().numpy();valid=b.crop_native(np.ones(label.shape,bool),meta)
    p=adjacent_pairs(yc,valid,seed_parts=(obs['case'],obs['repeat'],'L'));assert p.valid
    z=(outputs[-1][0,1]-outputs[-1][0,0]).detach().double().cpu().numpy()
    di,do=[b.crop_native(a,meta) for a in dist]
    far,matched,common,farinfo=far_controls(p,z,yc,di,do,valid,(obs['case'],obs['repeat']))
    assert far.valid
    pairs=dict(true=p,permuted=permuted_partners(p,(obs['case'],obs['repeat'],'C1',0)),far=far,
               permuted_1=permuted_partners(p,(obs['case'],obs['repeat'],'C1',1)),permuted_2=permuted_partners(p,(obs['case'],obs['repeat'],'C1',2)),
               far_matched=matched,true_common=common)
    for q in pairs.values():
        if len(q.plus):assert yc[tuple(q.plus.T)].all() and not yc[tuple(q.minus.T)].any()
    targets=Targets(pairs,y,bands['B']);base=targets(outputs[-1])
    masks={r:{s:b_losses.pool_native(bands[r][0]|bands[r][1],capture.nodes[s].shape[2:],device)[0].detach() for s in STAGES} for r in ('B','I')}
    energies={r:{s:(capture.nodes[s].detach().double()*masks[r][s].double()).square().sum((0,2,3,4)).sqrt().cpu().numpy() for s in STAGES} for r in masks}
    q={s:min(float(np.median(e[e>1e-12])) if (e>1e-12).any() else 0. for e in (energies[r][s] for r in masks)) for s in STAGES}
    return dict(outputs=outputs,features=capture.nodes,checks=checks,bands=bands,infos=infos,targets=targets,base=base,masks=masks,energies=energies,q=q,z=z,farinfo=farinfo)


def gradients(ctx,obs,mapping,out,tables):
    key={k:obs[k] for k in ('case','index','repeat')};features=ctx['features'];nodes=[features[s] for s in STAGES]
    bank={s:{} for s in STAGES};parameter_bank={s:{} for s in STAGES}
    for name,loss in ctx['base'].items():
        grads=torch.autograd.grad(loss[0],nodes+[mapping[s].parameter for s in STAGES],retain_graph=True,allow_unused=True)
        for j,s in enumerate(STAGES):
            g,gp=grads[j],grads[j+5]
            assert g is not None and gp is not None,('unexpected unreachable final path',s,name)
            assert g.shape==features[s].shape and gp.shape==mapping[s].parameter.shape
            v=g.detach().cpu().numpy()[0];pv=mapping[s].channel_vectors(gp).detach().cpu().numpy()
            bank[s][name]=v;parameter_bank[s][name]=pv
            if name in PRIMARY:
                np.save(out/'raw/parameters'/f'L_{obs["index"]}_{name}_{s}.npy',pv)
            for r in ('B','I'):
                stat=node_statistics(g,features[s],ctx['masks'][r][s])
                threshold=max(1e-12,1e-6*float(stat['gradient_norm'].max(initial=0)))
                for c in range(len(pv)):
                    energy=stat['activation_energy'][c]
                    tables['contributions'].append(dict(key,objective=name,stage=s,channel=c,support=r,coordinate_kind='feature_node',
                        gradient_norm=stat['gradient_norm'][c],masked_gradient_norm=stat['masked_gradient_norm'][c],activation_energy=energy,
                        inner=stat['signed_inner'][c],abs_gxa=stat['absolute_product_sum'][c],q=ctx['q'][s],
                        unit_response=-stat['signed_inner'][c]/energy if energy>1e-12 else None,
                        gradient_status='valid' if stat['gradient_norm'][c]>threshold else 'exact_zero' if stat['gradient_norm'][c]==0 else 'below_threshold',
                        max_dose_status=dose_alpha(energy,.10*ctx['q'][s])[1]))
            pn=np.linalg.norm(pv.astype(float),axis=1);pt=max(1e-12,1e-6*pn.max(initial=0))
            for c,n in enumerate(pn):
                tables['parameter_norms'].append(dict(key,objective=name,stage=s,channel=c,coordinate_kind='convolution_parameter',parameter=mapping[s].name,
                    parameter_shape=list(gp.shape),output_axis=mapping[s].output_axis,gradient_norm=n,status='valid' if n>pt else 'exact_zero' if n==0 else 'below_threshold'))
        del grads
    for s in STAGES:
        for a,c in combinations(bank[s],2):
            for kind,source in [('feature_same_crop',bank),('parameter_shared_coordinates',parameter_bank)]:
                cos,an,bn,status=cosine_rows(source[s][a],source[s][c])
                for channel,v in enumerate(cos):tables['gradient_relations'].append(dict(key,stage=s,left=a,right=c,channel=channel,coordinate_kind=kind,
                    signed_cosine=v,left_norm=an[channel],right_norm=bn[channel],status=status[channel]))
        np.savez_compressed(out/'raw/features'/f'L_{obs["index"]}_{s}.npz',activation=features[s].detach().cpu().numpy(),
            mask_B=ctx['masks']['B'][s].cpu().numpy(),mask_I=ctx['masks']['I'][s].cpu().numpy(),**{f'g_{n}':bank[s][n] for n in PRIMARY})
    return bank


def geometry(ctx,y,obs,tables,out):
    key={k:obs[k] for k in ('case','index','repeat')};z=ctx['z'];pairs=ctx['targets'].pairs
    predicted=ctx['outputs'][-1].detach().argmax(1)[:,None].bool()
    ps=surface6(predicted)[0,0].cpu().numpy();pred_dist=ndimage.distance_transform_edt(~ps) if ps.any() else None
    stats={name:pair_statistics(p,z,pred_dist if name.startswith('true') or name.startswith('permuted') else None) for name,p in pairs.items()}
    for name in ('permuted','permuted_1','permuted_2'):
        q=pairs[name];stats[name]['endpoint_distribution_check']=compare_endpoints(z,pairs['true'],q)
        stats[name]['unchanged_partners']=q.stats['unchanged_partners'];stats[name]['endpoint_multisets_exact']=True
    metric={k:float(v[0]) for k,v in prediction_metrics(ctx['outputs'][-1].detach(),y,*ctx['bands']['B']).items()}
    record=dict(key,pairs=stats,far_matching=ctx['farinfo'],baseline_metrics=metric,
                region_difficulty={r:distributions(z,*ctx['bands'][r]) for r in ('B','I')},coarse_masks={})
    for s in STAGES:
        a,c=ctx['masks']['B'][s],ctx['masks']['I'][s]
        record['coarse_masks'][s]=dict(B=b_losses.weight_stats(a),I=b_losses.weight_stats(c),
            weighted_overlap=float((a*c).sum()),support_overlap_cells=int(((a>0)&(c>0)).sum()),q=ctx['q'][s])
    tables['geometry'].append(record)
    np.savez_compressed(out/'raw'/f'L_{obs["index"]}_pairs.npz',**{name+'_'+side:getattr(p,side) for name,p in pairs.items() for side in ('plus','minus')})
    np.savez_compressed(out/'raw'/f'L_{obs["index"]}_prediction.npz',logits=ctx['outputs'][-1].detach().cpu().numpy(),
                        labels=y.cpu().numpy().astype(np.uint8),**{'mask_'+r:(ctx['bands'][r][0]|ctx['bands'][r][1]) for r in ('B','I')})
    for name,v in ctx['base'].items():tables['losses'].append(dict(key,objective=name,raw_loss=float(v[0].detach()),precision='FP64',LossNorm_applied=False))


def candidate_plan(tables,discovery,out):
    grouped=defaultdict(list)
    for r in tables['contributions']:
        if r['objective'] in PRIMARY:grouped[(r['stage'],r['channel'],r['support'],r['objective'])].append(r)
    plan=dict(discovery_cases=discovery,probe_cases_opened=False,primary_dose=.03,primary_margin=1.,stages={},scores={},eligibility={})
    for s in STAGES:
        channels=sorted({c for stage,c,_,_ in grouped if stage==s});eligible=[];score={r:{n:{} for n in PRIMARY} for r in ('B','I')}
        for c in channels:
            valid_indices=[]
            for index in sorted({v['index'] for k,rs in grouped.items() if k[0]==s for v in rs}):
                rows=[next(v for v in grouped[(s,c,r,n)] if v['index']==index) for r in ('B','I') for n in PRIMARY]
                if all(v['gradient_status']=='valid' and v['max_dose_status']=='valid' and v['unit_response'] is not None for v in rows):valid_indices.append(index)
            plan['eligibility'][f'{s}_{c}']=dict(valid_crops=len(valid_indices),total=8)
            if len(valid_indices)<6:continue
            eligible.append(c)
            for r in ('B','I'):
                for n in PRIMARY:
                    percase=[]
                    for case in discovery:
                        v=[v['unit_response'] for v in grouped[(s,c,r,n)] if v['case']==case and v['index'] in valid_indices]
                        if v:percase.append(float(np.mean(v)))
                    score[r][n][str(c)]=float(np.mean(percase))
        chosen=set();reasons=defaultdict(list)
        for n in PRIMARY:
            ordered=sorted(eligible,key=lambda c:(score['B'][n][str(c)],c))
            for c in ordered[:2]+ordered[-2:]:chosen.add(c);reasons[c].append(n+'_signed_extremes')
        if eligible:
            for c in fixed_rng(142,'C1_candidates',s).choice(eligible,min(4,len(eligible)),replace=False):chosen.add(int(c));reasons[int(c)].append('fixed_random')
            for c in (0,len(channels)//2,len(channels)-1):
                if c in eligible:chosen.add(c);reasons[c].append('C0_anchor')
        assert len(chosen)>=3,('insufficient discovery candidates',s)
        plan['stages'][s]=dict(eligible=eligible,channels=sorted(chosen),reasons=reasons);plan['scores'][s]=score
    write_json(out/'FROZEN_PLAN.json',plan)
    write_json(out/'FROZEN_PLAN_SHA256.json',dict(sha256=b.sha256(out/'FROZEN_PLAN.json'),frozen_before_probe=True))
    return plan


def intervene(student,x,y,ctx,obs,channels,out,tables):
    key={k:obs[k] for k in ('case','index','repeat')};base={n:float(v[0].detach()) for n,v in ctx['base'].items()}
    lookup={(r['stage'],r['support'],r['objective'],r['channel']):r for r in tables['contributions'] if r['index']==obs['index']}
    xb=x.expand(4,-1,-1,-1,-1).contiguous();nulls=defaultdict(list)
    with torch.no_grad():
        for s in STAGES:
            with attenuate(student,s,[None]*4,ctx['masks']['B'][s],[0.]*4,ctx['features'][s]) as trace:null_logits=student(xb)
            assert trace['calls']==1 and max(trace['actual_l2'])==0
            for n,v in ctx['targets'](null_logits).items():nulls[n].extend(v.cpu().tolist())
        eps={}
        for n in base:
            eps[n],record=calibrate_epsilon(base[n],nulls[n]);tables['noise'].append(dict(key,objective=n,epsilon=eps[n],**record))
        # Null thresholds fixed before any nonzero operation on this crop.
        for s in STAGES:
            for support in ('B','I'):
                mask=ctx['masks'][support][s]
                for dose in DOSES:
                    budget=dose*ctx['q'][s]
                    for start in range(0,len(channels[s]),3):
                        chosen=channels[s][start:start+3];padding=3-len(chosen)
                        statuses=[dose_alpha(ctx['energies'][support][s][c],budget) for c in chosen]
                        alphas=[a if a is not None else 0. for a,status in statuses]
                        with attenuate(student,s,[*chosen,*([None]*padding),None],mask,[*alphas,*([0.]*padding),0.],ctx['features'][s]) as trace:logits=student(xb)
                        assert trace['calls']==1
                        measured=ctx['targets'](logits);metrics=prediction_metrics(logits,y,*ctx['bands']['B'])
                        changed=(logits.argmax(1)!=logits[3:4].argmax(1)).flatten(1).float().mean(1).cpu().numpy()
                        for row,c in enumerate(chosen):
                            alpha,status=statuses[row]
                            if status=='valid' and not np.isclose(trace['actual_l2'][row],budget,rtol=2e-5,atol=1e-5):status='energy_matching_failed'
                            common=dict(key,stage=s,channel=c,support=support,dose=dose,status=status,alpha=alpha,target_l2=budget,
                                actual_l2=trace['actual_l2'][row],intended_l2=trace['intended_l2'][row],roundoff_l2=trace['subtraction_roundoff_l2'][row],node_max_error=trace['baseline_node_max_error'])
                            if status=='valid':
                                common.update(boundary_delta=float(measured['boundary'][row]-measured['boundary'][3]),
                                    region_delta=float(measured['region'][row]-measured['region'][3]),epsilon=eps['boundary'],changed_prediction_fraction=changed[row])
                                for n,v in metrics.items():common[n+'_delta']=float(v[row]-v[3])
                                common['boundary_direction']=direction(common['boundary_delta'],eps['boundary'])
                            tables['interventions'].append(common)
                            for n,v in measured.items():
                                record=dict(key,stage=s,channel=c,support=support,dose=dose,objective=n,status=status,epsilon=eps[n])
                                if status=='valid':
                                    r=lookup[(s,support,n,c)];pred=-alpha*r['inner'];actual=float(v[row]-v[3])
                                    record.update(delta_hat=pred,delta_actual=actual,predicted_direction=direction(pred,eps[n]),actual_direction=direction(actual,eps[n]),
                                        abs_gxa=alpha*r['abs_gxa'],unit_response=r['unit_response'],
                                        boundary_delta=common['boundary_delta'],boundary_epsilon=eps['boundary'],boundary_direction=common['boundary_direction'])
                                tables['local_attribution'].append(record)
                        del logits,measured,metrics


def save(tables,out):
    for name,rows in tables.items():
        if name in ('geometry','noise','node_identity','compatibility'):write_json(out/(name+'.json'),rows)
        else:write_csv(out/(name+'.csv'),rows)


def run(args,out):
    assert json.loads((ROOT/'TEST_RESULTS.json').read_text())['status']=='PASS'
    if args.phase!='smoke':
        assert json.loads((ROOT/'runs/smoke/execution_results.json').read_text())['status']=='PASS'
    started=time.monotonic();student,teacher,design=b.load_models(args.device)
    assert args.device.startswith('cuda')
    mapping=b.stage_parameters(student);snapshots=[b.state_snapshot(m) for m in (student,teacher)]
    rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(args.device).clone(),np.random.get_state(),random.getstate())
    torch.cuda.reset_peak_memory_stats(args.device)
    splits=split();cases=splits['discovery'][:2] if args.phase=='smoke' else splits[args.phase]
    plan=None;plan_hash=None
    if args.phase=='probe':
        assert args.discovery is not None
        assert json.loads((args.discovery/'execution_results.json').read_text())['status']=='PASS'
        plan_hash=b.sha256(args.discovery/'FROZEN_PLAN.json');assert plan_hash==json.loads((args.discovery/'FROZEN_PLAN_SHA256.json').read_text())['sha256']
        plan=json.loads((args.discovery/'FROZEN_PLAN.json').read_text());assert not plan['probe_cases_opened']
    tables=defaultdict(list);access=[];file_stats={};observations=json.loads((b.A_RUN/'observations.json').read_text())['L']
    for f in ('features','parameters'):(out/'raw'/f).mkdir(parents=True)
    write_json(out/'source_at_run.json',dict(phase=args.phase,base_commit=json.loads((ROOT/'PREFLIGHT.json').read_text())['base_commit'],
        sources={p.name:b.sha256(p) for p in ROOT.glob('*.py')},design_sha256=b.sha256(ROOT/'C1_DESIGN.md'),frozen_plan_sha256_before_probe=plan_hash,
        cases=cases,device=args.device,gpu=torch.cuda.get_device_name(args.device),torch=torch.__version__,numpy=np.__version__,TF32=False,mode='eval',checkpoint_sha256=design['checkpoint_sha256']))
    for case in cases:
        obslist=[o for o in observations if o['case']==case]
        _,label,crops=b.load_case_and_crops('L',case,design,obslist,access);dist=b_losses.distances(label)
        for r in access:
            p=Path(r['path']);file_stats[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns)
        for obs,x,y,meta in crops:
            x,y=x.to(args.device),y.to(args.device)
            ctx=context(student,x,y,label,dist,meta,obs,args.device)
            tables['node_identity'].extend([dict(case=case,index=obs['index'],repeat=obs['repeat'],**r) for r in ctx['checks']])
            tables['compatibility'].extend([dict(case=case,index=obs['index'],**r) for r in compatibility(ctx['outputs'],y,ctx['bands']['B'],ctx['infos']['B'],obs,mapping,ctx['features'],ctx['base'],ctx['targets'])])
            geometry(ctx,y,obs,tables,out);gradients(ctx,obs,mapping,out,tables)
            if args.phase!='discovery':
                channels={s:plan['stages'][s]['channels'] if plan else [0,ctx['features'][s].shape[1]//2,ctx['features'][s].shape[1]-1] for s in STAGES}
                intervene(student,x,y,ctx,obs,channels,out,tables)
            save(tables,out);write_json(out/'hdf5_access_log.json',access)
            print(f'C1 {args.phase} gradients {case} crop{obs["repeat"]}; {time.monotonic()-started:.1f}s',flush=True)
            del ctx,x,y
    if args.phase=='discovery':
        plan=candidate_plan(tables,cases,out)
        print('Discovery plan frozen; no probe data opened',flush=True)
        for case in cases:
            _,label,crops=b.load_case_and_crops('L',case,design,[o for o in observations if o['case']==case],access);dist=b_losses.distances(label)
            for obs,x,y,meta in crops:
                x,y=x.to(args.device),y.to(args.device);ctx=context(student,x,y,label,dist,meta,obs,args.device)
                intervene(student,x,y,ctx,obs,{s:plan['stages'][s]['channels'] for s in STAGES},out,tables)
                save(tables,out)
                print(f'C1 discovery interventions {case} crop{obs["repeat"]}; {time.monotonic()-started:.1f}s',flush=True)
                del ctx,x,y
    invariants=dict(student_state_unchanged=b.unchanged(student,snapshots[0]),teacher_state_unchanged=b.unchanged(teacher,snapshots[1]),
        teacher_frozen=all(not p.requires_grad for p in teacher.parameters()),no_grad_slots=all(p.grad is None for m in (student,teacher) for p in m.parameters()),
        both_eval=not student.training and not teacher.training,torch_rng_unchanged=torch.equal(rng[0],torch.get_rng_state()),cuda_rng_unchanged=torch.equal(rng[1],torch.cuda.get_rng_state(args.device)),
        numpy_rng_unchanged=all(np.array_equal(a,c) for a,c in zip(rng[2],np.random.get_state())),python_rng_unchanged=rng[3]==random.getstate(),
        no_leftover_hooks=all(not m._forward_hooks and not m._forward_pre_hooks for m in student.modules()),
        L_only_access=all(r['role']=='labeled' for r in access),data_stats_unchanged=all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in file_stats.items()),
        frozen_plan_unchanged=True if args.phase!='probe' else b.sha256(args.discovery/'FROZEN_PLAN.json')==plan_hash)
    assert all(invariants.values()),invariants
    write_json(out/'hdf5_access_log.json',access)
    result=dict(status='PASS',phase=args.phase,cases=len(cases),crops=2*len(cases),invariants=invariants,interventions=len(tables['interventions']),
        elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated(args.device)/2**20,optimizer_steps=0,checkpoint_writes=0,C2_C3_executed=False)
    write_json(out/'execution_results.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('smoke','discovery','probe'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--discovery',type=Path);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();out=output_path(args.output);out.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(run(args,out),indent=2))
    except BaseException:
        write_json(out/'execution_results.json',dict(status='FAIL',phase=args.phase,traceback=traceback.format_exc()));raise
