"""C0 only: fixed 2L+2U, final D1 objectives and matched single-channel probes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch

from support import (b,b_losses,ROOT,B_ROOT,STAGES,DOSES,selection,selected_channels,output_path,
                     write_json,write_csv,calibrate_epsilon,direction,dose_alpha,node_statistics)
from pairs import adjacent_pairs,teacher_pairs
from objectives_c0 import Objectives
from nodes import Capture,attenuate,NODE_SPECS
from interventions import prediction_metrics


def replay_reference(outputs,teacher_outputs,y,regions,info,weights,edge_info,obs,domain,mapping):
    old=json.loads((B_ROOT/'runs/la10_paired/loss_coverage.json').read_text())
    a,_,_=b.independent_losses(outputs,labels=y,teacher_outputs=teacher_outputs)
    np.testing.assert_allclose([float(v.detach()) for v in a],obs['losses'],atol=1e-7,rtol=1e-5)
    obj='band_w2' if domain=='L' else 'edge_w2'
    terms=(b_losses.labeled_terms(outputs,y[0].cpu().numpy(),regions,info) if domain=='L' else
           b_losses.unlabeled_terms(outputs,teacher_outputs,weights,edge_info))
    reference=[next(r for r in old if r['domain']==domain and r['index']==obs['index'] and r['objective']==obj and r['scale']==scale) for scale in b.SCALES]
    np.testing.assert_allclose([float(t.loss.detach()) for t in terms],[r['raw_loss'] for r in reference],atol=1e-7,rtol=1e-5)
    records=[]
    params=[mapping[s].parameter for s in STAGES]
    for name,loss in [('A',a[-1]),(obj,terms[-1].loss)]:
        grads=torch.autograd.grad(loss,params,retain_graph=True)
        for stage,g in zip(STAGES,grads):
            vectors=mapping[stage].channel_vectors(g).detach().cpu().numpy()
            path=(b.A_RUN/'raw'/f'{domain}_{stage}.npy') if name=='A' else (B_ROOT/'runs/la10_paired/raw'/f'{domain}_{obj}_{stage}.npy')
            original=np.load(path,mmap_mode='r')[obs['index'],3]
            error=np.linalg.norm((vectors-original).astype(float));norm=np.linalg.norm(original.astype(float));relative=error/max(norm,1e-30)
            assert error<=1e-9 or relative<=1e-4
            records.append(dict(domain=domain,index=obs['index'],reference=name,stage=stage,loss_head='D1',absolute_l2_error=error,relative_l2_error=relative))
    return records,float(terms[-1].loss.detach())


def run(args,out):
    started=time.monotonic()
    assert json.loads((ROOT/'TEST_RESULTS.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'REGRESSION_RESULTS.json').read_text())['status']=='PASS'
    student,teacher,design=b.load_models(args.device)
    if not args.device.startswith('cuda'):raise RuntimeError('The real-data C0 smoke requires GPU; CPU contracts run separately')
    write_json(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,device=args.device,
                                         gpu=torch.cuda.get_device_name(args.device),TF32=False,mode='eval',network_precision='FP32',loss_precision='FP64'))
    write_json(out/'source_at_run.json',dict(design_sha256=b.sha256(ROOT/'DESIGN.md'),
                                           sources={p.name:b.sha256(p) for p in sorted(ROOT.glob('*.py'))},
                                           tests_sha256=b.sha256(ROOT.parents[1]/'tests/test_stage1c_c0.py')))
    selected=selection();observations=json.loads((b.A_RUN/'observations.json').read_text())
    snapshots=[b.state_snapshot(m) for m in (student,teacher)]
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(args.device).clone()
    np_rng=np.random.get_state();python_rng=random.getstate()
    torch.cuda.reset_peak_memory_stats(args.device)
    mapping=b.stage_parameters(student)
    for folder in ('features','parameters'):(out/'raw'/folder).mkdir(parents=True)
    access,geometry,node_checks,replays,compatibility,noise,feature_rows,parameter_rows,delta_rows,metric_rows=[],[],[],[],[],[],[],[],[],[]
    file_stats={}
    for domain in ('L','U'):
        for case in selected[domain]:
            current=[o for o in observations[domain] if o['case']==case]
            image,label,crops=b.load_case_and_crops(domain,case,design,current,access)
            path=Path(access[-1]['path']);file_stats[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
            distances=b_losses.distances(label) if domain=='L' else None
            for obs,x,y,meta in crops:
                key=dict(domain=domain,case=case,index=obs['index'],repeat=obs['repeat'])
                x=x.to(args.device);y=y.to(args.device) if y is not None else None
                with Capture(student) as capture:outputs,features=student(x,deepsup=True,return_features=True)
                node_checks.extend([dict(key,**r) for r in capture.verify(features)])
                with torch.no_grad():targets=teacher(x,deepsup=True) if domain=='U' else None
                regions,info,weights,edge_info=None,None,None,None
                if domain=='L':
                    bands,infos=b_losses.labeled_regions(label,distances,meta,2,142,case,obs['repeat'])
                    regions,info=bands['B'],infos['B'];assert info['status']=='valid'
                    valid=b.crop_native(np.ones(label.shape,bool),meta)
                    pairset=adjacent_pairs(y[0].cpu().numpy(),valid,seed_parts=(case,obs['repeat'],'L'))
                    native_mask=regions[0]|regions[1]
                    objective=Objectives('L',pairset,labels=y,regions=regions)
                    primary='transition_m1'
                else:
                    assert y is None and label is None
                    pairset=teacher_pairs(targets[-1],(case,obs['repeat']))
                    weights,edge_info=b_losses.teacher_regions(targets,[2],142,case,obs['repeat'])['B2']
                    assert edge_info['status']=='valid'
                    native_mask=weights>0
                    objective=Objectives('U',pairset,teacher=targets[-1],weights=weights,edge_info=edge_info)
                    primary='u_transition'
                geometry.append(dict(key,status=pairset.status,shape=list(pairset.shape),stats=pairset.stats,
                                     pair_endpoint_mask_fraction=float(np.mean(np.r_[native_mask[tuple(pairset.plus.T)],native_mask[tuple(pairset.minus.T)]])) if len(pairset.plus) else None))
                write_json(out/'geometry.json',geometry)
                assert pairset.valid,geometry[-1]
                np.savez_compressed(out/'raw'/f'{domain}_{obs["index"]}_pairs.npz',plus=pairset.plus,minus=pairset.minus)
                references,old_b_final=replay_reference(outputs,targets,y,regions,info,weights,edge_info,obs,domain,mapping)
                replays.extend(references)
                base=objective(outputs[-1]);base_values={name:float(v[0].detach()) for name,v in base.items()}
                new_b=base_values['boundary_ce_w2' if domain=='L' else 'edge_w2']
                assert np.isclose(new_b,old_b_final,rtol=1e-5,atol=1e-7)
                compatibility.append(dict(key,A_four_losses_replayed=True,B_four_losses_replayed=True,B_fp32=old_b_final,C0_fp64_same_formula=new_b,difference=new_b-old_b_final))
                masks={stage:b_losses.pool_native(native_mask,capture.nodes[stage].shape[2:],args.device)[0].detach() for stage in STAGES}
                energies={stage:(capture.nodes[stage].detach().double()*masks[stage].double()).square().sum((0,2,3,4)).sqrt().cpu().numpy() for stage in STAGES}
                medians={stage:float(torch.tensor(e[e>1e-12]).median()) if (e>1e-12).any() else 0. for stage,e in energies.items()}
                stats={};parameters=[mapping[s].parameter for s in STAGES]
                for name,loss in base.items():
                    gradients=torch.autograd.grad(loss[0],tuple(capture.nodes.values())+tuple(parameters),retain_graph=True)
                    gnodes,gparams=gradients[:5],gradients[5:]
                    stats[name]={}
                    if name==primary:
                        interface=[features['encoder' if s[0]=='E' else 'decoder'][s] for s in STAGES]
                        interface_grad=torch.autograd.grad(loss[0],interface,retain_graph=True)
                        for g,ref in zip(gnodes,interface_grad):torch.testing.assert_close(g,ref,rtol=0,atol=0)
                    for stage,g,gp in zip(STAGES,gnodes,gparams):
                        stat=node_statistics(g,capture.nodes[stage],masks[stage]);stats[name][stage]=stat
                        norms=np.linalg.norm(mapping[stage].channel_vectors(gp).detach().cpu().numpy().astype(float),axis=1)
                        for channel in range(len(norms)):
                            feature_rows.append(dict(key,objective=name,loss_head='D1',stage=stage,channel=channel,
                                coordinate_kind='feature_node',gradient_norm=stat['gradient_norm'][channel],masked_gradient_norm=stat['masked_gradient_norm'][channel],
                                signed_inner_product=stat['signed_inner'][channel],absolute_product_sum=stat['absolute_product_sum'][channel],activation_energy=stat['activation_energy'][channel],
                                status='zero_feature_gradient' if stat['gradient_norm'][channel]==0 else 'zero_directional_derivative' if stat['signed_inner'][channel]==0 else 'finite'))
                            parameter_rows.append(dict(key,objective=name,stage=stage,channel=channel,coordinate_kind='convolution_parameter',parameter=mapping[stage].name,
                                                       shape=list(gp.shape),output_axis=mapping[stage].output_axis,gradient_norm=norms[channel]))
                        np.save(out/'raw/parameters'/f'{domain}_{obs["index"]}_{name}_{stage}.npy',mapping[stage].channel_vectors(gp).detach().cpu().numpy())
                        if name==primary:
                            np.savez_compressed(out/'raw/features'/f'{domain}_{obs["index"]}_{stage}.npz',activation=capture.nodes[stage].detach().cpu().numpy(),
                                                gradient=g.detach().cpu().numpy(),mask=masks[stage].cpu().numpy())
                    del gradients,gnodes,gparams
                # Every uncertainty threshold is frozen from zero-dose observations
                # before the first nonzero perturbation for this crop.
                xb=x.expand(4,-1,-1,-1,-1).contiguous();nulls=defaultdict(list)
                with torch.no_grad():
                    for stage in STAGES:
                        channels=selected_channels(capture.nodes[stage].shape[1])
                        with attenuate(student,stage,[*channels,None],masks[stage],[0.]*4,capture.nodes[stage]) as trace:
                            null_logits=student(xb)
                        assert trace['calls']==1 and all(v==0 for v in trace['actual_l2'])
                        for name,values in objective(null_logits).items():nulls[name].extend(values.cpu().tolist())
                    epsilons={}
                    for name in base:
                        epsilon,record=calibrate_epsilon(base_values[name],nulls[name]);epsilons[name]=epsilon
                        noise.append(dict(key,objective=name,epsilon=epsilon,null_samples=nulls[name],**record))
                    write_json(out/'noise_calibration.json',noise)
                    for stage in STAGES:
                        channels=selected_channels(capture.nodes[stage].shape[1])
                        for dose in DOSES:
                            budget=dose*medians[stage]
                            chosen=[dose_alpha(energies[stage][c],budget) for c in channels]
                            alphas=[a if a is not None else 0. for a,status in chosen]
                            with attenuate(student,stage,[*channels,None],masks[stage],[*alphas,0.],capture.nodes[stage]) as trace:
                                logits=student(xb)
                            assert trace['calls']==1
                            measured=objective(logits)
                            metrics=prediction_metrics(logits,y,*regions) if domain=='L' else {}
                            changed=(logits.argmax(1)!=logits[3:4].argmax(1)).flatten(1).float().mean(1).cpu().numpy()
                            for row,channel in enumerate(channels):
                                alpha,status=chosen[row]
                                common=dict(key,stage=stage,channel=channel,dose=dose,status=status,alpha=alpha,target_l2=budget,
                                            actual_l2=trace['actual_l2'][row],baseline_node_max_error=trace['baseline_node_max_error'])
                                common.update(intended_l2=trace['intended_l2'][row],subtraction_roundoff_l2=trace['subtraction_roundoff_l2'][row])
                                metric=dict(common,gt_metrics_status='available_L' if domain=='L' else 'unavailable_no_U_GT')
                                if status=='valid':
                                    assert np.isclose(trace['actual_l2'][row],budget,rtol=2e-5,atol=1e-5)
                                    metric['changed_prediction_fraction']=changed[row]
                                    for metric_name,value in metrics.items():metric[metric_name+'_delta_fp32']=float(value[row]-value[3])
                                    if domain=='L':
                                        metric['ce_delta_fp64']=float(measured['region_ce'][row]-measured['region_ce'][3])
                                        metric['boundary_ce_delta_fp64']=float(measured['boundary_ce_w2'][row]-measured['boundary_ce_w2'][3])
                                metric_rows.append(metric)
                                for name,values in measured.items():
                                    record=dict(common,objective=name,base_loss_fp64=base_values[name],control_loss_fp64=float(values[3]),epsilon=epsilons[name])
                                    if status=='valid':
                                        predicted=-alpha*stats[name][stage]['signed_inner'][channel]
                                        actual=float(values[row]-values[3])
                                        assert np.isfinite(predicted) and np.isfinite(actual)
                                        record.update(delta_hat=predicted,delta_actual=actual,predicted_direction=direction(predicted,epsilons[name]),
                                                      actual_direction=direction(actual,epsilons[name]),absolute_residual=abs(actual-predicted),
                                                      relative_residual=abs(actual-predicted)/max(abs(predicted),epsilons[name]),
                                                      absolute_net_prediction=abs(predicted),absolute_gxa_sum=alpha*stats[name][stage]['absolute_product_sum'][channel])
                                    else:record.update(delta_hat=None,delta_actual=None,predicted_direction='undefined',actual_direction='undefined')
                                    delta_rows.append(record)
                            del logits,measured
                for file,records in [('feature_gradient_stats.csv',feature_rows),('parameter_gradient_stats.csv',parameter_rows),
                                     ('deltas.csv',delta_rows),('intervention_metrics.csv',metric_rows)]:write_csv(out/file,records)
                write_json(out/'node_identity.json',node_checks);write_json(out/'reference_replay.json',replays)
                write_json(out/'AB_compatibility.json',compatibility);write_json(out/'hdf5_access_log.json',access)
                print(f'C0 {domain} {case} crop{obs["repeat"]} pairs{len(pairset.plus)}; {time.monotonic()-started:.1f}s',flush=True)
                del base,outputs,features,capture,stats,x,y,xb,objective,targets
    invariants=dict(student_state_unchanged=b.unchanged(student,snapshots[0]),teacher_state_unchanged=b.unchanged(teacher,snapshots[1]),
                    teacher_frozen=all(not p.requires_grad for p in teacher.parameters()),no_grad_slots=all(p.grad is None for m in (student,teacher) for p in m.parameters()),
                    both_eval=all(not m.training for m in (student,teacher)),torch_rng_unchanged=torch.equal(cpu_rng,torch.get_rng_state()),
                    cuda_rng_unchanged=torch.equal(cuda_rng,torch.cuda.get_rng_state(args.device)),python_rng_unchanged=python_rng==random.getstate(),
                    numpy_rng_unchanged=all(np.array_equal(a,c) for a,c in zip(np_rng,np.random.get_state())),
                    no_leftover_hooks=all(not m._forward_hooks and not m._forward_pre_hooks for m in student.modules()),
                    image_only_U=all(r['key']=='image' for r in access if r['role']=='unlabeled'),
                    data_stats_unchanged=all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in file_stats.items()))
    assert all(invariants.values()),invariants
    result=dict(status='PASS',phase='C0',invariants=invariants,cases=4,crops=8,pair_scale='native final D1 only',
                delta_records=len(delta_rows),intervention_records=len(metric_rows),feature_stat_rows=len(feature_rows),parameter_stat_rows=len(parameter_rows),
                compatibility_vector_checks=len(replays),elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated(args.device)/2**20,
                optimizer_steps=0,checkpoint_writes=0,C1_C2_C3_executed=False,
                interpretation='Execution/structural checks PASS; final C0 gate also requires independently summarized fidelity and integrity checks')
    write_json(out/'execution_results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();out=output_path(args.output);out.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(run(args,out),indent=2))
    except BaseException:
        write_json(out/'execution_results.json',dict(status='FAIL',phase='C0',traceback=traceback.format_exc()));raise
