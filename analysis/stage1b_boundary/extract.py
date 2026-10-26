"""Smoke/full paired boundary-gradient extraction; no training or U GT."""
import argparse
import csv
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from common import (ROOT,A_RUN,SCALES,clean,crop_native,extract_terms,gradient_validity,
                    independent_losses,load_case_and_crops,load_models,stage_parameters,
                    state_snapshot,unchanged,verify_protection,write_json)
from losses import Term,distances,labeled_regions,labeled_terms,teacher_regions,unlabeled_terms
from interventions import attenuation

L_OBJECTIVES=[f'{kind}_w{width}' for width in (1,2,3) for kind in ('band','random','core')]
U_OBJECTIVES=['edge_w1','edge_w2','edge_w3','shuffle_w2']


def run(args,output):
    start=time.monotonic()
    original=json.loads((ROOT/'PROTECTION_BEFORE.json').read_text())
    student,teacher,design=load_models(args.device)
    observations=json.loads((A_RUN/'observations.json').read_text())
    selected={d:(v[:1] if args.smoke else v) for d,v in observations.items()}
    mapping=stage_parameters(student)
    assert {s:v.describe() for s,v in mapping.items()}==json.loads((A_RUN/'parameter_mapping.json').read_text())
    snapshots=[state_snapshot(m) for m in (student,teacher)]
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(args.device).clone() if args.device.startswith('cuda') else None
    torch.cuda.reset_peak_memory_stats(args.device) if args.device.startswith('cuda') else None
    plan=dict(stage='1B',source_A=str(A_RUN),checkpoint=design['checkpoint'],checkpoint_sha256=design['checkpoint_sha256'],
              paired_iteration=15000,seed=design['seed'],smoke=args.smoke,widths=[1,2,3],primary_width=2,
              labeled_ids=design['labeled_ids'],unlabeled_ids=design['unlabeled_ids'],patch=design['patch'],
              precision='FP32; TF32 off; eval; same as A',
              pooling='average joint class-region mass; conditional soft occupancy; native voxel coordinates',
              labeled_sign='positive 0.5 CE inner + 0.5 CE outer; user line minus interpreted as formatting error',
              normalization='Raw gradients primary; secondary contribution uses the unchanged A denominator, never independently normalizes B',
              device=args.device,L_objectives=L_OBJECTIVES,U_objectives=U_OBJECTIVES,
              gradient_absolute_floor=design['absolute_gradient_floor'],gradient_relative_floor=design['relative_gradient_floor'],
              teacher_thresholds=dict(levelset=.5,min_native_support=16,max_band_fraction=.5,min_span=1e-4,min_gradient=1e-6,min_effective_cells=2),
              data_split=dict(discovery_L=design['labeled_ids'][::2],probe_L=design['labeled_ids'][1::2],
                              discovery_U=design['unlabeled_ids'][::2],probe_U=design['unlabeled_ids'][1::2]),
              intervention=dict(primary_K=2,random_groups=8,regions=['boundary','random','core'],
                                calibration='energy-matched; budget = 0.1 * minimum region median nonzero channel activation norm; pair budget sqrt(2) times singleton budget',
                                alpha_cap=.5,rank_scale='D1',rank_score='signed cosine times relative U gradient norm',
                                selection='discovery cases only; primary width2; no test.list selection'))
    write_json(output/'design.json',plan)
    write_json(output/'parameter_mapping.json',{s:v.describe() for s,v in mapping.items()})
    arrays={}
    if not args.smoke:
        (output/'raw').mkdir()
        for domain,objectives in (('L',L_OBJECTIVES),('U',U_OBJECTIVES)):
            arrays[domain]={}
            for objective in objectives:
                arrays[domain][objective]={}
                for stage,spec in mapping.items():
                    shape=(len(observations[domain]),4,spec.channels,spec.parameter.numel()//spec.channels)
                    array=np.lib.format.open_memmap(output/'raw'/f'{domain}_{objective}_{stage}.npy',mode='w+',dtype=np.float32,shape=shape)
                    array[:]=np.nan;arrays[domain][objective][stage]=array
    access,records,reproduction,data_stats=[],[],[],{}
    with (output/'observation_norms.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=('domain','index','case','repeat','objective','scale','stage','channel','status',
                                               'raw_norm','validity_floor','raw_mean_contribution_norm','A_denominator_contribution_norm'),lineterminator='\n')
        writer.writeheader()
        for domain in ('L','U'):
            cases=list(dict.fromkeys(o['case'] for o in selected[domain]))
            for ci,case in enumerate(cases):
                current=[o for o in selected[domain] if o['case']==case]
                image,label,crops=load_case_and_crops(domain,case,design,current,access)
                file=Path(access[-1]['path']);stat=file.stat();data_stats[str(file)]=(stat.st_size,stat.st_mtime_ns)
                distance_maps=distances(label) if domain=='L' else None
                for obs,x,y,meta in crops:
                    x=x.to(args.device)
                    outputs,features=student(x,deepsup=True,return_features=True)
                    targets=None
                    if domain=='U':
                        assert y is None and label is None
                        with torch.no_grad():targets=teacher(x,deepsup=True)
                    a_losses,a_total,_=independent_losses(outputs,labels=y.to(args.device) if domain=='L' else None,teacher_outputs=targets)
                    np.testing.assert_allclose([float(v.detach()) for v in a_losses],obs['losses'],atol=1e-7,rtol=1e-5)
                    families={}
                    if domain=='L':
                        y_native=y[0].numpy()
                        for width in (1,2,3):
                            regions,info=labeled_regions(label,distance_maps,meta,width,design['seed'],case,obs['repeat'])
                            for key,name in (('B','band'),('R','random'),('I','core')):
                                families[f'{name}_w{width}']=labeled_terms(outputs,y_native,regions[key],info[key])
                    else:
                        predicted=teacher_regions(targets,[1,2,3],design['seed'],case,obs['repeat'])
                        for key,(weights,info) in predicted.items():
                            name=('edge_w'+key[1:]) if key.startswith('B') else 'shuffle_w2'
                            families[name]=unlabeled_terms(outputs,targets,weights,info)
                    for objective,terms in families.items():
                        gradients=extract_terms(terms,mapping)
                        for si,(scale,term) in enumerate(zip(SCALES,terms)):
                            records.append(dict(domain=domain,index=obs['index'],case=case,repeat=obs['repeat'],objective=objective,
                                                scale=scale,status=term.status,raw_loss=float(term.loss.detach()) if term.valid else None,
                                                A_lossnorm_factor=obs['normalized_scale_factor'],stats=term.stats))
                            for stage,spec in mapping.items():
                                vectors=gradients[stage][si]
                                if vectors is None:
                                    norms=[None]*spec.channels;floors=[None]*spec.channels
                                    reason='unreachable' if scale not in spec.reachable else 'objective_invalid:'+term.status
                                    statuses=[reason]*spec.channels
                                else:
                                    norms,statuses,floors=gradient_validity(vectors,design['absolute_gradient_floor'],design['relative_gradient_floor'])
                                    if not args.smoke:arrays[domain][objective][stage][obs['index'],si]=vectors
                                for channel,(norm,status,floor) in enumerate(zip(norms,statuses,floors)):
                                    writer.writerow(clean(dict(domain=domain,index=obs['index'],case=case,repeat=obs['repeat'],objective=objective,scale=scale,
                                                               stage=stage,channel=channel,status=status,raw_norm=norm,validity_floor=floor,
                                                               raw_mean_contribution_norm=None if norm is None else norm/4,
                                                               A_denominator_contribution_norm=None if norm is None else norm*obs['normalized_scale_factor'])))
                        del gradients
                    if args.smoke:
                        replay=extract_terms([Term(t,'valid',{}) for t in a_losses],mapping)
                        for stage,spec in mapping.items():
                            old=np.load(A_RUN/'raw'/f'{domain}_{stage}.npy',mmap_mode='r')[obs['index']]
                            for si,scale in enumerate(SCALES):
                                if replay[stage][si] is None:continue
                                difference=np.linalg.norm((replay[stage][si]-old[si]).astype(np.float64))
                                reference=np.linalg.norm(old[si].astype(np.float64))
                                relative=difference/max(reference,1e-30)
                                assert difference<=1e-9 or relative<=1e-4
                                reproduction.append(dict(domain=domain,stage=stage,scale=scale,relative_L2_error=relative))
                        # The final path must react and zero-alpha must be identity.
                        if domain=='L':
                            f=features['decoder']['D3']
                            with torch.no_grad(),attenuation(student,'D3',[list(range(f.shape[1]))],torch.ones((1,1,*f.shape[2:]),device=x.device),[.25]):
                                changed=student(x)
                            assert float((changed-outputs[-1].detach()).abs().max())>0
                    del families,outputs,features,targets,a_losses,a_total,x,y
                write_json(output/'hdf5_access_log.json',access)
                write_json(output/'loss_coverage.json',records)
                if not args.smoke:
                    for objective in arrays[domain].values():
                        for array in objective.values():array.flush()
                print(f'{"SMOKE" if args.smoke else "FULL"} {domain} {ci+1}/{len(cases)} {case}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    invariants=dict(student_state_unchanged=unchanged(student,snapshots[0]),teacher_state_unchanged=unchanged(teacher,snapshots[1]),
                    grad_slots_unchanged=all(p.grad is None for m in (student,teacher) for p in m.parameters()),
                    teacher_frozen=all(not p.requires_grad for p in teacher.parameters()),
                    both_eval=not student.training and not teacher.training,
                    torch_rng_unchanged=torch.equal(cpu_rng,torch.get_rng_state()),
                    cuda_rng_unchanged=cuda_rng is None or torch.equal(cuda_rng,torch.cuda.get_rng_state(args.device)),
                    unlabeled_image_only=all(r['key']=='image' for r in access if r['role']=='unlabeled'),
                    data_stats_unchanged=all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in data_stats.items()))
    protection=verify_protection(original)
    write_json(output/'protection_after.json',protection)
    assert all(invariants.values()) and protection['status']=='PASS'
    write_json(output/'A_replay.json',reproduction)
    result=dict(status='PASS',smoke=args.smoke,invariants=invariants,objective_scale_observations=len(records),
                raw_observations=sum(map(len,selected.values())),optimizer_steps=0,checkpoint_writes=0,
                elapsed_seconds=time.monotonic()-start,
                peak_allocated_mib=torch.cuda.max_memory_allocated(args.device)/2**20 if args.device.startswith('cuda') else None)
    write_json(output/'extraction_results.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-proof',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'runs'):parser.error('Output must be in Stage1B/runs')
    if not args.smoke:
        if args.smoke_proof is None:parser.error('Full extraction requires --smoke-proof from a successful smoke')
        smoke=json.loads(args.smoke_proof.read_text())
        assert smoke['status']=='PASS' and smoke['smoke']
    output.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(clean(run(args,output)),indent=2))
    except BaseException:
        write_json(output/'extraction_results.json',dict(status='FAIL',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
