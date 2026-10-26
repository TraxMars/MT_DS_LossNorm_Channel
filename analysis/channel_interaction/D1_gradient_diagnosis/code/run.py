"""Paired-dataset D1-A. Full patches; eval first, then 2L+2U; no optimizer."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import PROJECT,ROOT,SPECS,checkpoint,new_output,sha256,write_json
from diagnostic import SCALES,load_crop,measure_unit,stage_parameters
from networks import VNetDS

def save_observation(out,records,raw,meta):
    index=len(records);name=f'observation_{index:03d}.npz'
    np.savez(out/'raw'/name,**raw)
    meta['raw_file']='raw/'+name;meta['index']=index
    records.append(meta)
    write_json(out/'observations.json',records)

def dataset_run(ds,args,out):
    started=time.monotonic();out.mkdir();(out/'raw').mkdir()
    ck,manifest,path=checkpoint(ds);config=ck['config'];spec=SPECS[ds]
    device=torch.device(args.device)
    student=VNetDS(n_channels=1,n_classes=2,normalization='batchnorm',has_dropout=True).to(device).eval()
    teacher=VNetDS(n_channels=1,n_classes=2,normalization='batchnorm',has_dropout=True).to(device).eval()
    student.load_state_dict(ck['student'],strict=True);teacher.load_state_dict(ck['teacher'],strict=True)
    teacher.requires_grad_(False);del ck
    mapping={k:v for k,v in stage_parameters(student).items() if k in ('E3','D1')}
    labeled=manifest['labeled_ids']
    selected=sorted(np.random.default_rng(args.seed).choice(len(manifest['unlabeled_ids']),16,replace=False))
    unlabeled=[manifest['unlabeled_ids'][i] for i in selected]
    if ds=='LA':
        historical=json.loads((PROJECT/'analysis/stage1a_gradients/runs/la10_iter15000_seed142_fp32/design.json').read_text())
        assert historical['labeled_ids']==labeled and historical['unlabeled_ids']==unlabeled
        assert historical['checkpoint_sha256']==sha256(path)
    design=dict(dataset=ds,checkpoint=str(path),checkpoint_sha256=sha256(path),iteration=15000,
                labeled_ids=labeled,unlabeled_ids=unlabeled,unlabeled_population=spec['unlabeled'],
                patch=spec['patch'],seed=args.seed,repeats=2,eval_batch_size=1,train_batch='2L+2U',
                mini_batches=3,mini_repeats=2,train_dropout_seeds='10000 + 100*batch + repeat',
                order='all eval single-case; matched eval mini; train mini; deterministic train replay',
                root=config['dataset_root'],device=str(device),torch=torch.__version__,dtype='float32',TF32=False,
                augmentation='same Stage1A uniform image-shape-only crops, no rotations',
                model_mapping={s:p.describe() for s,p in mapping.items()},
                precision_policy='FP32 first; on original Stage1A reconstruction failure restore and replay entire L/U unit in FP64; original thresholds unchanged',
                lossnorm='weight*g_head/(4*max(mean_four_head_loss,1e-8)); L weight=1, U weight=.3',
                validity='max(1e-12,1e-6 * maximum channel norm), computed separately per side/observation/head',
                statistics='all L x U case pairs; separate L/U case bootstrap 200 draws; mini descriptive only',
                no_channel_correspondence_across_datasets=True)
    write_json(out/'design.json',design)
    records=[];access=[];stats={};restorations=[]
    for role,cases in (('L',labeled),('U',unlabeled)):
        for case in cases:
            for repeat in range(2):
                x,y,crop=load_crop(ds,config['dataset_root'],case,role,spec['patch'],repeat,args.seed,access,stats)
                measured,protection=measure_unit(student,teacher,{role:(x.to(device),None if y is None else y.to(device))},mapping,config)
                raw,meta=measured[role]
                assert not protection['changed_buffers_before_restore']
                restorations.append(protection)
                save_observation(out,records,raw,dict(mode='eval',unit='case',role=role,cases=[case],
                                 repeat=repeat,crops=[crop],**meta))
                write_json(out/'restoration_checks.json',restorations);write_json(out/'hdf5_access_log.json',access)
                print(f'{ds} eval {role} {case} crop {repeat}: {time.monotonic()-started:.1f}s',flush=True)
    # Mini-batches use two distinct L and U cases; same deterministic crops in both modes.
    # LA pairs 0,1 / 2,3 / 4,5; its last two L cases participate in single-case eval.
    batches=[dict(L=[labeled[(2*b+j)%len(labeled)] for j in range(2)],U=unlabeled[2*b:2*b+2]) for b in range(3)]
    write_json(out/'mini_batch_design.json',batches)
    replay_reference=None
    for mode in ('eval','train'):
        for b,cases in enumerate(batches):
            for repeat in range(2):
                seed=10000+100*b+repeat
                inputs={};cropmeta={}
                for role in ('L','U'):
                    crops=[load_crop(ds,config['dataset_root'],case,role,spec['patch'],repeat,args.seed,access,stats) for case in cases[role]]
                    x=torch.cat([c[0] for c in crops]).to(device)
                    y=torch.cat([c[1] for c in crops]).to(device) if role=='L' else None
                    inputs[role]=(x,y);cropmeta[role]=[c[2] for c in crops]
                measured,protection=measure_unit(student,teacher,inputs,mapping,config,train=mode=='train',seed=seed)
                for role,(values,meta) in measured.items():
                    save_observation(out,records,values,dict(mode=mode,unit='mini',role=role,cases=cases[role],
                                     batch=b,repeat=repeat,dropout_seed=seed,crops=cropmeta[role],**meta))
                    if mode=='train' and b==0 and repeat==0 and role=='U': replay_reference=values
                if mode=='train': assert protection['changed_buffers_before_restore']
                restorations.append(dict(mode=mode,batch=b,repeat=repeat,**protection))
                write_json(out/'restoration_checks.json',restorations);write_json(out/'hdf5_access_log.json',access)
                if protection['failed_attempts']:print(f'{ds} {mode} mini {b}/{repeat}: FP64 numerical replay',flush=True)
                print(f'{ds} {mode} mini {b} repeat {repeat}: {time.monotonic()-started:.1f}s',flush=True)
    # Replay the full first 2L+2U unit with exactly the same Dropout seed and initial BN state.
    inputs={}
    for role in ('L','U'):
        crops=[load_crop(ds,config['dataset_root'],case,role,spec['patch'],0,args.seed,access,stats) for case in batches[0][role]]
        x=torch.cat([c[0] for c in crops]).to(device)
        y=torch.cat([c[1] for c in crops]).to(device) if role=='L' else None
        inputs[role]=(x,y)
    measured,protection=measure_unit(student,teacher,inputs,mapping,config,train=True,seed=10000)
    replay=measured['U'][0]
    restorations.append(dict(mode='train_replay',**protection))
    replay_errors={s:float(np.nanmax(np.abs(replay[s]-replay_reference[s]))) for s in replay}
    assert all(v==0 for v in replay_errors.values()),replay_errors
    assert all(r['key']=='image' for r in access if r['role']=='unlabeled')
    assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in stats.items())
    assert sha256(path)==design['checkpoint_sha256']
    write_json(out/'hdf5_access_log.json',access)
    write_json(out/'restoration_checks.json',restorations)
    write_json(out/'data_file_stats.json',stats)
    result=dict(status='PASS',observations=len(records),unlabeled_gt_reads=0,optimizer_instances=0,optimizer_steps=0,
                parameter_updates=0,checkpoint_writes=0,train_replay_max_abs=replay_errors,
                fp64_fallback_units=sum(bool(r.get('failed_attempts')) for r in restorations),
                state_restoration_checks=len(restorations),elapsed_seconds=time.monotonic()-started)
    write_json(out/'execution.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=142);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--d0',type=Path,default=ROOT/'D0_integrity/results/round1/integrity.json')
    args=p.parse_args();assert args.seed==142
    gate=json.loads(args.d0.read_text())
    assert gate['status']=='PASS' and set(gate['datasets'])==set(SPECS)
    out=new_output('D1_gradient_diagnosis',args.output)
    write_json(out/'D0_gate.json',dict(path=str(args.d0.resolve()),sha256=sha256(args.d0),status='PASS'))
    torch.set_num_threads(args.threads);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    write_json(out/'source_manifest.json',{str(p.relative_to(PROJECT)):sha256(p) for p in [
        *sorted((ROOT/'D1_gradient_diagnosis/code').glob('*.py')),ROOT/'common.py',
        PROJECT/'analysis/stage1a_gradients/core.py',PROJECT/'analysis/stage1c_boundary_attribution/nodes.py',
        PROJECT/'code/objectives.py',PROJECT/'code/networks/vnet_ds.py']})
    try:
        results={ds:dataset_run(ds,args,out/ds) for ds in SPECS}
        write_json(out/'execution.json',dict(status='PASS',datasets=results))
    except BaseException:
        write_json(out/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise

if __name__=='__main__': main()
