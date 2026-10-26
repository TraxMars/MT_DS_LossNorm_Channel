"""D1-A parameter gradients, with transactional model/RNG restoration."""
import ast
from contextlib import contextmanager
import pickle
from pathlib import Path
import random
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import PROJECT,sha256
from core import (SCALES,deterministic_crop,gradient_validity,independent_losses,
                  parameter_gradients,read_image_only,read_labeled_case,signed_cosine,stage_parameters)

def c0_capture():
    """Load exactly the generic C0 identity code, without support/boundary imports.

    Only NODE_SPECS and Capture are selected from the local, audited Python AST.
    No attenuation, position targets, boundary losses or C0 support is executed.
    """
    path=PROJECT/'analysis/stage1c_boundary_attribution/nodes.py'
    tree=ast.parse(path.read_text())
    selected=[node for node in tree.body if
              (isinstance(node,ast.ClassDef) and node.name=='Capture') or
              (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NODE_SPECS' for t in node.targets))]
    assert len(selected)==2
    namespace={'torch':torch,'STAGES':('E3','E4','D3','D2','D1')}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['Capture']

Capture=c0_capture()

@contextmanager
def preserve(models,train=False,seed=None):
    """Restore parameters, BN, modes, grad slots and all random states, even on errors."""
    states=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
    flags=[[(sub,sub.training) for sub in m.modules()] for m in models]
    params=[[(p,p.requires_grad,None if p.grad is None else p.grad.detach().clone()) for p in m.parameters()] for m in models]
    rng=(torch.get_rng_state().clone(),np.random.get_state(),random.getstate(),
         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    report={}
    try:
        for m in models: m.train(train)
        if seed is not None:
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        yield report
    finally:
        report['parameters_unchanged_before_restore']=all(
            torch.equal(p.detach(),state[name]) for m,state in zip(models,states) for name,p in m.named_parameters())
        report['changed_buffers_before_restore']=[
            f'{index}:{name}' for index,(m,state) in enumerate(zip(models,states))
            for name,b in m.named_buffers() if not torch.equal(b,state[name])]
        report['grad_slots_unchanged']=all(
            (prior is None and p.grad is None) or (prior is not None and p.grad is not None and torch.equal(prior,p.grad))
            for group in params for p,_,prior in group)
        for m,state,group,mode_flags in zip(models,states,params,flags):
            m.load_state_dict(state,strict=True)
            for p,requires_grad,grad in group:
                p.requires_grad_(requires_grad);p.grad=grad
            for sub,training in mode_flags: sub.training=training
        torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])
        if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])
        report['state_restored']=all(torch.equal(v,states[i][k]) for i,m in enumerate(models) for k,v in m.state_dict().items())
        report['modes_restored']=all(sub.training==flag for group in flags for sub,flag in group)
        report['requires_grad_restored']=all(p.requires_grad==r for group in params for p,r,_ in group)
        report['rng_restored']=(torch.equal(torch.get_rng_state(),rng[0]) and
            pickle.dumps(np.random.get_state())==pickle.dumps(rng[1]) and random.getstate()==rng[2] and
            (rng[3] is None or all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng[3]))))
        assert all(report[k] for k in ('parameters_unchanged_before_restore','grad_slots_unchanged',
                                      'state_restored','modes_restored','requires_grad_restored','rng_restored')),report

def extract(student,teacher,x,y,mapping,config):
    with Capture(student) as capture:
        outputs,features=student(x,deepsup=True,return_features=True)
        identities=capture.verify(features)
        targets=None
        if y is None:
            with torch.no_grad(): targets=teacher(x,deepsup=True)
            assert all(not t.requires_grad for t in targets)
        losses,total,details=independent_losses(outputs,labels=y,teacher_outputs=targets)
        assert torch.isfinite(torch.stack(losses)).all()
        gradients,checks=parameter_gradients(losses,total,mapping,eps=config['normalization_eps'],
                                            weight=1. if y is not None else config['consistency'])
        for stage,items in gradients.items():
            for s,g in zip(SCALES,items):
                if g is not None: assert np.isfinite(g).all(),(stage,s)
        meta=dict(losses=[float(l.detach()) for l in losses],aggregate_loss=float(total.detach()),
                  details=details,checks=checks,node_identity=identities)
    return gradients,meta

def pack(gradients):
    result={}
    for stage,values in gradients.items():
        shape=next(v.shape for v in values if v is not None)
        result[stage]=np.stack([np.full(shape,np.nan,dtype=np.float32) if v is None else v for v in values])
    return result

def measure_unit(student,teacher,inputs,mapping,config,train=False,seed=None):
    """Strict FP32 first; retry the entire unit in FP64 on reconstruction failure.

    Same rule for both datasets/modes, independent of cosine sign or performance.
    Original Stage1A thresholds/formulas remain unchanged. Both L/U roles use the
    same accepted precision; failed partial measurements are never published.
    """
    failures=[]
    for dtype in (torch.float32,torch.float64):
        if dtype==torch.float64:
            student.double();teacher.double()
        try:
            result={}
            with preserve((student,teacher),train=train,seed=seed) as protection:
                for role,(x,y) in inputs.items():
                    values,meta=extract(student,teacher,x.to(dtype=dtype),y,mapping,config)
                    meta['gradient_precision']=str(dtype)
                    result[role]=(pack(values),meta)
            protection['accepted_precision']=str(dtype)
            protection['failed_attempts']=failures
            return result,protection
        except AssertionError as error:
            if dtype==torch.float64 or 'gradient reconstruction' not in str(error): raise
            failures.append(dict(precision=str(dtype),reason=str(error),restoration=protection))
        finally:
            if dtype==torch.float64:
                student.float();teacher.float()
    raise AssertionError('Unreachable precision policy')

def load_crop(dataset,root,case,role,patch,repeat,seed,access,stats):
    from data import case_path
    path=case_path(root,dataset,case)
    s=path.stat();stats[str(path)]=(s.st_size,s.st_mtime_ns)
    if role=='L': image,label=read_labeled_case(path,access,case)
    else: image,label=read_image_only(path,access,'unlabeled',case),None
    x,y,crop=deterministic_crop(image,label,patch,seed,case,repeat)
    crop.update(case=case,repeat=repeat,role=role)
    return x,y,crop
