"""Record Stage1A reconstruction errors under controlled FP32/FP64 backward.

This calibration retains the exact Stage1A body and disables only its two
reconstruction-error raises to record their values. Production extraction keeps
the original strict checker; this function is never used to accept measurements.
"""
import argparse
import ast
import copy
import inspect
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import ROOT,SPECS,checkpoint,new_output,write_json
from diagnostic import independent_losses,load_crop,preserve,stage_parameters
import core
from networks import VNetDS

def recording_gradients():
    tree=ast.parse(inspect.getsource(core.parameter_gradients))
    removed=[]
    class Record(ast.NodeTransformer):
        def visit_Raise(self,node):
            text=ast.unparse(node)
            if 'gradient reconstruction' in text:
                removed.append(text);return ast.copy_location(ast.Pass(),node)
            return node
    tree=Record().visit(tree);assert len(removed)==2
    namespace=dict(core.__dict__)
    exec(compile(tree,'<Stage1A recording-only calibration>','exec'),namespace)
    return namespace['parameter_gradients']

def run(dtype,out,ds='LA'):
    torch.set_num_threads(4);started=time.monotonic()
    ck,manifest,path=checkpoint(ds)
    models=[VNetDS(n_channels=1,n_classes=2,normalization='batchnorm',has_dropout=True).to(dtype).eval() for _ in range(2)]
    s,t=models;s.load_state_dict(ck['student']);t.load_state_dict(ck['teacher']);t.requires_grad_(False)
    config=ck['config'];spec=SPECS[ds]
    ids=[manifest['unlabeled_ids'][i] for i in sorted(np.random.default_rng(142).choice(spec['unlabeled'],16,replace=False))]
    mapping={k:v for k,v in stage_parameters(s).items() if k in ('D1','E3')}
    access=[];stats={}
    with preserve(models,train=True,seed=10001):
        crops=[load_crop(ds,config['dataset_root'],c,'L',spec['patch'],1,142,access,stats) for c in manifest['labeled_ids'][:2]]
        x=torch.cat([c[0] for c in crops]).to(dtype)
        with torch.no_grad(): s(x,deepsup=True)
        crops=[load_crop(ds,config['dataset_root'],c,'U',spec['patch'],1,142,access,stats) for c in ids[:2]]
        x=torch.cat([c[0] for c in crops]).to(dtype)
        outputs=s(x,deepsup=True)
        with torch.no_grad(): targets=t(x,deepsup=True)
        losses,total,_=independent_losses(outputs,teacher_outputs=targets)
        raw,checks=recording_gradients()(losses,total,mapping,weight=.3)
        for stage,values in raw.items():
            mean=sum(v for v in values if v is not None)/4
            for row in checks['gradient_reconstruction']:
                if row['stage']==stage:
                    row['raw_reconstructed_norm']=float(np.linalg.norm(mean.astype(np.float64)))
                    row['lossnorm_reconstructed_norm']=float(np.linalg.norm(mean.astype(np.float64))*.3/checks['lossnorm_denominator'])
        write_json(out/'calibration.json',dict(dataset=ds,dtype=str(dtype),checks=checks,elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dtype',choices=('float32','float64'),required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=new_output('D1_gradient_diagnosis',args.output)
    for ds in SPECS:
        sub=out/ds;sub.mkdir();run(getattr(torch,args.dtype),sub,ds)
