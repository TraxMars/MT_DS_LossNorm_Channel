"""Capture and attenuate exactly the same final-prediction graph sites."""
from contextlib import contextmanager
import torch
from support import STAGES

NODE_SPECS={'E3':('block_three','output'),'E4':('block_four','output'),
            'D3':('block_seven','input'),'D2':('block_eight','input'),'D1':('out_conv','input')}


class Capture:
    def __init__(self,model):self.model=model;self.nodes={};self.branches={};self.calls={};self.handles=[]
    def __enter__(self):
        for stage,(name,site) in NODE_SPECS.items():
            def save(t,stage=stage):
                self.nodes[stage]=t;self.calls[stage]=self.calls.get(stage,0)+1
            module=getattr(self.model,name)
            if site=='input':self.handles.append(module.register_forward_pre_hook(lambda m,args,save=save:save(args[0])))
            else:self.handles.append(module.register_forward_hook(lambda m,args,out,save=save:save(out)))
        for name in ('block_six_up','block_seven_up'):
            self.handles.append(getattr(self.model,name).register_forward_hook(lambda m,args,out,name=name:self.branches.__setitem__(name,out)))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
    def verify(self,features):
        rows=[]
        for stage in STAGES:
            feature=features['encoder' if stage[0]=='E' else 'decoder'][stage]
            assert self.nodes[stage] is feature and self.calls[stage]==1
            rows.append(dict(stage=stage,module=NODE_SPECS[stage][0],site=NODE_SPECS[stage][1],same_object=True,
                             shape=list(feature.shape),requires_grad=feature.requires_grad,forward_calls=self.calls[stage]))
        for stage,name,skip in [('D3','block_six_up','E3'),('D2','block_seven_up','E2')]:
            expected=self.branches[name]+features['encoder'][skip]
            assert torch.equal(expected,self.nodes[stage])
            next(r for r in rows if r['stage']==stage)['skip_fusion_max_error']=float((expected-self.nodes[stage]).detach().abs().max())
        return rows


@contextmanager
def attenuate(model,stage,channels,mask,alphas,baseline=None):
    name,site=NODE_SPECS[stage]
    stats={'calls':0}
    def apply(value):
        stats['calls']+=1
        if len(channels)!=value.shape[0] or len(alphas)!=value.shape[0]:raise ValueError('Batch contract mismatch')
        selector=torch.zeros((len(channels),value.shape[1],1,1,1),device=value.device,dtype=value.dtype)
        for row,c in enumerate(channels):
            if c is not None:selector[row,c]=1
        a=torch.as_tensor(alphas,device=value.device,dtype=value.dtype).detach().reshape(-1,1,1,1,1)
        delta=value*selector*mask.detach().to(value)*a
        changed=value-delta
        actual=value.detach()-changed.detach()
        stats['intended_l2']=delta.detach().double().flatten(1).norm(dim=1).cpu().tolist()
        stats['actual_l2']=actual.double().flatten(1).norm(dim=1).cpu().tolist()
        stats['subtraction_roundoff_l2']=(actual.double()-delta.detach().double()).flatten(1).norm(dim=1).cpu().tolist()
        if baseline is not None:
            stats['baseline_node_max_error']=float((value-baseline.detach()).abs().max())
            torch.testing.assert_close(value,baseline.detach().expand_as(value),rtol=1e-6,atol=1e-7)
        return changed
    module=getattr(model,name)
    handle=(module.register_forward_pre_hook(lambda m,args:(apply(args[0]),)) if site=='input' else
            module.register_forward_hook(lambda m,args,out:apply(out)))
    try:yield stats
    finally:handle.remove()
