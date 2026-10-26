"""Independent final-head diagnostic objectives, never imported by train.py."""
import torch
import torch.nn.functional as F
from support import b_losses


def check_final(logits,pairs):
    if not isinstance(logits,torch.Tensor) or logits.ndim!=5 or logits.shape[1]!=2:
        raise ValueError('Pass final D1 logits tensor, not a multi-head list')
    if tuple(logits.shape[2:])!=tuple(pairs.shape):raise ValueError('Pair/logit coordinate mismatch')
    if not pairs.valid:raise ValueError('Invalid position pairs: '+pairs.status)


def sample(volume,coordinates):
    c=torch.as_tensor(coordinates,device=volume.device,dtype=torch.long)
    return volume[:,c[:,0],c[:,1],c[:,2]]


def transition(logits,pairs,margin=1.):
    check_final(logits,pairs)
    z=logits.double()[:,1]-logits.double()[:,0]
    return F.softplus(margin-sample(z,pairs.plus)+sample(z,pairs.minus)).mean(1)


def probability_gap(logits,pairs):
    check_final(logits,pairs)
    p=logits.double().softmax(1)[:,1]
    return sample(p,pairs.plus)-sample(p,pairs.minus)


def pair_consistency(logits,teacher,pairs):
    target=probability_gap(teacher.detach(),pairs).detach()
    return (probability_gap(logits,pairs)-target).square().mean(1)


class Objectives:
    def __init__(self,domain,pairs,labels=None,regions=None,teacher=None,weights=None,edge_info=None):
        self.domain,self.pairs=domain,pairs
        if domain=='U' and (labels is not None or regions is not None):raise ValueError('U GT/GT regions forbidden')
        if domain not in ('L','U'):raise ValueError('Unknown domain')
        self.labels=labels.detach() if labels is not None else None
        self.regions=regions
        self.teacher=teacher.detach() if teacher is not None else None
        self.weights=weights;self.edge_info=edge_info
        if domain=='L' and (labels is None or regions is None):raise ValueError('L requires true labels and native band')
        if domain=='U' and (teacher is None or weights is None):raise ValueError('U requires detached predictions')

    def __call__(self,logits):
        check_final(logits,self.pairs)
        if self.domain=='L':
            ce=F.cross_entropy(logits.double(),self.labels.expand(logits.shape[0],-1,-1,-1),reduction='none')
            inner,outer=[torch.as_tensor(a,device=logits.device,dtype=torch.float64)[None] for a in self.regions]
            if min(float(inner.sum()),float(outer.sum()))<=0:raise ValueError('Empty boundary band')
            return dict(region_ce=ce.flatten(1).mean(1),boundary_ce_w2=.5*((ce*inner).flatten(1).sum(1)/inner.sum()+(ce*outer).flatten(1).sum(1)/outer.sum()),
                        transition_m0_5=transition(logits,self.pairs,.5),transition_m1=transition(logits,self.pairs,1.),transition_m2=transition(logits,self.pairs,2.))
        probability=logits.double().softmax(1)
        target=self.teacher.double().softmax(1).detach()
        ds=b_losses.spatial_gradient(probability[:,1:2]);dt=b_losses.spatial_gradient(target[:,1:2]).detach()
        weights=torch.as_tensor(self.weights,device=logits.device,dtype=torch.float64)[None,None].detach()
        if self.edge_info['status']!='valid' or float(weights.sum())<=0:raise ValueError('Invalid reference Teacher edge weights')
        return dict(mt=(probability-target).square().flatten(1).mean(1),
                    edge_w2=((ds-dt).square().mean(1,keepdim=True)*weights).flatten(1).sum(1)/weights.sum(),
                    u_transition=pair_consistency(logits,self.teacher,self.pairs))
