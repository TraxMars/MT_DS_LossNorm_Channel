"""Temporary feature attenuation for final predictions, never parameter updates."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

DETAIL_STAGES=("E3","E4","D3","D2","D1")


@contextmanager
def attenuation(model,stage,channel_sets,mask,alphas):
    """D3/D2 are intercepted on the downstream final-prediction path.

    Hooking only out_conv_3/out_conv_2 would modify an auxiliary head but leave
    final predictions unchanged. Their upstream auxiliary logits are not used
    as intervention outcomes here. Returned tensors are new, never in-place.
    """
    target={"E3":("block_three",False),"E4":("block_four",False),
            "D3":("block_seven",True),"D2":("block_eight",True),"D1":("out_conv",True)}
    name,pre=target[stage]
    module=getattr(model,name)
    stats={"calls":0}
    def apply(value):
        stats["calls"]+=1
        assert len(channel_sets)==value.shape[0]
        selector=torch.zeros((value.shape[0],value.shape[1],1,1,1),device=value.device,dtype=value.dtype)
        for row,channels in enumerate(channel_sets):selector[row,channels]=1
        a=torch.as_tensor(alphas,device=value.device,dtype=value.dtype).reshape(-1,1,1,1,1)
        change=value*selector*mask.to(value)*a
        stats["actual_l2"]=change.flatten(1).norm(dim=1).detach().cpu().tolist()
        return value-change
    if pre:
        handle=module.register_forward_pre_hook(lambda m,args:(apply(args[0]),))
    else:
        handle=module.register_forward_hook(lambda m,args,out:apply(out))
    try:
        yield stats
    finally:
        handle.remove()


def surface6(mask):
    padded=F.pad(mask.float(),(1,1,1,1,1,1),value=0)>0
    h,w,d=mask.shape[2:]
    eroded=mask.clone()
    for delta in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
        a,b,c=delta
        eroded=eroded & padded[:,:,1+a:1+a+h,1+b:1+b+w,1+c:1+c+d]
    return mask & ~eroded


def prediction_metrics(logits,labels,inner,outer):
    """Final-head Dice/CE and GT-contour F1 (Chebyshev tolerance=1 voxel)."""
    batch=logits.shape[0]
    truth=labels.bool().expand(batch,-1,-1,-1)[:,None]
    probability=logits.softmax(1)[:,1:2]
    predicted=probability>.5
    sumv=lambda value:value.flatten(1).sum(1)
    denom=sumv(predicted)+sumv(truth)
    dice=(2*sumv(predicted & truth)).float()/denom.clamp_min(1)
    soft=(2*sumv(probability*truth)+1e-8)/(sumv(probability)+sumv(truth)+1e-8)
    ce=F.cross_entropy(logits,labels.expand(batch,-1,-1,-1),reduction="none")[:,None]
    im=torch.as_tensor(inner,device=logits.device,dtype=logits.dtype)[None,None]
    om=torch.as_tensor(outer,device=logits.device,dtype=logits.dtype)[None,None]
    band_ce=.5*(sumv(ce*im)/im.sum().clamp_min(1)+sumv(ce*om)/om.sum().clamp_min(1))
    valid=torch.zeros_like(truth);valid[:,:,2:-2,2:-2,2:-2]=True
    ps,gs=surface6(predicted)&valid,surface6(truth)&valid
    pg=F.max_pool3d(gs.float(),3,1,1)>0
    gp=F.max_pool3d(ps.float(),3,1,1)>0
    precision=sumv(ps & pg).float()/sumv(ps).clamp_min(1)
    recall=sumv(gs & gp).float()/sumv(gs).clamp_min(1)
    bf1=2*precision*recall/(precision+recall).clamp_min(1e-12)
    bf1=torch.where(sumv(gs)>0,bf1,torch.full_like(bf1,float("nan")))
    return dict(dice=dice,soft_dice=soft,ce=ce.flatten(1).mean(1),boundary_ce=band_ce,boundary_f1=bf1)
