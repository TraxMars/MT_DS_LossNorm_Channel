"""Diagnostic boundary objectives only. Not imported by the training code."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from common import crop_native, fixed_rng


@dataclass
class Term:
    loss: object
    status: str
    stats: dict

    @property
    def valid(self):
        return self.loss is not None and self.status == "valid"


def distances(label):
    label = np.asarray(label, dtype=bool)
    # Full-volume geometry, before cropping; explicit outside-volume background.
    inner = ndimage.distance_transform_edt(np.pad(label, 1))[1:-1,1:-1,1:-1]
    outer = ndimage.distance_transform_edt(~label)
    return inner, outer


def labeled_regions(label, distance_maps, meta, width, seed, case, repeat):
    y = crop_native(label, meta).astype(bool)
    valid = crop_native(np.ones(label.shape, dtype=bool), meta)
    di, do = [crop_native(x, meta) for x in distance_maps]
    inner = y & valid & (di > 0) & (di <= width)
    outer = ~y & valid & (do > 0) & (do <= width)
    ni, no = int(inner.sum()), int(outer.sum())
    result = {"B": (inner, outer)}
    stats = {"B": dict(native_inner=ni, native_outer=no, status="valid" if min(ni,no)>=8 else "no_effective_boundary")}
    for name, fg, bg in (("R", y & valid, ~y & valid),
                         ("I", y & valid & (di > width), ~y & valid & (do > width))):
        generator = fixed_rng(seed, case, repeat, width, name)
        fi, bi = np.flatnonzero(fg), np.flatnonzero(bg)
        info = dict(native_inner=ni, native_outer=no, available_foreground=len(fi), available_background=len(bi))
        if min(ni,no)<8 or len(fi)<ni or len(bi)<no:
            result[name] = None
            stats[name] = dict(**info,status="insufficient_matched_region")
            continue
        a, b = np.zeros(y.shape,bool), np.zeros(y.shape,bool)
        a.flat[generator.choice(fi,ni,replace=False)] = True
        b.flat[generator.choice(bi,no,replace=False)] = True
        assert a.sum()==ni and b.sum()==no and not (a & b).any()
        result[name]=(a,b)
        stats[name]=dict(**info,status="valid",boundary_overlap_fraction=float(((a|b)&(inner|outer)).sum()/max(ni+no,1)))
    return result,stats


def pool_native(array, size, device):
    tensor = torch.as_tensor(np.asarray(array,dtype=np.float32)[None,None],device=device)
    factors = tuple(a//b for a,b in zip(array.shape,size))
    if any(a%b for a,b in zip(array.shape,size)):
        raise ValueError("Volume average requires divisible sizes")
    return F.avg_pool3d(tensor,kernel_size=factors,stride=factors),factors


def weight_stats(weights):
    array=weights.detach().double()
    total=float(array.sum())
    squared=float(array.square().sum())
    support=int((array>0).sum())
    return dict(mass=total,support=support,support_fraction=support/array.numel(),
                effective_cells=total*total/squared if squared>0 else 0,
                maximum=float(array.max()),mean=float(array.mean()))


def labeled_terms(outputs, label_crop, regions, native_info):
    terms=[]
    for output in outputs:
        if regions is None or native_info["status"]!="valid":
            terms.append(Term(None,native_info["status"],dict(native_info)))
            continue
        logp=output.log_softmax(1)
        occupancy,_=pool_native(label_crop,output.shape[2:],output.device)
        parts=[]
        info=dict(native_info)
        valid=True
        for side,mask in zip(("inner","outer"),regions):
            weight,factors=pool_native(mask,output.shape[2:],output.device)
            # Joint class/region masses are averaged, rather than multiplying
            # separately averaged GT and mask. This exactly conserves native CE
            # when the coarse logit is constant within each pooling cell.
            fg,_=pool_native(mask*np.asarray(label_crop),output.shape[2:],output.device)
            bg=weight-fg
            metrics=weight_stats(weight)
            info.update({side+"_"+key:value for key,value in metrics.items()})
            info[side+"_occupancy_mean"]=float((weight*occupancy).sum()/weight.sum()) if metrics["mass"]>0 else None
            info[side+"_conditional_foreground"]=float(fg.sum()/weight.sum()) if metrics["mass"]>0 else None
            info[side+"_reconstructed_native_mass"]=metrics["mass"]*np.prod(factors)
            if metrics["mass"]<=0 or metrics["effective_cells"]<2:
                valid=False
            else:
                parts.append(-(fg*logp[:,1:2]+bg*logp[:,0:1]).sum()/weight.sum())
        terms.append(Term(.5*(parts[0]+parts[1]) if valid else None,
                          "valid" if valid else "insufficient_coarse_boundary_coverage",info))
    return terms


def spatial_gradient(probability,spacing=(1,1,1)):
    """Central 3D differences, one-sided endpoints, units/original-image voxel."""
    components=[]
    for axis,h in zip((2,3,4),spacing):
        n=probability.shape[axis]
        if n==1:
            components.append(probability*0)
            continue
        def cut(start,stop):
            slices=[slice(None)]*5;slices[axis]=slice(start,stop)
            return probability[tuple(slices)]
        first=(cut(1,2)-cut(0,1))/h
        last=(cut(n-1,n)-cut(n-2,n-1))/h
        middle=(cut(2,n)-cut(0,n-2))/(2*h)
        components.append(torch.cat((first,middle,last),dim=axis) if n>2 else torch.cat((first,last),dim=axis))
    return torch.cat(components,dim=1)


def teacher_regions(teacher_outputs,widths,seed,case,repeat):
    probability=teacher_outputs[-1].detach().softmax(1)[:,1:2]
    p=probability[0,0].cpu().numpy()
    hard=p>=.5
    gradient=spatial_gradient(probability).square().sum(1).sqrt()[0].cpu().numpy()
    interior=np.zeros(p.shape,bool);interior[4:-4,4:-4,4:-4]=True
    di,do=distances(hard)
    level=((hard & (di<=1))|(~hard & (do<=1))) & interior
    base=dict(probability_min=float(p.min()),probability_max=float(p.max()),
              probability_span=float(np.ptp(p)),teacher_foreground_fraction=float(hard.mean()),
              teacher_gradient_max=float(gradient.max()),candidate_levelset_voxels=int(level.sum()))
    reason="valid"
    if np.ptp(p)<1e-4 or gradient.max()<1e-6:
        reason="teacher_near_constant_probability"
    elif not hard.any() or hard.all() or level.sum()<16:
        reason="teacher_insufficient_levelset"
    result={}
    for width in widths:
        band=((hard & (di<=width))|(~hard & (do<=width))) & interior
        fraction=float(band.sum()/max(interior.sum(),1))
        weight=band.astype(np.float32)*gradient
        if weight.max()>0:weight/=weight.max()
        status=reason
        if status=="valid" and fraction>.5:status="teacher_boundary_too_broad"
        if status=="valid" and np.count_nonzero(weight)<16:status="teacher_insufficient_weight_support"
        info=dict(base,width=width,native_band_fraction=fraction,native_support=int(np.count_nonzero(weight)),
                  native_weight_sum=float(weight.sum()),status=status,
                  teacher_weights_detached=True,confidence_is_not_boundary_accuracy=True)
        result[f"B{width}"]=(weight,info)
        if width==2:
            shuffled=np.zeros_like(weight)
            values=weight[interior].copy()
            fixed_rng(seed,case,repeat,"teacher_shuffle").shuffle(values)
            shuffled[interior]=values
            result["R2"]=(shuffled,dict(info,control="spatial_permutation_same_native_weight_histogram"))
    return result


def unlabeled_terms(outputs,teacher_outputs,weights,info):
    result=[]
    native_shape=weights.shape
    for student,teacher in zip(outputs,teacher_outputs):
        metrics=dict(info)
        w,_=pool_native(weights,student.shape[2:],student.device)
        metrics.update(weight_stats(w))
        spacing=tuple(a/b for a,b in zip(native_shape,student.shape[2:]))
        ds=spatial_gradient(student.softmax(1)[:,1:2],spacing)
        dt=spatial_gradient(teacher.detach().softmax(1)[:,1:2],spacing).detach()
        metrics["scale_teacher_gradient_max"]=float(dt.abs().max())
        status=info["status"]
        if status=="valid" and (metrics["mass"]<=0 or metrics["effective_cells"]<2):
            status="insufficient_coarse_teacher_coverage"
        if status=="valid" and metrics["scale_teacher_gradient_max"]<1e-6:
            status="teacher_scale_gradient_near_zero"
        loss=(w*(ds-dt).square().mean(1,keepdim=True)).sum()/w.sum() if status=="valid" else None
        result.append(Term(loss,status,metrics))
    return result
