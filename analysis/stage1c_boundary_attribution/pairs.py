"""Native 6-adjacent oriented pairs; no coarse pseudo-boundaries or U GT."""
from dataclasses import dataclass,replace
import numpy as np
from support import fixed_rng


@dataclass
class PairSet:
    plus: np.ndarray
    minus: np.ndarray
    shape: tuple
    status: str
    stats: dict

    @property
    def valid(self):return self.status=='valid'


def adjacent_pairs(label,valid=None,border=4,maximum=2048,minimum=32,seed_parts=()):
    y=np.asarray(label)
    if y.ndim!=3 or not np.isfinite(y).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Binary finite native grid required')
    y=y.astype(bool)
    allowed=np.ones_like(y,bool) if valid is None else np.asarray(valid,dtype=bool).copy()
    if allowed.shape!=y.shape:raise ValueError('Validity grid mismatch')
    interior=np.zeros_like(y,bool)
    if all(n>2*border for n in y.shape):
        interior[tuple(slice(border,n-border) for n in y.shape)]=True
    allowed &= interior
    positive,negative=[],[]
    for axis in range(3):
        a=[slice(None)]*3;b=[slice(None)]*3
        a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        cross=(y[a]!=y[b])&allowed[a]&allowed[b]
        p=np.argwhere(cross);q=p.copy();q[:,axis]+=1
        first_fg=y[tuple(p.T)]
        positive.append(np.where(first_fg[:,None],p,q))
        negative.append(np.where(first_fg[:,None],q,p))
    plus=np.concatenate(positive).astype(np.int64);minus=np.concatenate(negative).astype(np.int64)
    candidates=len(plus)
    if maximum is not None and len(plus)>maximum:
        take=fixed_rng(142,'C0_pairs',*seed_parts).choice(len(plus),maximum,replace=False)
        take.sort();plus,minus=plus[take],minus[take]
    stats=dict(candidate_faces=candidates,selected_pairs=len(plus),unique_plus=len(np.unique(plus,axis=0)),
               unique_minus=len(np.unique(minus,axis=0)),border=border,units='native_voxel',
               per_pair_foreground=1,per_pair_background=1)
    return PairSet(plus,minus,y.shape,'valid' if len(plus)>=minimum else 'insufficient_native_pairs',stats)


def teacher_pairs(logits,seed_parts=(),minimum=32):
    p=logits.detach().double().softmax(1)[0,1].cpu().numpy()
    if not np.isfinite(p).all():raise ValueError('Nonfinite Teacher probabilities')
    raw=adjacent_pairs(p>=.5,border=4,maximum=None,minimum=minimum)
    gaps=p[tuple(raw.plus.T)]-p[tuple(raw.minus.T)]
    good=np.flatnonzero(gaps>1e-3)
    before=len(good)
    if len(good)>2048:
        good=fixed_rng(142,'C0_teacher_pairs',*seed_parts).choice(good,2048,replace=False);good.sort()
    plus,minus=raw.plus[good],raw.minus[good]
    status='valid'
    if np.ptp(p)<1e-4:status='teacher_near_constant'
    elif len(plus)<minimum:status='insufficient_teacher_pairs'
    stats=dict(raw.stats,selected_pairs=len(plus),candidate_after_gap_filter=before,probability_span=float(np.ptp(p)),
               unique_plus=len(np.unique(plus,axis=0)),unique_minus=len(np.unique(minus,axis=0)),
               positive_probability_quantiles=np.quantile(p[tuple(plus.T)],[0,.5,1]).tolist() if len(plus) else [],
               negative_probability_quantiles=np.quantile(p[tuple(minus.T)],[0,.5,1]).tolist() if len(minus) else [],
               probability_gap_quantiles=np.quantile(gaps[good],[0,.5,1]).tolist() if len(good) else [],
               reliability_status='not_assessed_in_C0',coordinate_alignment='same_native_crop_no_transform',teacher_stop_gradient=True)
    return PairSet(plus,minus,p.shape,status,stats)


def map_pairs(pairset,shape,minimum=32,occupancy=None):
    shape=tuple(shape)
    if len(shape)!=3 or any(n<=0 for n in shape):raise ValueError('Invalid target grid')
    plus=pairset.plus*np.array(shape)//np.array(pairset.shape)
    minus=pairset.minus*np.array(shape)//np.array(pairset.shape)
    keep=np.any(plus!=minus,axis=1)
    plus,minus=plus[keep],minus[keep]
    _,indices=np.unique(np.concatenate([plus,minus],axis=1),axis=0,return_index=True)
    duplicate=len(plus)-len(indices);indices.sort();plus,minus=plus[indices],minus[indices]
    if occupancy is not None:
        q=np.asarray(occupancy)
        if q.shape!=shape:raise ValueError('Coarse occupancy geometry mismatch')
        pure=(q[tuple(plus.T)]>=.75)&(q[tuple(minus.T)]<=.25)
        plus,minus=plus[pure],minus[pure]
    count=len(plus)
    status='valid' if count>=minimum else 'unresolved_coarse_geometry'
    if count>=minimum and shape!=pairset.shape and occupancy is None:status='coarse_geometry_not_validated'
    return PairSet(plus,minus,shape,status,
                   dict(source_pairs=len(keep),collapsed_pairs=int((~keep).sum()),duplicate_pairs_removed=duplicate,retained=count,
                        coarse_occupancy_validated=occupancy is not None))


def flipped_coordinates(coords,shape,axes):
    result=np.array(coords,copy=True)
    for axis in axes:result[:,axis]=shape[axis]-1-result[:,axis]
    return result


def permuted_partners(pairset,seed_parts=()):
    order=fixed_rng(142,'C0_permuted_partners',*seed_parts).permutation(len(pairset.minus))
    return replace(pairset,minus=pairset.minus[order],stats=dict(pairset.stats,control='same_endpoint_marginals_permuted_partners',
                   spatial_geometry_not_matched=True,unchanged_partners=int(np.sum(order==np.arange(len(order))))))
