"""Endpoint-preserving permutations and explicit, imperfect far controls."""
import numpy as np
from scipy import ndimage
from scipy.stats import wasserstein_distance
from bridge import PairSet,fixed_rng

def describe(a):
    a=np.asarray(a,dtype=float)
    return dict(n=a.size,mean=float(a.mean()),std=float(a.std()),q=np.quantile(a,[0,.1,.5,.9,1]).tolist()) if a.size else dict(n=0)

def values(z,p):return z[tuple(p.plus.T)],z[tuple(p.minus.T)]

def distributions(z,fg,bg):
    result={}
    for name,mask in [('fg',fg),('bg',bg)]:
        v=z[mask];ce=np.logaddexp(0,-v if name=='fg' else v)
        prob=1/(1+np.exp(-np.clip(v,-700,700)))
        entropy=np.logaddexp(0,v)-prob*v
        result[name]=dict(logit=describe(v),ce=describe(ce),entropy=describe(entropy),error=float((v<=0).mean() if name=='fg' else (v>0).mean()))
    return result

def compare_endpoints(z,p,q):
    result={}
    for side,a,c in zip(('fg','bg'),values(z,p),values(z,q)):
        sign=-1 if side=='fg' else 1
        result[side]=dict(logit_wasserstein=float(wasserstein_distance(a,c)),ce_wasserstein=float(wasserstein_distance(np.logaddexp(0,sign*a),np.logaddexp(0,sign*c))),
                          logit_mean_difference=float(c.mean()-a.mean()),absolute_error_gap=float(np.mean(sign*c>0)-np.mean(sign*a>0)))
    return result

def subset(p,take,**stats):
    take=np.asarray(take,dtype=int)
    return PairSet(p.plus[take],p.minus[take],p.shape,'valid' if len(take)>=32 else 'insufficient_pairs',dict(p.stats,**stats,selected_pairs=len(take)))

def far_controls(p,z,y,di,do,valid,seed):
    interior=np.zeros(y.shape,bool);interior[4:-4,4:-4,4:-4]=True
    pools=[np.argwhere(valid&interior&y.astype(bool)&(di>2)),np.argwhere(valid&interior&~y.astype(bool)&(do>2))]
    n=len(p.plus);rng=fixed_rng(142,'C1_far',*seed)
    if min(map(len,pools))<n:
        invalid=PairSet(np.empty((0,3),int),np.empty((0,3),int),p.shape,'insufficient_far_pool',{})
        return invalid,invalid,invalid,dict(status='insufficient_far_pool',available=list(map(len,pools)))
    full=PairSet(*[pool[rng.choice(len(pool),n,replace=False)] for pool in pools],p.shape,'valid',dict(control='far_unmatched_difficulty',selected_pairs=n))
    # Fixed width 0.5 bins imply absolute z discrepancy <0.5, same class,
    # unique coordinates. Target endpoint multiplicity remains explicit.
    mapped=[]
    for side,pool in enumerate(pools):
        v=z[tuple(pool.T)];bins=np.floor(v/.5).astype(int);groups={}
        for k in np.unique(bins):
            ids=np.flatnonzero(bins==k);groups[int(k)]=rng.permutation(ids).tolist()
        target=(p.plus,p.minus)[side];match=np.full((n,3),-1,int)
        for i in rng.permutation(n):
            k=int(np.floor(z[tuple(target[i])]/.5));g=groups.get(k,[])
            if g:match[i]=pool[g.pop()]
        mapped.append(match)
    keep=np.flatnonzero((mapped[0][:,0]>=0)&(mapped[1][:,0]>=0))
    true_common=subset(p,keep,control='true_on_far_common_support')
    matched=PairSet(mapped[0][keep],mapped[1][keep],p.shape,'valid' if len(keep)>=32 else 'insufficient_pairs',dict(control='far_z_bin_matched',selected_pairs=len(keep)))
    info=dict(status=matched.status,available=list(map(len,pools)),requested=n,matched=len(keep),retained_fraction=len(keep)/n,unmatched_fraction=1-len(keep)/n,
              bin_width=.5,logit_caliper=.5,without_replacement=True,full_control_difficulty=compare_endpoints(z,p,full))
    if len(keep):info['matched_control_difficulty']=compare_endpoints(z,true_common,matched)
    return full,matched,true_common,info

def pair_statistics(p,z,pred_distance=None):
    if not len(p.plus):return dict(status=p.status,count=0)
    a,c=values(z,p);distance=np.linalg.norm(p.plus-p.minus,axis=1);gap=a-c
    unique_a=len(np.unique(p.plus,axis=0));unique_c=len(np.unique(p.minus,axis=0))
    result=dict(status=p.status,count=len(a),distance=describe(distance),unique_fg=unique_a,unique_bg=unique_c,
                fg_reuse_fraction=1-unique_a/len(a),bg_reuse_fraction=1-unique_c/len(a),fg_logit=describe(a),bg_logit=describe(c),
                fg_absolute_logit=describe(abs(a)),bg_absolute_logit=describe(abs(c)),logit_gap=describe(gap),groups=[],margins=[])
    for fg_ok,bg_ok in ((True,True),(True,False),(False,True),(False,False)):
        keep=((a>0)==fg_ok)&((c<=0)==bg_ok)
        row=dict(fg_correct=fg_ok,bg_correct=bg_ok,count=int(keep.sum()),fraction=float(keep.mean()))
        if keep.any():
            row.update(fg_logit=describe(a[keep]),bg_logit=describe(c[keep]),absolute_fg_logit=describe(abs(a[keep])),absolute_bg_logit=describe(abs(c[keep])),gap=describe(gap[keep]),
                       transition=float(np.logaddexp(0,1-gap[keep]).mean()),endpoint_boundary_ce=float((.5*np.logaddexp(0,-a[keep])+.5*np.logaddexp(0,c[keep])).mean()))
            if pred_distance is not None:row['gt_inner_to_predicted_contour_distance']=describe(pred_distance[tuple(p.plus[keep].T)])
        result['groups'].append(row)
    for m in (.5,1.,2.):
        satisfied=gap>=m;wrong=(a<=0)|(c>0)
        result['margins'].append(dict(margin=m,satisfied_fraction=float(satisfied.mean()),satisfied_and_wrong_fraction=float((satisfied&wrong).mean()),
            wrong_fraction_given_satisfied=float(wrong[satisfied].mean()) if satisfied.any() else None,loss=float(np.logaddexp(0,m-gap).mean())))
    return result
