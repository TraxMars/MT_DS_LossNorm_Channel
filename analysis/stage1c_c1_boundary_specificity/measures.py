"""Validity-aware comparisons; no spatial feature alignment across cases."""
import numpy as np
from scipy.stats import spearmanr

def cosine_rows(a,b):
    a=np.asarray(a).reshape(len(a),-1);b=np.asarray(b).reshape(len(b),-1)
    an=np.sqrt(np.einsum('ij,ij->i',a,a,dtype=np.float64));bn=np.sqrt(np.einsum('ij,ij->i',b,b,dtype=np.float64))
    def valid(n):return np.isfinite(n)&(n>max(1e-12,1e-6*n.max(initial=0)))
    ok=valid(an)&valid(bn);cos=np.full(len(a),np.nan)
    cos[ok]=np.einsum('ij,ij->i',a[ok],b[ok],dtype=np.float64)/(an[ok]*bn[ok])
    cos[ok]=np.clip(cos[ok],-1,1)
    status=np.where(~np.isfinite(an)|~np.isfinite(bn),'nonfinite',np.where((an==0)|(bn==0),'zero_gradient',np.where(ok,'valid','below_threshold')))
    return cos,an,bn,status

def rho(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float);ok=np.isfinite(x)&np.isfinite(y)
    x,y=x[ok],y[ok]
    if len(x)<3:return None,'insufficient_channels'
    if np.ptp(x)==0 or np.ptp(y)==0:return None,'constant_response'
    return float(spearmanr(x,y).statistic),'valid'

def bootstrap(values,indices):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all() or not len(values):return dict(mean=None,ci=None,status='missing_case_statistic')
    draws=values[indices].mean(1)
    return dict(mean=float(values.mean()),ci=np.quantile(draws,[.025,.975]).tolist(),status='exploratory_case_bootstrap')
