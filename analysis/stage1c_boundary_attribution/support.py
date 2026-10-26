"""Read-only Stage1A/B bridge and C0-only output guards."""
from pathlib import Path
import sys
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
B_ROOT=PROJECT/'analysis/stage1b_boundary'
sys.path.insert(0,str(B_ROOT))
import common as b
import losses as b_losses
from common import write_json,write_csv,fixed_rng

STAGES=('E3','E4','D3','D2','D1')
DOSES=(.01,.03,.10)
MARGINS=(.5,1.,2.)
L_IDS=('06SR5RBREL16DQ6M8LWS','1D7CUD1955YZPGK8XHJX')
U_IDS=('3C2QTUNI0852XV7ZH4Q1','57SGAJMLCTCH92QUA0EE')


def output_path(path):
    p=Path(path).resolve()
    if p==ROOT/'runs' or not p.is_relative_to(ROOT/'runs'):
        raise ValueError('C0 output must stay within its independent runs directory')
    return p


def selection():
    design=json.loads((B_ROOT/'runs/la10_paired/design.json').read_text())
    assert list(L_IDS)==design['data_split']['discovery_L'][:2]
    assert list(U_IDS)==design['data_split']['discovery_U'][:2]
    return {'L':L_IDS,'U':U_IDS}


def selected_channels(channels):
    if channels<3:raise ValueError('At least three channels required for fixed smoke positions')
    return (0,channels//2,channels-1)


def calibrate_epsilon(base,null_values):
    null=np.asarray(null_values,dtype=float)
    if not np.isfinite(base) or not np.isfinite(null).all() or null.size==0:
        raise ValueError('Finite null observations are required')
    observed=float(np.max(np.abs(null-base)))
    floor=max(1e-9,1e-6*max(abs(float(base)),1e-3))
    return max(floor,5*observed),dict(base=float(base),null_max_difference=observed,fixed_floor=floor,null_count=null.size)


def direction(value,epsilon):
    if value is None or not np.isfinite(value):return 'undefined'
    if abs(value)<=epsilon:return 'indeterminate'
    return 'positive' if value>0 else 'negative'


def dose_alpha(energy,budget):
    if not np.isfinite(energy) or energy<=1e-12:return None,'zero_or_nonfinite_activation'
    if not np.isfinite(budget) or budget<=0:return None,'invalid_budget'
    alpha=budget/energy
    if alpha>.5:return None,'alpha_cap_exceeded'
    return alpha,'valid'


def node_statistics(gradient,feature,mask):
    if gradient.shape!=feature.shape:raise ValueError('Feature gradient shape mismatch')
    g,f,m=gradient.detach().double(),feature.detach().double(),mask.detach().double()
    product=g*f*m
    dimensions=(0,2,3,4)
    return dict(gradient_norm=g.square().sum(dimensions).sqrt().cpu().numpy(),
                masked_gradient_norm=(g*m).square().sum(dimensions).sqrt().cpu().numpy(),
                activation_energy=(f*m).square().sum(dimensions).sqrt().cpu().numpy(),
                signed_inner=product.sum(dimensions).cpu().numpy(),
                absolute_product_sum=product.abs().sum(dimensions).cpu().numpy())
