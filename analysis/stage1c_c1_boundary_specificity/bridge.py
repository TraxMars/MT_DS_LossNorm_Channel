"""Read-only C0/1B imports; C1 outputs cannot enter historical directories."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
C0=PROJECT/'analysis/stage1c_boundary_attribution'
sys.path.insert(0,str(C0))
from support import (b,b_losses,STAGES,DOSES,write_json,write_csv,fixed_rng,
                     calibrate_epsilon,direction,dose_alpha,node_statistics)
from nodes import Capture,attenuate,NODE_SPECS
from pairs import PairSet,adjacent_pairs,permuted_partners
from objectives_c0 import Objectives,transition
from interventions import prediction_metrics,surface6
PRIMARY=('region','boundary','transition','permuted','far')

def output_path(path):
    p=Path(path).resolve()
    if p==ROOT/'runs' or not p.is_relative_to(ROOT/'runs'):
        raise ValueError('Only independent C1/runs children may be written')
    return p

def split():
    import json
    d=json.loads((b.ROOT/'runs/la10_paired/design.json').read_text())['data_split']
    return {s:d[s+'_L'] for s in ('discovery','probe')}
