"""Shared read-only inputs, serialization and output confinement for D stages."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASELINE = PROJECT.parent / 'MT_DS_LossNorm'
BOUNDARY = PROJECT.parent / 'MT_DS_LossNorm_Boundary'
sys.path.insert(0, str(PROJECT / 'analysis/stage1a_gradients'))
from core import sha256

SPECS = {
    'LA': dict(run='LA_8l', labeled=8, unlabeled=72, patch=[112,112,80]),
    'Pancreas': dict(run='Pancreas_6l', labeled=6, unlabeled=56, patch=[96,96,96]),
}

def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,np.bool_)): return value.item()
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, Path): return str(value)
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def new_output(stage, path):
    path=Path(path).resolve()
    parent=(ROOT/stage/'results').resolve()
    if path==parent or not path.is_relative_to(parent):
        raise ValueError(f'Output must be a new child of {parent}')
    path.mkdir(parents=True,exist_ok=False)
    return path

def checkpoint(dataset):
    spec=SPECS[dataset]
    path=PROJECT/'Results/seed_42'/spec['run']/'last_checkpoint.pth'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    cfg=ck['config']
    assert cfg['dataset']==dataset and cfg['labelnum']==spec['labeled']
    assert list(cfg['patch_size'])==spec['patch'] and ck['iteration']==15000
    assert cfg['consistency']==.3 and cfg['normalization_eps']==1e-8
    manifest=json.loads((path.parent/'data_split_manifest.json').read_text())
    assert manifest['dataset']==dataset
    left,right=manifest['labeled_ids'],manifest['unlabeled_ids']
    assert len(left)==spec['labeled'] and len(right)==spec['unlabeled']
    assert len(set(left+right))==len(left)+len(right)
    train=manifest['splits']['train']
    assert train['ids']==left+right and sha256(train['path'])==train['sha256']
    return ck,manifest,path
