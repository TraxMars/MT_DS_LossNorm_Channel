"""Read-only SHA audit of baseline, copied history, checkpoints and split lists."""
import argparse
import difflib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import BASELINE,BOUNDARY,PROJECT,ROOT,SPECS,checkpoint,new_output,sha256,write_json
from networks import VNetDS

def protected_paths():
    for root in (PROJECT,BASELINE,BOUNDARY):
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(s in p.parts for s in ('.git','__pycache__','.pytest_cache')): continue
            if root==PROJECT and p.is_relative_to(ROOT): continue
            if p.suffix in ('.h5','.hdf5'): continue
            yield p
    for ds in SPECS:
        yield from sorted((PROJECT.parent/'Dataset'/ds).glob('*.list'))

def snapshot():
    result={}
    for p in protected_paths():
        result[str(p)]={'sha256':sha256(p),'bytes':p.stat().st_size}
    return result

def verify(before):
    changes=[]
    for name,expected in before.items():
        p=Path(name)
        if not p.is_file() or p.stat().st_size!=expected['bytes'] or sha256(p)!=expected['sha256']:
            changes.append(name)
    additions=sorted(set(map(str,protected_paths()))-set(before))
    return dict(status='PASS' if not changes and not additions else 'FAIL',
                checked_files=len(before),changed_or_missing=changes,new_historical_files=additions)

def audit(out):
    torch.set_num_threads(4)
    started=time.monotonic()
    before=snapshot()
    write_json(out/'protection_before.json',before)
    print(f'Protected {len(before)} historical files',flush=True)
    comparisons=[]
    for rel in ('code/data.py','code/objectives.py','code/train.py','code/evaluate.py','code/networks/vnet_ds.py'):
        a,b=BASELINE/rel,PROJECT/rel
        same=sha256(a)==sha256(b)
        comparisons.append(dict(path=rel,baseline_sha256=sha256(a),copy_sha256=sha256(b),equal=same))
        if not same:
            (out/(Path(rel).name+'.diff')).write_text(''.join(difflib.unified_diff(
                a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile=str(a),tofile=str(b))))
    # Historical A/C0 are copied verbatim; compare every available file, including raw arrays.
    historical=[]
    for name in ('stage1a_gradients','stage1c_boundary_attribution'):
        for p in sorted((PROJECT/'analysis'/name).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts: continue
            other=BOUNDARY/p.relative_to(PROJECT)
            equal=other.is_file() and before[str(p)]['sha256']==before[str(other)]['sha256']
            historical.append(dict(path=str(p.relative_to(PROJECT)),equal=equal))
    write_json(out/'history_copy_comparison.json',historical)
    datasets={}
    for ds,spec in SPECS.items():
        ck,manifest,path=checkpoint(ds)
        model=VNetDS(n_channels=1,n_classes=2,normalization='batchnorm',has_dropout=True)
        roles={}
        for role in ('student','teacher'):
            state=ck[role]
            model.load_state_dict(state,strict=True)
            assert all(torch.isfinite(t).all().item() for t in state.values())
            roles[role]=dict(keys=len(state),elements=sum(t.numel() for t in state.values()),
                             finite=True,strict_load=True,shapes={k:list(v.shape) for k,v in state.items()})
        source=BASELINE/path.relative_to(PROJECT)
        assert sha256(path)==sha256(source)
        lists=[]
        for split,record in manifest['splits'].items():
            ids=Path(record['path']).read_text().split()
            assert ids==record['ids'] and sha256(record['path'])==record['sha256']
            if split!='train': assert not set(ids)&set(manifest['splits']['train']['ids'])
            lists.append(dict(split=split,path=record['path'],sha256=record['sha256'],cases=len(ids)))
        datasets[ds]=dict(status='PASS',checkpoint=str(path),sha256=sha256(path),baseline_equal=True,
                          iteration=ck['iteration'],pairing='same serialized checkpoint',roles=roles,
                          patch=spec['patch'],labeled_ids=manifest['labeled_ids'],unlabeled_ids=manifest['unlabeled_ids'],
                          splits=lists,student_teacher_different=any(not torch.equal(ck['student'][k],ck['teacher'][k]) for k in ck['student']))
        print(f'{ds} checkpoint and splits PASS',flush=True)
    assert all(r['equal'] for r in historical)
    assert all(r['equal'] for r in comparisons if r['path']!='code/networks/vnet_ds.py')
    after=verify(before)
    assert after['status']=='PASS'
    result=dict(status='PASS',datasets=datasets,source_comparison=comparisons,
                history_copy_files=len(historical),history_copy_equal=True,protection=after,
                original_loader_reads_U_label=True,new_loader_policy='U image-only; no HDF5 whole-file hashing',
                cuda_available=torch.cuda.is_available(),torch=torch.__version__,elapsed_seconds=time.monotonic()-started,
                preexisting_worktree=subprocess.check_output(['git','status','--short'],cwd=PROJECT,text=True))
    write_json(out/'integrity.json',result)
    write_json(out/'protection_after.json',after)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--verify',type=Path)
    args=p.parse_args();out=new_output('D0_integrity',args.output)
    if args.verify:
        r=verify(json.loads(args.verify.read_text()));write_json(out/'protection_after.json',r)
        assert r['status']=='PASS';print(r)
    else: audit(out)
