"""Read-only completeness of both datasets, serialized state digests and source SHA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import BASELINE,PROJECT,ROOT,SPECS,checkpoint,clean,new_output,sha256,write_json
from data import case_path

def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def run(out):
    torch.set_num_threads(2);result={}
    for ds in SPECS:
        ck,manifest,path=checkpoint(ds)
        source=BASELINE/path.relative_to(PROJECT)
        originals=torch.load(source,map_location='cpu',weights_only=False)
        digests={role:state_digest(ck[role]) for role in ('student','teacher')}
        assert all(digests[role]==state_digest(originals[role]) for role in digests)
        assert digests['student']!=digests['teacher']
        splitpath=path.parent/'data_split_manifest.json'
        assert sha256(splitpath)==sha256(BASELINE/splitpath.relative_to(PROJECT))
        cfgpath=path.parent/'config.json'
        assert json.loads(cfgpath.read_text())==clean(ck['config'])
        files=[]
        for split,record in manifest['splits'].items():
            for case in record['ids']:
                p=case_path(ck['config']['dataset_root'],ds,case);stat=p.stat()
                assert p.is_file() and stat.st_size>0
                files.append(dict(split=split,case=case,path=str(p),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
        result[ds]=dict(status='PASS',state_sha256=digests,baseline_state_equal=True,
                       disk_config_equals_checkpoint=True,baseline_split_manifest_equal=True,
                       data_inventory=files,content_reads=0,GT_reads=0)
    assert result['LA']['state_sha256']['student']!=result['Pancreas']['state_sha256']['student']
    write_json(out/'supplement.json',dict(status='PASS',datasets=result))
    write_json(out/'source_sha256.json',{str(p.relative_to(PROJECT)):sha256(p) for p in sorted((ROOT/'D0_integrity/code').glob('*.py'))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    run(new_output('D0_integrity',args.output))
