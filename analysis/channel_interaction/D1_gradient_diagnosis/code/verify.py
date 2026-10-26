"""Verify real evidence for both datasets and protect statistical validity."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import PROJECT,ROOT,SPECS,sha256,write_json
from diagnostic import SCALES,gradient_validity,signed_cosine

def verify(run):
    assert json.loads((run/'execution.json').read_text())['status']=='PASS'
    assert set(json.loads((run/'comparison.json').read_text()))==set(SPECS)
    sources=json.loads((run/'source_manifest.json').read_text())
    # Only acquisition dependencies were executed by run.py; later analysis has its own SHA.
    extraction_names={'diagnostic.py','run.py','common.py','core.py','nodes.py','objectives.py','vnet_ds.py'}
    for name,expected in sources.items():
        if Path(name).name not in extraction_names:continue
        assert sha256(PROJECT/name)==expected,(name,'extraction source changed')
    summary_source=json.loads((run/'summary_source.json').read_text())
    assert sha256(summary_source['path'])==summary_source['sha256']
    result={}
    for ds,spec in SPECS.items():
        root=run/ds;obs=json.loads((root/'observations.json').read_text())
        design=json.loads((root/'design.json').read_text())
        execution=json.loads((root/'execution.json').read_text())
        logs=json.loads((root/'hdf5_access_log.json').read_text())
        assert execution['status']=='PASS' and execution['optimizer_steps']==0 and execution['parameter_updates']==0
        assert all(row['key']=='image' for row in logs if row['role']=='unlabeled')
        assert set(row['case'] for row in logs if row['role']=='unlabeled')==set(design['unlabeled_ids'])
        assert design['patch']==spec['patch'] and len(design['labeled_ids'])==spec['labeled']
        assert len([o for o in obs if o['unit']=='case'])==(spec['labeled']+16)*2
        assert len([o for o in obs if o['unit']=='mini'])==24
        restores=json.loads((root/'restoration_checks.json').read_text())
        assert all(all(row[k] for k in ('parameters_unchanged_before_restore','state_restored','rng_restored','modes_restored','requires_grad_restored','grad_slots_unchanged')) for row in restores)
        for row in restores:
            for failed in row.get('failed_attempts',[]):
                assert 'gradient reconstruction' in failed['reason']
                assert all(failed['restoration'][k] for k in ('parameters_unchanged_before_restore','state_restored','rng_restored','grad_slots_unchanged'))
                assert row['accepted_precision']=='torch.float64'
            if row.get('mode','').startswith('train'):
                assert any(k.startswith('0:') for k in row['changed_buffers_before_restore'])
                assert any(k.startswith('1:') for k in row['changed_buffers_before_restore'])
            else:assert not row['changed_buffers_before_restore']
        for name,expected in json.loads((root/'data_file_stats.json').read_text()).items():
            stat=Path(name).stat();assert [stat.st_size,stat.st_mtime_ns]==expected
        assert sha256(design['checkpoint'])==design['checkpoint_sha256']
        max_raw=max_effective=max_scaling_error=0.;invalid=0
        for o in obs:
            with np.load(root/o['raw_file']) as data:
                assert set(data.files)=={'D1','E3'}
                for stage in data.files:
                    for h,head in enumerate(SCALES):
                        values=data[stage][h]
                        if stage=='D1' and head!='D1':assert np.isnan(values).all();continue
                        assert np.isfinite(values).all()
                        _,status,_=gradient_validity(values);invalid+=int(np.sum(status!='valid'))
                assert len(o['node_identity'])==5 and all(n['same_object'] and n['forward_calls']==1 for n in o['node_identity'])
                for c in o['checks']['gradient_reconstruction']:
                    max_raw=max(max_raw,c['raw_relative_l2_error']);max_effective=max(max_effective,c['normalized_relative_l2_error'])
                    assert c['raw_relative_l2_error']<1e-4 or c['raw_max_abs_error']<1e-9
                    assert c['normalized_relative_l2_error']<1e-4 or c['normalized_max_abs_error']<1e-9
        # At each observed L/U mini pair, positive LossNorm scalars preserve the raw direction.
        rows=list(csv.DictReader((root/'mini_relations.csv').open()))
        indexed={(r['mode'],r['batch'],r['repeat'],r['stage'],r['head'],r['channel'],r['policy']):r for r in rows}
        for key,r in indexed.items():
            if key[-1]!='raw':continue
            e=indexed[(*key[:-1],'lossnorm')]
            if r['valid']=='True' and e['valid']=='True':
                error=abs(float(r['cosine'])-float(e['cosine']));max_scaling_error=max(max_scaling_error,error)
                assert error<1e-6
        assert execution['train_replay_max_abs']=={'E3':0.0,'D1':0.0}
        result[ds]=dict(status='PASS',observations=len(obs),max_raw_relative_reconstruction=max_raw,
            max_lossnorm_relative_reconstruction=max_effective,invalid_channels=invalid,
            max_raw_lossnorm_mini_cosine_difference=max_scaling_error,
            model_BN_RNG_grad_restored=True,U_GT_reads=0,optimizer_steps=0)
    write_json(run/'verification.json',dict(status='PASS',datasets=result))
    write_json(run/'verification_source.json',dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();assert run.is_relative_to(ROOT/'D1_gradient_diagnosis/results')
    assert not (run/'verification.json').exists()
    print(verify(run))
