"""Independent artifact checks after frozen diagnostics (no data/model forward)."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from common import require_stage1b_run,ROOT,A_RUN,SCALES,verify_protection,write_json


def verify(run,ablation):
    run,ablation=require_stage1b_run(run),require_stage1b_run(ablation)
    design=json.loads((run/'design.json').read_text());mapping=json.loads((run/'parameter_mapping.json').read_text())
    coverage=json.loads((run/'loss_coverage.json').read_text());checks={}
    observations=json.loads((A_RUN/'observations.json').read_text())
    checks['checkpoint_and_cases_match_A']=(design['checkpoint_sha256']==json.loads((A_RUN/'design.json').read_text())['checkpoint_sha256'] and
        design['labeled_ids']==list(dict.fromkeys(o['case'] for o in observations['L'])) and
        design['unlabeled_ids']==list(dict.fromkeys(o['case'] for o in observations['U'])))
    checks['parameter_mapping_exact_A']=mapping==json.loads((A_RUN/'parameter_mapping.json').read_text())
    keys=set();control_checks=[];mass_checks=[]
    for r in coverage:
        keys.add((r['domain'],r['index'],r['objective'],r['scale']))
        if r['domain']=='L' and r['status']=='valid':
            stats=r['stats']
            for side in ('inner','outer'):
                mass_checks.append(abs(stats[side+'_reconstructed_native_mass']-stats['native_'+side])<1e-3)
                control_checks.append(stats[side+'_effective_cells']>=2)
    checks['1088_unique_objective_scale_records']=len(keys)==len(coverage)==1088
    checks['native_class_region_mass_conserved']=all(mass_checks)
    checks['valid_terms_have_effective_coverage']=all(control_checks)
    lookup={(r['domain'],r['index'],r['objective'],r['scale']):r for r in coverage}
    paired_counts=[]
    for index in range(16):
        for width in (1,2,3):
            b=lookup[('L',index,f'band_w{width}','D1')]
            for name in ('random','core'):
                r=lookup[('L',index,f'{name}_w{width}','D1')]
                if r['status']=='valid':paired_counts.append(all(b['stats'][k]==r['stats'][k] for k in ('native_inner','native_outer')))
    checks['controls_exact_native_voxel_and_class_counts']=all(paired_counts)
    reachability,nan_masks,shapes=[],[],[]
    finite_elements=0
    for d,objs,n in [('L',design['L_objectives'],16),('U',design['U_objectives'],32)]:
        for obj in objs:
            for stage,spec in mapping.items():
                a=np.load(run/'raw'/f'{d}_{obj}_{stage}.npy',mmap_mode='r')
                expected_kernel=np.prod(spec['weight_shape'])//spec['channels']
                shapes.append(a.shape==(n,4,spec['channels'],expected_kernel))
                for i in range(n):
                    for s,scale in enumerate(SCALES):
                        valid=lookup[(d,i,obj,scale)]['status']=='valid'
                        reached=scale in spec['reachable_scales']
                        block=a[i,s]
                        if reached and valid:reachability.append(np.isfinite(block).all());finite_elements+=block.size
                        else:nan_masks.append(np.isnan(block).all())
    checks['raw_parameter_gradient_shapes_correct']=all(shapes)
    checks['reachable_valid_gradients_finite']=all(reachability)
    checks['unreachable_or_invalid_nan_not_fake_zero']=all(nan_masks)
    # Recompute a same-observation parameter cosine without analysis helpers.
    selected=[]
    for stage in ('E3','D1'):
        a=np.load(run/'raw'/f'L_band_w2_{stage}.npy',mmap_mode='r')[0,3].astype(float)
        b=np.load(A_RUN/'raw'/f'L_{stage}.npy',mmap_mode='r')[0,3].astype(float)
        direct=(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        selected.append(dict(stage=stage,boundary_vs_seg_same_observation_median=float(np.median(direct))))
    matrix=list(csv.DictReader((run/'method_scale_summary.csv').open()))
    checks['612_method_stage_head_cells']=len(matrix)==17*9*4
    checks['matrix_graph_NA_matches_real_reachability']=all(
        (r['graph_reachable']=='True')==(r['scale'] in mapping[r['stage']]['reachable_scales']) and
        (r['graph_reachable']=='True' or not r.get('L_U_cosine_median')) for r in matrix)
    recomputed=[]
    for stage in ('E3','D1'):
        for method in ('A','B3_w2'):
            gradients=[]
            for domain,objective,n in [('L','band_w2',8),('U','edge_w2',16)]:
                path=(A_RUN/'raw'/f'{domain}_{stage}.npy') if method=='A' else (run/'raw'/f'{domain}_{objective}_{stage}.npy')
                raw=np.load(path,mmap_mode='r')
                gradients.append(np.asarray(raw[:,3],dtype=float).reshape(n,2,*raw.shape[2:]).mean(1))
            a,b=gradients;na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1)
            va=na>np.maximum(design['gradient_absolute_floor'],design['gradient_relative_floor']*na.max(-1,keepdims=True))
            vb=nb>np.maximum(design['gradient_absolute_floor'],design['gradient_relative_floor']*nb.max(-1,keepdims=True))
            good=va.T[:,:,None]&vb.T[:,None,:]
            cosine=np.einsum('ick,jck->cij',a,b)/(na.T[:,:,None]*nb.T[:,None,:])
            cosine[~good]=np.nan
            mean=np.nanmean(cosine,axis=(1,2));independent=float(np.median(mean[good.mean((1,2))>=.75]))
            saved=float(next(r for r in matrix if r['stage']==stage and r['scale']=='D1' and r['method']==method)['L_U_cosine_median'])
            recomputed.append(dict(stage=stage,method=method,saved=saved,independent=independent,match=bool(np.isclose(saved,independent,atol=1e-12,rtol=1e-12))))
    checks['independently_recomputed_LU_cells_match']=all(r['match'] for r in recomputed)
    for dirname,key in [(run,'extraction_results.json'),(ablation,'ablation_results.json')]:
        r=json.loads((dirname/key).read_text());checks[key+'_pass']=r['status']=='PASS'
        checks[key+'_no_updates']=r['optimizer_steps']==0 and r['checkpoint_writes']==0
    access=json.loads((run/'hdf5_access_log.json').read_text())
    checks['U_image_only']=all(r['key']=='image' for r in access if r['role']=='unlabeled')
    group=list(csv.DictReader((ablation/'groups.csv').open()));single=list(csv.DictReader((ablation/'singletons.csv').open()))
    good=[r for r in group if r['status']=='valid']
    checks['joint_K_exactly2']=all(len(json.loads(r['channels'].replace("'",'"')))==2 for r in group)
    # csv stores list repr, JSON-compatible for integer/float lists.
    checks['group_l2_matched']=all(np.isclose(float(r['target_l2']),float(r['actual_l2']),rtol=2e-5,atol=1e-5) for r in good)
    checks['component_l2_conserved']=all(np.isclose(sum(z*z for z in json.loads(r['single_component_l2'])),float(r['target_l2'])**2,rtol=5e-5,atol=1e-5) for r in good)
    checks['singleton_l2_matched']=all(np.isclose(float(r['target_l2']),float(r['actual_l2']),rtol=2e-5,atol=1e-5) for r in single if r['status']=='valid')
    checks['alpha_cap_respected']=all(float(r['alpha'])<=.5 for r in single+good if r['status']=='valid')
    checks['probe_disjoint_from_discovery']=not(set(design['data_split']['probe_L'])&set(design['data_split']['discovery_L']))
    checks['only_probe_cases_ablated']=set(r['case'] for r in group)<=set(design['data_split']['probe_L'])
    checks['each_detailed_stage_has_final_prediction_effect']=all(any(float(r['changed_prediction_fraction'])>0 for r in good if r['stage']==stage) for stage in ('E3','E4','D3','D2','D1'))
    statuses=Counter()
    with (run/'observation_norms.csv').open() as f:
        for r in csv.DictReader(f):
            statuses[r['status']]+=1
            if r['status']=='valid':
                if not float(r['raw_norm'])>float(r['validity_floor']):raise AssertionError('Invalid reliable norm')
    protection=verify_protection(json.loads((ROOT/'PROTECTION_BEFORE.json').read_text()))
    checks['historical_files_sha256_unchanged']=protection['status']=='PASS'
    checks['unit_tests_pass']=json.loads((ROOT/'TEST_RESULTS.json').read_text())['status']=='PASS'
    checks['stage0_stage1a_core_regressions_pass']=all(json.loads((ROOT/name).read_text())['status']=='PASS' for name in ('STAGE0_REGRESSION.json','STAGE1A_REGRESSION.json','CORE_REGRESSION.json'))
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,protection=protection,
                gradient_observation_statuses=dict(statuses),finite_raw_gradient_elements=finite_elements,
                paired_cosine_independent_spotchecks=selected,
                independently_recomputed_LU_cells=recomputed,
                limitations=['HDF5 size/mtime checked by extraction; contents not hashed/read for protection.',
                             'U ground truth is never opened; no U boundary accuracy claim.',
                             'No feature-map gradients generated: all saved gradient vectors are convolution parameter coordinates.'])
    write_json(run/'verification_results.json',report)
    assert report['status']=='PASS',checks
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--ablation',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.run.resolve(),a.ablation.resolve()),indent=2))
