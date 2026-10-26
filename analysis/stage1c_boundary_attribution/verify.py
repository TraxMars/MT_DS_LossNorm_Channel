"""Independent C0 artifact, scalar-product, numerical interval and history audit."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from support import ROOT,B_ROOT,STAGES,DOSES,b,selection,output_path,write_json


def read(path):return list(csv.DictReader(path.open()))


def verify(run):
    run=output_path(run);checks={}
    execution=json.loads((run/'execution_results.json').read_text())
    summary=json.loads((run/'summary_results.json').read_text())
    source=json.loads((run/'source_at_run.json').read_text())
    checks['executed_only_C0']=execution['phase']=='C0' and not execution['C1_C2_C3_executed'] and execution['optimizer_steps']==execution['checkpoint_writes']==0
    checks['frozen_execution_invariants']=execution['status']=='PASS' and all(execution['invariants'].values())
    checks['predeclared_design_unchanged_since_execution']=b.sha256(ROOT/'DESIGN.md')==source['design_sha256']
    checks['executed_source_files_unchanged']=all(b.sha256(ROOT/p)==h for p,h in source['sources'].items()) and b.sha256(ROOT.parents[1]/'tests/test_stage1c_c0.py')==source['tests_sha256']
    checks['unit_and_historical_regressions_pass']=all(json.loads((ROOT/name).read_text())['status']=='PASS' for name in ('TEST_RESULTS.json','REGRESSION_RESULTS.json'))
    geometry=json.loads((run/'geometry.json').read_text());nodes=json.loads((run/'node_identity.json').read_text())
    checks['eight_native_pair_sets_valid']=len(geometry)==8 and all(r['status']=='valid' and r['stats']['selected_pairs']==2048 for r in geometry)
    checks['40_live_node_identities']=len(nodes)==40 and all(r['same_object'] and r['forward_calls']==1 and r['requires_grad'] for r in nodes)
    checks['decoder_skip_fusion_exact']=all(r.get('skip_fusion_max_error',0)==0 for r in nodes)
    selected=selection();checks['only_predeclared_discovery_cases']=all(r['case'] in selected[r['domain']] for r in geometry)
    checks['two_original_crops_per_case']=all(sorted(r['repeat'] for r in geometry if r['case']==case)==[0,1] for cases in selected.values() for case in cases)
    observations=json.loads((b.A_RUN/'observations.json').read_text());a_design=json.loads((b.A_RUN/'design.json').read_text())
    access=[];truths={}
    for case in selected['L']:
        obs=[o for o in observations['L'] if o['case']==case]
        _,_,crops=b.load_case_and_crops('L',case,a_design,obs,access)
        for o,x,y,meta in crops:truths[o['index']]=y[0].numpy()
    pair_tests=[]
    for r in geometry:
        p=np.load(run/'raw'/f'{r["domain"]}_{r["index"]}_pairs.npz');a,c=p['plus'],p['minus']
        okay=(a.shape==c.shape==(2048,3) and (np.abs(a-c).sum(1)==1).all() and
              (a>=4).all() and (c>=4).all() and (a<np.array(r['shape'])-4).all() and (c<np.array(r['shape'])-4).all())
        if r['domain']=='L':okay=okay and truths[r['index']][tuple(a.T)].all() and not truths[r['index']][tuple(c.T)].any()
        else:okay=okay and r['stats']['positive_probability_quantiles'][0]>=.5 and r['stats']['negative_probability_quantiles'][-1]<.5 and r['stats']['probability_gap_quantiles'][0]>1e-3 and r['stats']['reliability_status']=='not_assessed_in_C0'
        pair_tests.append(bool(okay))
    checks['native_pair_coordinates_and_class_direction']=all(pair_tests)
    write_json(run/'verification_hdf5_access_log.json',access)
    deltas=read(run/'deltas.csv');metrics=read(run/'intervention_metrics.csv');features=read(run/'feature_gradient_stats.csv');params=read(run/'parameter_gradient_stats.csv')
    checks['all_three_fixed_doses_only']=set(float(r['dose']) for r in deltas)==set(DOSES)
    checks['1440_loss_differences_360_interventions']=len(deltas)==1440 and len(metrics)==360
    checks['actual_energy_matches_budget']=all(np.isclose(float(r['actual_l2']),float(r['target_l2']),rtol=2e-5,atol=1e-5) and float(r['alpha'])<=.5 for r in metrics if r['status']=='valid')
    checks['actual_subtraction_energy_and_roundoff_reported']=all(
        np.isfinite(float(r['subtraction_roundoff_l2'])) and float(r['subtraction_roundoff_l2'])>=0 and
        abs(float(r['actual_l2'])-float(r['intended_l2']))<=float(r['subtraction_roundoff_l2'])*(1+1e-6)+1e-12
        for r in metrics if r['status']=='valid')
    checks['unmatched_not_fake_zero']=all(not r['delta_actual'] and r['actual_direction']=='undefined' for r in deltas if r['status']!='valid')
    checks['feature_and_parameter_coordinates_separate']=all(r['coordinate_kind']=='feature_node' for r in features) and all(r['coordinate_kind']=='convolution_parameter' for r in params)
    checks['all_recorded_gradient_stats_finite']=all(np.isfinite(float(r['gradient_norm'])) for r in features+params)
    noise=json.loads((run/'noise_calibration.json').read_text());nlookup={};formula=[]
    for r in noise:
        expected=max(1e-9,1e-6*max(abs(r['base']),1e-3),5*max(abs(x-r['base']) for x in r['null_samples']))
        formula.append(r['epsilon']==expected and r['null_count']==20)
        nlookup[(r['domain'],r['index'],r['objective'])]=expected
    checks['uncertainty_calibrated_by_fixed_null_only_formula']=len(noise)==32 and all(formula)
    interval=[];residual=[]
    for r in deltas:
        eps=nlookup[(r['domain'],int(r['index']),r['objective'])]
        interval.append(float(r['epsilon'])==eps)
        if r['status']!='valid':continue
        actual,pred=float(r['delta_actual']),float(r['delta_hat'])
        for value,key in [(actual,'actual_direction'),(pred,'predicted_direction')]:
            expected='indeterminate' if abs(value)<=eps else 'positive' if value>0 else 'negative'
            interval.append(r[key]==expected)
        residual.append(np.isclose(float(r['absolute_residual']),abs(actual-pred),atol=1e-15,rtol=1e-12))
    checks['near_noise_directions_never_forced']=all(interval)
    checks['residuals_recomputed']=all(residual)
    # Independent numpy reduction of every archived primary feature gradient.
    groups={}
    for r in deltas:
        if r['objective']==('transition_m1' if r['domain']=='L' else 'u_transition'):
            groups.setdefault((r['domain'],int(r['index']),r['stage']),[]).append(r)
    discrepancies=[];shape_tests=[];finite=[]
    for (d,index,stage),rows in groups.items():
        data=np.load(run/'raw/features'/f'{d}_{index}_{stage}.npz')
        f,g,m=data['activation'].astype(float),data['gradient'].astype(float),data['mask'].astype(float)
        shape_tests.append(f.shape==g.shape and f.ndim==5 and m.shape==(1,1,*f.shape[2:]))
        finite.append(np.isfinite(f).all() and np.isfinite(g).all() and np.isfinite(m).all())
        inner=(g*f*m).sum((0,2,3,4));amplitude=np.abs(g*f*m).sum((0,2,3,4))
        for r in rows:
            if r['status']!='valid':continue
            expected=-float(r['alpha'])*inner[int(r['channel'])]
            discrepancies.append(abs(expected-float(r['delta_hat'])))
            assert np.isclose(expected,float(r['delta_hat']),atol=1e-12,rtol=1e-9)
            assert np.isclose(float(r['absolute_gxa_sum']),float(r['alpha'])*amplitude[int(r['channel'])],atol=1e-12,rtol=1e-9)
        del f,g,m,data
    checks['40_primary_raw_feature_fields_valid']=len(groups)==40 and all(shape_tests) and all(finite)
    checks['all_360_primary_predictions_independently_recomputed']=len(discrepancies)==360 and max(discrepancies)<1e-10
    ptests=[];pnorms=[]
    pgroup={}
    for r in params:pgroup.setdefault((r['domain'],int(r['index']),r['objective'],r['stage']),[]).append(r)
    for (d,i,obj,s),rows in pgroup.items():
        a=np.load(run/'raw/parameters'/f'{d}_{i}_{obj}_{s}.npy',mmap_mode='r')
        shape=json.loads(rows[0]['shape']);channels=len(rows)
        ptests.append(a.shape==(channels,np.prod(shape)//channels) and np.isfinite(a).all())
        norms=np.linalg.norm(a.astype(float),axis=1)
        pnorms.extend(np.isclose(norms[int(r['channel'])],float(r['gradient_norm']),rtol=1e-10,atol=1e-14) for r in rows)
    checks['160_raw_parameter_fields_separate_and_norms_match']=len(pgroup)==160 and all(ptests) and all(pnorms)
    replay=json.loads((run/'reference_replay.json').read_text())
    checks['80_A_B_final_head_parameter_vectors_replayed']=len(replay)==80 and all(r['absolute_l2_error']<=1e-9 or r['relative_l2_error']<=1e-4 for r in replay)
    log=json.loads((run/'hdf5_access_log.json').read_text())
    checks['unlabeled_HDF5_image_key_only']=all(r['key']=='image' for r in log if r['role']=='unlabeled')
    checks['fidelity_predeclared_gates_pass']=summary['status']=='PASS'
    protection=b.verify_protection(json.loads((ROOT/'PROTECTION_BEFORE.json').read_text()))
    checks['711_historical_SHA256_unchanged']=protection['status']=='PASS' and protection['checked_files']==711
    checks['no_baseline_tracked_changes']=subprocess.run(['git','diff','--quiet','07a0b90','--','code','Results','summary','analysis/stage_features','analysis/stage1a_gradients','analysis/stage1b_boundary'],cwd=ROOT.parents[1]).returncode==0
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',phase='C0',checks=checks,protection=protection,
                independent_feature_prediction_max_absolute_error=max(discrepancies),
                maximum_AB_parameter_relative_error=max(r['relative_l2_error'] for r in replay),
                uncertainty_rule='Fixed before interventions; no thresholds adjusted based on deltas',
                deferred=['C1 boundary specificity/matched difficulty study','C2 Teacher enhancement reliability','C3 channel groups','formal training'])
    write_json(run/'verification_results.json',result)
    assert result['status']=='PASS',checks
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(json.dumps(b.clean(verify(a.run)),indent=2))
