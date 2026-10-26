"""C0 local numerical fidelity only; no channel selection or boundary claims."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from support import ROOT,DOSES,output_path,write_json,write_csv


def correlation(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float);good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<3 or np.std(x[good])==0 or np.std(y[good])==0:return None
    return float(spearmanr(x[good],y[good]).statistic)


def number(row,key):return float(row[key]) if row.get(key) else None


def summaries(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['domain'],r['objective'],r['stage'],float(r['dose']))].append(r)
    output=[]
    for (domain,obj,stage,dose),all_rows in groups.items():
        valid=[r for r in all_rows if r['status']=='valid']
        actual=[number(r,'delta_actual') for r in valid];pred=[number(r,'delta_hat') for r in valid]
        defined=[r for r in valid if r['actual_direction'] in ('positive','negative')]
        correct=[r['actual_direction']==r['predicted_direction'] for r in defined]
        bysign={sign:[r for r in defined if r['actual_direction']==sign] for sign in ('positive','negative')}
        balanced=None
        if all(bysign.values()):balanced=float(np.mean([np.mean([r['predicted_direction']==sign for r in v]) for sign,v in bysign.items()]))
        output.append(dict(domain=domain,objective=obj,stage=stage,dose=dose,total=len(all_rows),valid=len(valid),invalid=len(all_rows)-len(valid),
             actual_indeterminate=sum(r['actual_direction']=='indeterminate' for r in valid),
             actual_indeterminate_fraction=sum(r['actual_direction']=='indeterminate' for r in valid)/len(valid) if valid else None,
             predicted_indeterminate=sum(r['predicted_direction']=='indeterminate' for r in valid),
             resolved_actual=len(defined),sign_accuracy=float(np.mean(correct)) if correct else None,balanced_sign_accuracy=balanced,
             majority_direction_baseline=max(map(len,bysign.values()))/len(defined) if defined else None,
             spearman=correlation(pred,actual),absolute_net_strength_spearman=correlation(np.abs(pred),np.abs(actual)),
             absolute_gxa_sum_strength_spearman=correlation([number(r,'absolute_gxa_sum') for r in valid],np.abs(actual)),
             median_absolute_residual=float(np.median([number(r,'absolute_residual') for r in valid])) if valid else None,
             median_relative_residual=float(np.median([number(r,'relative_residual') for r in valid])) if valid else None,
             median_epsilon=float(np.median([number(r,'epsilon') for r in valid])) if valid else None))
    return output


def local_gate(rows,domain,objective):
    bank=defaultdict(dict)
    selected=[r for r in rows if r['domain']==domain and r['objective']==objective]
    for r in selected:bank[(r['case'],r['index'],r['stage'],r['channel'])][float(r['dose'])]=r
    eligible=[]
    for key,pack in bank.items():
        if not all(d in pack and pack[d]['status']=='valid' for d in DOSES):continue
        small=pack[.01]
        if abs(number(small,'delta_hat'))>5*number(small,'epsilon') and number(small,'alpha')<=.05:eligible.append((key,pack))
    stats=[]
    for dose in DOSES:
        vals=[p[dose] for _,p in eligible]
        stats.append(dict(dose=dose,eligible=len(vals),median_relative_residual=float(np.median([number(r,'relative_residual') for r in vals])) if vals else None,
                          median_absolute_residual=float(np.median([number(r,'absolute_residual') for r in vals])) if vals else None,
                          median_epsilon=float(np.median([number(r,'epsilon') for r in vals])) if vals else None))
    resolved=[r for r in selected if float(r['dose'])==.01 and r['status']=='valid' and r['actual_direction'] in ('positive','negative')]
    accuracy=float(np.mean([r['actual_direction']==r['predicted_direction'] for r in resolved])) if resolved else None
    small,large=stats[0],stats[-1]
    tests=dict(at_least_10_locally_resolved=len(eligible)>=10,
               small_median_relative_error_at_most_025=small['median_relative_residual'] is not None and small['median_relative_residual']<=.25,
               relative_error_not_worse_than_large_plus_002=bool(eligible) and small['median_relative_residual']<=large['median_relative_residual']+.02,
               absolute_error_decreases_to_noise_floor=bool(eligible) and small['median_absolute_residual']<=large['median_absolute_residual']+small['median_epsilon'],
               resolved_small_dose_sign_accuracy_at_least_080=accuracy is not None and accuracy>=.8)
    return dict(domain=domain,objective=objective,status='PASS' if all(tests.values()) else 'FAIL',tests=tests,
                eligible_cells=len(eligible),total_fixed_cells=len(bank),resolved_small_sign_cases=len(resolved),small_dose_sign_accuracy=accuracy,
                dose_statistics=stats,eligible_coordinates=[list(key) for key,_ in eligible])


def summarize(run):
    run=output_path(run)
    assert json.loads((run/'execution_results.json').read_text())['status']=='PASS'
    rows=list(csv.DictReader((run/'deltas.csv').open()));summary=summaries(rows)
    gates=[local_gate(rows,'L','transition_m1'),local_gate(rows,'U','u_transition')]
    write_csv(run/'fidelity_by_stage_dose.csv',summary)
    write_json(run/'fidelity_gates.json',gates)
    count=Counter(r['actual_direction'] for r in rows if r['status']=='valid')
    result=dict(status='PASS' if all(g['status']=='PASS' for g in gates) else 'FAIL',phase='C0',fidelity_cells=len(summary),
                valid_delta_rows=sum(r['status']=='valid' for r in rows),invalid_delta_rows=sum(r['status']!='valid' for r in rows),
                actual_directions=dict(count),indeterminate_fraction=count['indeterminate']/sum(count.values()) if count else None,
                main_gates=[{k:g[k] for k in ('domain','objective','status','eligible_cells','small_dose_sign_accuracy')} for g in gates],
                interpretation='Local Taylor/implementation fidelity only; does not establish boundary specificity, reliable channels, or training benefit.')
    write_json(run/'summary_results.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(json.dumps(summarize(a.run),indent=2))
