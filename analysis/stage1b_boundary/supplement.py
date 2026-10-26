"""Channel-level cross-scale responses, losses, and controlled scaling summaries."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from common import require_stage1b_run,A_RUN,SCALES,observation_patterns,describe,write_csv,write_json
from analyze import sources
from threadpoolctl import threadpool_limits


def supplement(run):
    run=require_stage1b_run(run)
    design=json.loads((run/'design.json').read_text());mapping=json.loads((run/'parameter_mapping.json').read_text())
    cross_rows=[]
    for stage,spec in mapping.items():
        reachable=[SCALES.index(s) for s in spec['reachable_scales']]
        for d,obj in [('L','A'),('L','band_w2'),('L','random_w2'),('L','core_w2'),('U','A'),('U','edge_w2'),('U','shuffle_w2')]:
            _,cross,rank,cancel=observation_patterns(sources(run,d,obj,stage),reachable,design['gradient_absolute_floor'],design['gradient_relative_floor'])
            for c in range(spec['channels']):
                row=dict(stage=stage,domain=d,objective=obj,channel=c,effective_rank_median=describe(rank[:,c])['median'],
                         cancellation_ratio_median=describe(cancel[:,c])['median'])
                for a in range(4):
                    for b in range(a+1,4):
                        values=describe(cross[:,c,a,b]);row[f'{SCALES[a]}_{SCALES[b]}_cosine']=values['median'];row[f'{SCALES[a]}_{SCALES[b]}_valid_n']=values['n']
                cross_rows.append(row)
    write_csv(run/'channel_multiscale.csv',cross_rows)
    coverage=json.loads((run/'loss_coverage.json').read_text());groups=defaultdict(list)
    for row in coverage:groups[(row['domain'],row['objective'],row['scale'])].append(row)
    losses=[]
    for (d,obj,scale),rows in groups.items():
        valid=[r for r in rows if r['status']=='valid'];native=[r['raw_loss'] for r in valid]
        losses.append(dict(domain=d,objective=obj,scale=scale,raw_loss=describe(native),
                           unchanged_A_factor=describe([r['A_lossnorm_factor'] for r in valid]),
                           A_denominator_contribution_value=describe([r['raw_loss']*r['A_lossnorm_factor'] for r in valid])))
    write_json(run/'loss_scaling.json',dict(records=losses,
         interpretation='B uses raw gradients; controlled A denominator is a sensitivity calculation, not independent B LossNorm. Within an observation, positive scaling does not rotate a nonzero channel gradient. Cross-crop weighted averages can rotate.'))
    write_json(run/'supplement_results.json',dict(status='PASS',channel_multiscale_rows=len(cross_rows),loss_scale_groups=len(losses)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):supplement(a.run.resolve())
