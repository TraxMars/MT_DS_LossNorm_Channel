"""Independent final-head C1 objectives; no sum/normalization or training import."""
from bridge import Objectives,transition

class Targets:
    def __init__(self,pairs,labels,regions):
        self.pairs=pairs
        self.original=Objectives('L',pairs['true'],labels=labels,regions=regions)
    def __call__(self,logits):
        old=self.original(logits)
        result=dict(region=old['region_ce'],boundary=old['boundary_ce_w2'],transition=old['transition_m1'],
                    transition_m0_5=old['transition_m0_5'],transition_m2=old['transition_m2'])
        for name,p in self.pairs.items():
            if name=='true':continue
            if p.valid:result[name]=transition(logits,p,1.)
        return result
