"""Statistical algebra tests with artificial vectors; never reads real images."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from summarize import ratio,summarize_dataset
from stats_tools import bootstrap_pair_mean,pairwise_cosine
from diagnostic import signed_cosine

class StatisticsTests(unittest.TestCase):
    def test_cancellation_and_broadcast(self):
        left=np.ones((2,1,3,4));right=np.ones((1,5,3,4))
        np.testing.assert_allclose(ratio(left,right),np.ones((2,5,3)))
        np.testing.assert_allclose(ratio(left,-right),np.zeros((2,5,3)))
        self.assertTrue(np.isnan(ratio(left,np.zeros_like(right))).all())

    def test_case_bootstrap_keeps_two_domains(self):
        left=np.ones((3,2,4));right=-np.ones((5,2,4))
        scores,valid,_,_=pairwise_cosine(left,right)
        self.assertEqual(scores.shape,(2,3,5));self.assertTrue(valid.all())
        np.testing.assert_allclose(bootstrap_pair_mean(scores,draws=200),-np.ones((2,2)))

    def test_before_average_normalization_can_change_direction(self):
        # Positive rescaling preserves each observation; weighted averaging can reverse a mean.
        observations=np.array([[3.,0.],[-1.,1.]])
        weights=np.array([.1,10.])
        self.assertGreater(signed_cosine(observations,observations*weights[:,None])[0].min(),.999)
        self.assertLess(np.dot(observations.mean(axis=0),(observations*weights[:,None]).mean(axis=0)),0)

    def test_zero_channels_never_enter_pairwise_conflict_denominator(self):
        left=np.zeros((2,3,4));right=np.ones((5,3,4))
        scores,valid,_,_=pairwise_cosine(left,right)
        self.assertFalse(valid.any());self.assertTrue(np.isnan(scores).all())
        self.assertTrue(np.isnan(bootstrap_pair_mean(scores)).all())

    def test_summary_pipeline_retains_dead_and_unreachable_channels(self):
        # Artificial vectors exercise the same per-case, mini, CSV and JSON pipeline.
        from common import write_json
        generator=np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);out=run/'Pancreas';out.mkdir();(out/'raw').mkdir()
            write_json(out/'design.json',dict(labeled_ids=['L0','L1'],unlabeled_ids=['U0','U1']))
            write_json(out/'execution.json',dict(fp64_fallback_units=0))
            observations=[]
            def add(mode,unit,role,repeat,batch=0,case='case'):
                index=len(observations);e=generator.normal(size=(4,3,4));e[:,0]=0
                d=generator.normal(size=(4,2,4));d[:3]=np.nan
                name=f'raw/{index}.npz';np.savez(out/name,E3=e,D1=d)
                observations.append(dict(index=index,mode=mode,unit=unit,role=role,repeat=repeat,batch=batch,
                    cases=[case],raw_file=name,checks=dict(normalized_scale_factor=2.,gradient_reconstruction=[
                        dict(stage=s,raw_relative_l2_error=0.,normalized_relative_l2_error=0.) for s in ('D1','E3')])))
            for role in ('L','U'):
                for case in range(2):
                    for repeat in range(2):add('eval','case',role,repeat,case=f'{role}{case}')
            for mode in ('eval','train'):
                for batch in range(3):
                    for repeat in range(2):
                        for role in ('L','U'):add(mode,'mini',role,repeat,batch=batch)
            write_json(out/'observations.json',observations)
            result=summarize_dataset('Pancreas',run)
            self.assertEqual(result['status'],'PASS')
            self.assertGreater(result['invalid_observation_channels'],0)
            self.assertEqual(len(result['scale_summary']),10)
            self.assertTrue((out/'mini_relations.csv').is_file())
            self.assertNotIn('NaN',(out/'summary.json').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
