"""Boundary diagnostic definitions, invalid cases, stop-grad and interventions."""
from pathlib import Path
import sys
import unittest
import time
import numpy as np
import torch
import torch.nn.functional as F

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'analysis/stage1b_boundary'))
from common import require_stage1b_run,ROOT,A_RUN,VNetDS,extract_terms,stage_parameters,write_json
from losses import (Term,distances,labeled_regions,labeled_terms,pool_native,spatial_gradient,
                    teacher_regions,unlabeled_terms)
from interventions import attenuation
from ablate import energy_budget,matched_alpha
from summarize_ablations import ci


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(142)

    def fixture(self):
        y=np.zeros((32,32,32),np.int64);y[7:25,7:25,7:25]=1
        meta=dict(padding=[[0,0]]*3,starts=[0]*3,patch=list(y.shape))
        regions,stats=labeled_regions(y,distances(y),meta,1,142,'synthetic',0)
        return y,meta,regions,stats

    def test_balanced_ce_positive_outer_sign_and_volume_average_conservation(self):
        y,_,regions,stats=self.fixture()
        coarse=torch.randn(1,2,4,4,4,requires_grad=True)
        term=labeled_terms([coarse],y,regions['B'],stats['B'])[0]
        up=F.interpolate(coarse,size=y.shape,mode='nearest')
        ce=F.cross_entropy(up,torch.tensor(y)[None],reduction='none')[0]
        a,b=regions['B']
        expected=.5*(ce[torch.tensor(a)].mean()+ce[torch.tensor(b)].mean())
        torch.testing.assert_close(term.loss,expected,atol=1e-6,rtol=1e-6)
        self.assertAlmostEqual(term.stats['inner_reconstructed_native_mass'],a.sum())
        native=torch.zeros(1,2,32,32,32,requires_grad=True)
        t=labeled_terms([native],y,regions['B'],stats['B'])[0]
        g,=torch.autograd.grad(t.loss,native)
        self.assertTrue((g[0,1][torch.tensor(b)]>0).all())
        self.assertTrue((g[0,1][torch.tensor(a)]<0).all())

    def test_matched_regions_class_counts_and_no_duplicate_voxels(self):
        y,_,regions,stats=self.fixture()
        a,b=regions['B']
        for name in ('R','I'):
            self.assertEqual(stats[name]['status'],'valid')
            x,z=regions[name]
            self.assertEqual(x.sum(),a.sum());self.assertEqual(z.sum(),b.sum())
            self.assertTrue(y[x].all() and not y[z].any())
        self.assertFalse(((regions['I'][0]|regions['I'][1])&(a|b)).any())

    def test_full_volume_boundary_before_crop_has_no_artificial_crop_face(self):
        y=np.ones((40,40,40),np.int64)
        meta=dict(padding=[[0,0]]*3,starts=[10]*3,patch=[16]*3)
        _,stats=labeled_regions(y,distances(y),meta,1,142,'full',0)
        self.assertEqual(stats['B']['native_inner'],0)
        self.assertEqual(stats['B']['status'],'no_effective_boundary')

    def test_invalid_empty_boundary_and_insufficient_control_are_explicit(self):
        y,meta,_,_=self.fixture()
        y[:]=0
        regions,stats=labeled_regions(y,distances(y),meta,1,142,'empty',0)
        terms=labeled_terms([torch.zeros(1,2,4,4,4)],y,regions['B'],stats['B'])
        self.assertFalse(terms[0].valid);self.assertIsNone(terms[0].loss)
        y[7:25,7:25,7:25]=1
        _,stats=labeled_regions(y,distances(y),meta,3,142,'thick',0)
        self.assertEqual(stats['I']['status'],'insufficient_matched_region')

    def test_explicit_difference_operator_native_voxel_units(self):
        ramp=torch.arange(8.).reshape(1,1,8,1,1).expand(1,1,8,5,6)*4
        gradient=spatial_gradient(ramp,(4,2,1))
        torch.testing.assert_close(gradient[:,0],torch.ones_like(gradient[:,0]))
        torch.testing.assert_close(gradient[:,1:],torch.zeros_like(gradient[:,1:]))

    def test_constant_teacher_and_absent_levelset_are_invalid(self):
        logits=torch.zeros(1,2,24,24,24)
        regions=teacher_regions([logits]*4,[1,2,3],142,'flat',0)
        self.assertEqual(regions['B2'][1]['status'],'teacher_near_constant_probability')
        terms=unlabeled_terms([logits]*4,[logits]*4,*regions['B2'])
        self.assertTrue(all(not t.valid for t in terms))

    def test_teacher_stop_gradient_and_weight_histogram_control(self):
        ramp=torch.linspace(-4,4,24).reshape(1,1,24,1,1).expand(1,1,24,24,24)
        target=torch.cat([-ramp,ramp],1).clone().requires_grad_(True)
        source=(target.detach()+torch.randn_like(target)*.1).requires_grad_(True)
        regions=teacher_regions([target]*4,[1,2,3],142,'ramp',0)
        weights,info=regions['B2']
        self.assertEqual(info['status'],'valid')
        np.testing.assert_array_equal(np.sort(weights.ravel()),np.sort(regions['R2'][0].ravel()))
        term=unlabeled_terms([source],[target],weights,info)[0]
        g,t=torch.autograd.grad(term.loss,(source,target),allow_unused=True)
        self.assertIsNotNone(g);self.assertGreater(float(g.norm()),0)
        self.assertIsNone(t)

    def test_all_stage_gradients_and_existing_grad_slots_preserved(self):
        model=VNetDS(n_channels=1,n_filters=2,normalization='batchnorm',has_dropout=True).eval()
        mapping=stage_parameters(model)
        p=next(model.parameters());p.grad=torch.full_like(p,.25);prior=p.grad.clone()
        outputs=model(torch.randn(1,1,32,32,32),deepsup=True)
        y,_,regions,stats=self.fixture()
        terms=labeled_terms(outputs,y,regions['B'],stats['B'])
        raw=extract_terms(terms,mapping)
        for stage,spec in mapping.items():
            for scale,g in zip(('D4','D3','D2','D1'),raw[stage]):
                self.assertEqual(g is not None,scale in spec.reachable)
        torch.testing.assert_close(p.grad,prior,atol=0,rtol=0)

    def test_decoder_hook_changes_final_output_not_only_auxiliary_head(self):
        model=VNetDS(n_channels=1,n_filters=2,normalization='batchnorm').eval()
        x=torch.randn(1,1,32,32,32)
        with torch.no_grad():
            baseline,features=model(x,return_features=True)
            handle=model.out_conv_3.register_forward_pre_hook(lambda m,args:(args[0]*0,))
            ignored=model(x);handle.remove()
            torch.testing.assert_close(ignored,baseline,atol=0,rtol=0)
            size=features['decoder']['D3'].shape
            with attenuation(model,'D3',[list(range(size[1]))],torch.ones((1,1,*size[2:])),[1.]) as stats:
                changed=model(x)
            self.assertEqual(stats['calls'],1)
            self.assertGreater(float((changed-baseline).abs().max()),0)
            with attenuation(model,'D3',[[0]],torch.ones((1,1,*size[2:])),[0.]):
                unchanged=model(x)
            torch.testing.assert_close(unchanged,baseline,atol=0,rtol=0)

    def test_linear_head_logits_additive_but_ce_nonadditivity_is_not_synergy(self):
        head=torch.nn.Conv3d(2,2,1,bias=False)
        with torch.no_grad():head.weight.copy_(torch.tensor([1.,2.,-1.,-2.]).reshape_as(head.weight))
        x=torch.ones(1,2,2,2,2)
        a=x.clone();a[:,0]*=.5
        b=x.clone();b[:,1]*=.5
        joint=x*.5
        z0,za,zb,zj=[head(t) for t in (x,a,b,joint)]
        torch.testing.assert_close(zj-za-zb+z0,torch.zeros_like(z0),atol=0,rtol=0)
        y=torch.ones(1,2,2,2,dtype=torch.long)
        contrast=F.cross_entropy(zj,y)-F.cross_entropy(za,y)-F.cross_entropy(zb,y)+F.cross_entropy(z0,y)
        self.assertGreater(abs(float(contrast.detach())),1e-5)

    def test_energy_matching_and_invalid_cap_are_explicit(self):
        feature=torch.arange(1.,25.).reshape(1,3,2,2,2)
        masks={'B':torch.ones(1,1,2,2,2),'R':torch.ones(1,1,2,2,2)*.5}
        budget,energies=energy_budget(feature,masks)
        self.assertGreater(budget,0)
        for e in energies.values():
            alpha,status=matched_alpha(float(e[:2].norm()),budget*2**.5)
            self.assertEqual(status,'valid')
            self.assertAlmostEqual(float((e[:2]*alpha).norm()),budget*2**.5,places=5)
        self.assertEqual(matched_alpha(1e-13,budget)[1],'zero_or_nonfinite_activation')
        self.assertEqual(matched_alpha(1.,1.)[1],'unmatchable_budget_exceeds_alpha_cap')

    def test_result_guard_protects_old_directories(self):
        self.assertEqual(require_stage1b_run(ROOT/'runs'/'example'),ROOT/'runs'/'example')
        for path in (A_RUN,PROJECT/'Results',ROOT/'runs'):
            with self.assertRaises(ValueError):require_stage1b_run(path)

    def test_paired_case_bootstrap_uses_shared_indices(self):
        effect=[.1,-.2,.3,.4]
        self.assertEqual(ci(effect,'A'),ci(effect,'B3'))
        self.assertEqual(ci([0.]*4)['low'],0.)
        self.assertEqual(ci([0.]*4)['high'],0.)

    def test_pointwise_final_head_has_no_remote_band_ce_effect(self):
        head=torch.nn.Conv3d(3,2,1)
        activation=torch.randn(1,3,4,4,4)
        changed=activation.clone();changed[:,:,2:]*=.5
        target=torch.zeros(1,4,4,4,dtype=torch.long)
        original_ce=F.cross_entropy(head(activation),target,reduction='none')
        changed_ce=F.cross_entropy(head(changed),target,reduction='none')
        torch.testing.assert_close(original_ce[:,:2],changed_ce[:,:2],atol=0,rtol=0)


if __name__=='__main__':
    start=time.monotonic()
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests))
    write_json(PROJECT/'analysis/stage1b_boundary/TEST_RESULTS.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
               tests_run=result.testsRun,failures=[(str(t),s) for t,s in result.failures],errors=[(str(t),s) for t,s in result.errors],
               elapsed_seconds=time.monotonic()-start))
    raise SystemExit(0 if result.wasSuccessful() else 1)
