"""C0 geometry, node/perturbation identity, local Taylor and uncertainty contracts."""
from pathlib import Path
import sys
import time
import unittest
from unittest import mock
import numpy as np
import torch
import torch.nn.functional as F

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'analysis/stage1c_boundary_attribution'))
from support import (b,b_losses,ROOT,STAGES,DOSES,calibrate_epsilon,direction,dose_alpha,
                     node_statistics,output_path,write_json)
from pairs import adjacent_pairs,teacher_pairs,map_pairs,flipped_coordinates,permuted_partners
from objectives_c0 import transition,pair_consistency,Objectives
from nodes import Capture,attenuate,NODE_SPECS
from interventions import attenuation as old_attenuate


class C0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(142)
    def fixture(self):
        y=np.zeros((32,32,32),np.int64);y[8:24,8:24,8:24]=1
        return y,adjacent_pairs(y,seed_parts=('fixture',))
    def model(self):return b.VNetDS(n_channels=1,n_filters=2,normalization='batchnorm',has_dropout=True).eval()

    def test_pairs_oriented_native_local_exclude_border(self):
        y,p=self.fixture();self.assertTrue(p.valid)
        self.assertTrue(y[tuple(p.plus.T)].all());self.assertFalse(y[tuple(p.minus.T)].any())
        np.testing.assert_array_equal(np.abs(p.plus-p.minus).sum(1),1)
        self.assertTrue((p.plus>=4).all() and (p.minus<28).all())
        self.assertEqual(p.stats['per_pair_foreground'],p.stats['per_pair_background'])

    def test_pair_cap_determinism_and_global_rng_unchanged(self):
        y,_=self.fixture();state=np.random.get_state()
        a=adjacent_pairs(y,maximum=50,seed_parts=('fixed',));b1=adjacent_pairs(y,maximum=50,seed_parts=('fixed',))
        np.testing.assert_array_equal(a.plus,b1.plus);np.testing.assert_array_equal(a.minus,b1.minus)
        self.assertEqual(len(a.plus),50);np.testing.assert_array_equal(state[1],np.random.get_state()[1])

    def test_empty_boundary_and_invalid_padding(self):
        y,p=self.fixture();self.assertFalse(adjacent_pairs(y,valid=np.zeros_like(y)).valid)
        self.assertFalse(adjacent_pairs(np.zeros_like(y)).valid)
        self.assertFalse(adjacent_pairs(np.ones_like(y)).valid)
        with self.assertRaises(ValueError):transition(torch.zeros(1,2,32,32,32),adjacent_pairs(np.zeros_like(y)))

    def test_transition_gradient_class_direction(self):
        _,p=self.fixture();x=torch.zeros(1,2,32,32,32,requires_grad=True)
        g,=torch.autograd.grad(transition(x,p).sum(),x)
        self.assertTrue((g[0,1][tuple(p.plus.T)]<0).all())
        self.assertTrue((g[0,1][tuple(p.minus.T)]>0).all())

    def test_transition_shift_invariance_not_absolute_classification(self):
        y,p=self.fixture();z=np.where(y,5.,-5.)
        a=torch.tensor(np.stack([np.zeros_like(z),z])[None])
        c=a.clone();c[:,1]+=10
        torch.testing.assert_close(transition(a,p),transition(c,p),atol=0,rtol=0)
        truth=torch.tensor(y)[None]
        self.assertGreater(float(F.cross_entropy(c,truth)),float(F.cross_entropy(a,truth)))

    def test_final_head_only_and_coordinate_shape_contract(self):
        _,p=self.fixture();x=torch.zeros(1,2,32,32,32)
        with self.assertRaises(ValueError):transition([x]*4,p)
        with self.assertRaises(ValueError):transition(x[:,:,:16],p)
        values=[float(transition(x,p,m)) for m in (.5,1.,2.)]
        self.assertTrue(values[0]<values[1]<values[2])

    def test_flip_inverse_coordinate_alignment_exact(self):
        _,p=self.fixture();a=torch.arange(32**3).reshape(32,32,32)
        axes=(0,2);coords=flipped_coordinates(p.plus,p.shape,axes)
        np.testing.assert_array_equal(flipped_coordinates(coords,p.shape,axes),p.plus)
        torch.testing.assert_close(a[tuple(p.plus.T)],torch.flip(a,axes)[tuple(coords.T)],atol=0,rtol=0)

    def test_coarse_collapse_and_missing_occupancy_are_invalid(self):
        y,p=self.fixture()
        collapsed=map_pairs(p,(1,1,1),minimum=1)
        self.assertFalse(collapsed.valid);self.assertEqual(collapsed.stats['collapsed_pairs'],len(p.plus))
        coarse=map_pairs(p,(8,8,8),minimum=1)
        self.assertFalse(coarse.valid);self.assertEqual(coarse.status,'coarse_geometry_not_validated')
        occupancy=F.avg_pool3d(torch.tensor(y,dtype=torch.float32)[None,None],4)[0,0].numpy()
        coarse=map_pairs(p,(8,8,8),minimum=1,occupancy=occupancy)
        self.assertTrue(coarse.valid)
        self.assertTrue((occupancy[tuple(coarse.plus.T)]>=.75).all())
        self.assertTrue((occupancy[tuple(coarse.minus.T)]<=.25).all())

    def test_partner_permutation_preserves_endpoint_marginals(self):
        _,p=self.fixture();q=permuted_partners(p,('fixed',))
        np.testing.assert_array_equal(q.plus,p.plus)
        np.testing.assert_array_equal(np.sort(q.minus,axis=0),np.sort(p.minus,axis=0))
        self.assertTrue(q.stats['spatial_geometry_not_matched'])

    def test_teacher_stop_gradient_and_degenerate_prediction(self):
        y,_=self.fixture();z=torch.tensor(np.where(y,3.,-3.))[None,None]
        t=torch.cat([-z,z],1).requires_grad_(True);p=teacher_pairs(t,('teacher',))
        self.assertTrue(p.valid);self.assertEqual(p.stats['reliability_status'],'not_assessed_in_C0')
        s=t.detach().clone();s[:,1]+=.2*torch.linspace(-1.,1.,32)[None,:,None,None];s.requires_grad_(True)
        loss=pair_consistency(s,t,p).sum();g,gt=torch.autograd.grad(loss,(s,t),allow_unused=True)
        self.assertGreater(float(g.norm()),0);self.assertIsNone(gt)
        self.assertFalse(teacher_pairs(torch.zeros_like(t)).valid)

    def test_u_labels_rejected_and_image_only_loader(self):
        y,p=self.fixture()
        with self.assertRaises(ValueError):Objectives('U',p,labels=torch.tensor(y)[None])
        class ImageOnly:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getitem__(self,key):
                if key!='image':raise AssertionError('Forbidden U GT lookup')
                return np.zeros((8,8,8),np.float32)
        with mock.patch('core.h5py.File',return_value=ImageOnly()):
            accesses=[];b.read_image_only(Path('unlabeled.h5'),accesses,'unlabeled','dummy')
        self.assertEqual([r['key'] for r in accesses],['image'])

    def test_all_five_nodes_and_skip_fusion_identity(self):
        model=self.model();x=torch.randn(1,1,32,32,32)
        with Capture(model) as capture:out,features=model(x,deepsup=True,return_features=True)
        rows=capture.verify(features);self.assertEqual(len(rows),5)
        loss=out[-1].square().mean()
        nodes=[capture.nodes[s] for s in STAGES]
        gradients=torch.autograd.grad(loss,nodes,retain_graph=True)
        reference=torch.autograd.grad(loss,[features['encoder' if s[0]=='E' else 'decoder'][s] for s in STAGES])
        for node,g,r in zip(nodes,gradients,reference):
            self.assertEqual(g.shape,node.shape);torch.testing.assert_close(g,r,atol=0,rtol=0)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_new_interventions_equal_stage1b_final_path(self):
        model=self.model();x=torch.randn(1,1,32,32,32)
        with torch.no_grad():base,features=model(x,return_features=True)
        for s in STAGES:
            f=features['encoder' if s[0]=='E' else 'decoder'][s];mask=torch.ones(1,1,*f.shape[2:])
            with torch.no_grad(),attenuate(model,s,[0],mask,[.2],f) as trace:new=model(x)
            with torch.no_grad(),old_attenuate(model,s,[[0]],mask,[.2]):old=model(x)
            self.assertEqual(trace['calls'],1);torch.testing.assert_close(new,old,atol=0,rtol=0)
            with torch.no_grad(),attenuate(model,s,[0],mask,[0.]):null=model(x)
            torch.testing.assert_close(null,base,atol=0,rtol=0)

    def test_decoder_auxiliary_gradient_is_not_downstream_intervention(self):
        model=self.model();x=torch.randn(1,1,32,32,32)
        # A known active synthetic path, rather than an accidentally dead ReLU
        # channel in an untrained random network, is required for this contract.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module,(torch.nn.Conv3d,torch.nn.ConvTranspose3d)):
                    module.weight.fill_(.01)
                    if module.bias is not None:module.bias.fill_(.1)
        with Capture(model) as capture:out,features=model(x,deepsup=True,return_features=True)
        for s,head in [('D3',1),('D2',2)]:
            gradient,=torch.autograd.grad(out[head].square().mean(),capture.nodes[s],retain_graph=True)
            self.assertGreater(float(gradient.norm()),0)
            f=capture.nodes[s];mask=torch.ones(1,1,*f.shape[2:])
            with torch.no_grad(),attenuate(model,s,[0],mask,[.5]):changed=model(x,deepsup=True)
            torch.testing.assert_close(out[head],changed[head],atol=0,rtol=0)
            self.assertGreater(float((changed[-1]-out[-1]).detach().abs().max()),0)

    def test_parameter_gradients_are_separate_and_grad_slots_preserved(self):
        model=self.model();parameters=b.stage_parameters(model)
        for param in model.parameters():param.grad=torch.full_like(param,.25)
        with Capture(model) as c:out,_=model(torch.randn(1,1,32,32,32),deepsup=True,return_features=True)
        tensor_grads=torch.autograd.grad(out[-1].square().mean(),tuple(c.nodes.values()),retain_graph=True)
        parameter_grads=torch.autograd.grad(out[-1].square().mean(),[parameters[s].parameter for s in STAGES])
        for s,g,p in zip(STAGES,tensor_grads,parameter_grads):
            self.assertEqual(g.shape,c.nodes[s].shape);self.assertEqual(p.shape,parameters[s].parameter.shape)
            self.assertNotEqual(tuple(g.shape),tuple(p.shape))
        self.assertTrue(all(torch.equal(p.grad,torch.full_like(p,.25)) for p in model.parameters()))

    def test_signed_taylor_small_dose_has_correct_direction_and_error_order(self):
        f=torch.tensor([2.,1.],dtype=torch.float64).reshape(1,2,1,1,1).requires_grad_(True)
        target=torch.tensor([0.,2.],dtype=torch.float64).reshape_as(f)
        loss=(f-target).square().sum()/2;g,=torch.autograd.grad(loss,f)
        stats=node_statistics(g,f,torch.ones(1,1,1,1,1))
        for c in (0,1):
            remainder=[]
            for alpha in (.001,.003,.01):
                changed=f.detach().clone();changed[:,c]*=1-alpha
                actual=float(((changed-target).square().sum()/2-loss.detach()))
                predicted=-alpha*stats['signed_inner'][c]
                self.assertEqual(np.sign(actual),np.sign(predicted));remainder.append(abs(actual-predicted)/alpha)
            self.assertLess(remainder[0],remainder[-1])

    def test_net_and_absolute_gxa_not_interchangeable(self):
        f=torch.ones(1,1,1,1,2);g=torch.tensor([1.,-1.]).reshape_as(f)
        stats=node_statistics(g,f,torch.ones(1,1,1,1,2))
        self.assertEqual(stats['signed_inner'][0],0);self.assertEqual(stats['absolute_product_sum'][0],2)
        self.assertEqual(direction(0,1e-9),'indeterminate')

    def test_actual_energy_measures_tensor_subtraction_not_only_formula(self):
        class Pointwise(torch.nn.Module):
            def __init__(self):super().__init__();self.out_conv=torch.nn.Identity()
            def forward(self,x):return self.out_conv(x)
        model=Pointwise();value=torch.tensor([1.,3.,7.],dtype=torch.float32).reshape(1,1,1,1,3)
        with attenuate(model,'D1',[0],torch.ones_like(value),[.000003]) as trace:changed=model(value)
        actual=float((value-changed).double().norm());self.assertEqual(actual,trace['actual_l2'][0])
        self.assertGreater(trace['subtraction_roundoff_l2'][0],0)

    def test_uncertainty_interval_closed_and_null_adaptive_only(self):
        eps,stats=calibrate_epsilon(1.,[1.,1.000001,1.000003])
        self.assertAlmostEqual(eps,15e-6,places=12)
        for value in (-eps,0.,eps):self.assertEqual(direction(value,eps),'indeterminate')
        self.assertEqual(direction(2*eps,eps),'positive');self.assertEqual(direction(-2*eps,eps),'negative')
        self.assertEqual(direction(None,eps),'undefined')
        with self.assertRaises(ValueError):calibrate_epsilon(0.,[])

    def test_fixed_doses_and_unmatched_energy_not_clipped(self):
        self.assertEqual(DOSES,(.01,.03,.10))
        self.assertEqual(dose_alpha(1.,.03),(.03,'valid'))
        self.assertEqual(dose_alpha(0.,.03)[1],'zero_or_nonfinite_activation')
        self.assertEqual(dose_alpha(.01,.1)[1],'alpha_cap_exceeded')

    def test_c0_output_guard_rejects_historical_results(self):
        self.assertEqual(output_path(ROOT/'runs/example'),ROOT/'runs/example')
        for path in (b.A_RUN,PROJECT/'Results',b.ROOT/'runs/new',ROOT/'runs'):
            with self.assertRaises(ValueError):output_path(path)


if __name__=='__main__':
    start=time.monotonic();result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(C0Tests))
    write_json(ROOT/'TEST_RESULTS.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=[(str(t),s) for t,s in result.failures],errors=[(str(t),s) for t,s in result.errors],elapsed_seconds=time.monotonic()-start))
    raise SystemExit(0 if result.wasSuccessful() else 1)
