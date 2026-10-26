"""C1 scientific controls, cross-target separation and protected execution."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'analysis/stage1c_c1_boundary_specificity'))
from bridge import (b,b_losses,ROOT,C0,STAGES,DOSES,PRIMARY,adjacent_pairs,permuted_partners,
                    Capture,attenuate,calibrate_epsilon,direction,dose_alpha,output_path)
from control_pairs import far_controls,values,compare_endpoints,pair_statistics
from targets import Targets
from measures import cosine_rows,rho,bootstrap
from runner import candidate_plan
from objectives_c0 import transition

class C1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(142)
    def fixture(self):
        y=np.zeros((32,32,32),np.int64);y[7:25,7:25,7:25]=1
        p=adjacent_pairs(y,maximum=128,seed_parts=('C1_test',));di,do=b_losses.distances(y)
        z=np.where(y,2.,-2.)+.01*np.indices(y.shape)[0]
        return y,p,di,do,z

    def test_far_and_matched_geometry_class_count_and_caliper(self):
        y,p,di,do,z=self.fixture()
        f,m,t,info=far_controls(p,z,y,di,do,np.ones_like(y,bool),('fixture',))
        self.assertTrue(f.valid and m.valid and t.valid)
        self.assertEqual(len(f.plus),len(p.plus));self.assertEqual(len(m.plus),len(t.plus))
        self.assertTrue(y[tuple(f.plus.T)].all());self.assertFalse(y[tuple(f.minus.T)].any())
        self.assertTrue((di[tuple(f.plus.T)]>2).all() and (do[tuple(f.minus.T)]>2).all())
        self.assertEqual(len(np.unique(f.plus,axis=0)),len(f.plus))
        for a,c in zip(values(z,m),values(z,t)):self.assertTrue((abs(a-c)<.5+1e-12).all())

    def test_far_matching_insufficient_support_is_explicit(self):
        y,p,di,do,z=self.fixture();z[y.astype(bool)&(di>2)]=20;z[(~y.astype(bool))&(do>2)]=-20
        _,m,t,info=far_controls(p,z,y,di,do,np.ones_like(y,bool),('no_support',))
        self.assertFalse(m.valid or t.valid);self.assertEqual(info['unmatched_fraction'],1.)
        f,*_=far_controls(p,z,y,di,do,np.zeros_like(y,bool),('invalid',));self.assertFalse(f.valid)

    def test_permutation_preserves_multisets_and_reports_geometry(self):
        y,p,_,_,z=self.fixture();q=permuted_partners(p,('C1',0))
        np.testing.assert_array_equal(q.plus,p.plus)
        np.testing.assert_array_equal(np.sort(np.ravel_multi_index(q.minus.T,y.shape)),np.sort(np.ravel_multi_index(p.minus.T,y.shape)))
        self.assertEqual(compare_endpoints(z,p,q)['bg']['logit_wasserstein'],0.)
        self.assertTrue((np.linalg.norm(q.plus-q.minus,axis=1)>1).any())

    def test_pairing_has_no_explicit_geometric_term(self):
        y,p,_,_,_=self.fixture();q=permuted_partners(p,('C1',0))
        z=torch.tensor(np.where(y,2.,-2.));logits=torch.stack([torch.zeros_like(z),z])[None].requires_grad_()
        a=transition(logits,p);c=transition(logits,q)
        ga,=torch.autograd.grad(a,logits,retain_graph=True);gc,=torch.autograd.grad(c,logits)
        torch.testing.assert_close(a,c,atol=0,rtol=0);torch.testing.assert_close(ga,gc,atol=1e-17,rtol=1e-14)

    def test_relative_margin_satisfied_with_incorrect_classification(self):
        y,p,_,_,_=self.fixture();z=np.where(y,15.,5.)
        stat=pair_statistics(p,z)
        for r in stat['margins']:
            self.assertEqual(r['satisfied_fraction'],1.);self.assertEqual(r['wrong_fraction_given_satisfied'],1.)
        group=next(g for g in stat['groups'] if g['fg_correct'] and not g['bg_correct'])
        self.assertEqual(group['fraction'],1.)

    def test_endpoint_correctness_four_exhaustive_categories(self):
        y,p,_,_,z=self.fixture();s=pair_statistics(p,z)
        self.assertEqual(sum(g['count'] for g in s['groups']),len(p.plus))
        self.assertAlmostEqual(sum(g['fraction'] for g in s['groups']),1.)

    def test_target_bank_preserves_C0_raw_formulas_no_sum(self):
        y,p,_,_,z=self.fixture();inner=(y>0);outer=~inner
        logits=torch.tensor(np.stack([np.zeros_like(z),z])[None]);labels=torch.tensor(y)[None]
        t=Targets(dict(true=p,permuted=permuted_partners(p,('C1',0)),far=p),labels,(inner,outer))
        terms=t(logits);old=t.original(logits)
        torch.testing.assert_close(terms['boundary'],old['boundary_ce_w2'],atol=0,rtol=0)
        torch.testing.assert_close(terms['transition'],old['transition_m1'],atol=0,rtol=0)
        self.assertTrue(all(n in terms for n in PRIMARY))

    def test_final_head_and_native_coordinate_contract(self):
        y,p,_,_,_=self.fixture()
        with self.assertRaises(ValueError):transition(torch.zeros(1,2,16,16,16),p)
        with self.assertRaises(ValueError):transition([torch.zeros(1,2,32,32,32)]*4,p)

    def test_same_node_capture_matches_skip_and_final_gradient(self):
        m=b.VNetDS(n_channels=1,n_filters=2,normalization='batchnorm',has_dropout=True).eval()
        with Capture(m) as c:out,f=m(torch.randn(1,1,32,32,32),deepsup=True,return_features=True)
        self.assertEqual(len(c.verify(f)),5)
        gs=torch.autograd.grad(out[-1].square().mean(),list(c.nodes.values()),retain_graph=True)
        for s,g in zip(STAGES,gs):self.assertEqual(g.shape,c.nodes[s].shape)
        self.assertFalse(any(mod._forward_hooks or mod._forward_pre_hooks for mod in m.modules()))

    def test_D1_disjoint_support_cannot_change_boundary_logits(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.out_conv=torch.nn.Conv3d(2,2,1)
            def forward(self,x):return self.out_conv(x)
        m=Model().eval();x=torch.randn(1,2,8,8,8);base=m(x)
        mask=torch.zeros(1,1,8,8,8);mask[:,:,:4]=1
        with attenuate(m,'D1',[0],mask,[.1],x):altered=m(x)
        torch.testing.assert_close(base[:,:,4:],altered[:,:,4:],atol=0,rtol=0)
        self.assertGreater(float((base[:,:,:4]-altered[:,:,:4]).detach().abs().max()),0)

    def test_parameter_and_feature_coordinate_kinds_separate(self):
        m=b.VNetDS(n_channels=1,n_filters=2,normalization='batchnorm',has_dropout=True).eval();mapping=b.stage_parameters(m)
        with Capture(m) as c:out,_=m(torch.randn(1,1,32,32,32),deepsup=True,return_features=True)
        gs=torch.autograd.grad(out[-1].square().mean(),[c.nodes['D3'],mapping['D3'].parameter])
        self.assertNotEqual(gs[0].shape,gs[1].shape);self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_cosine_zero_gradient_invalid_and_signed(self):
        a=np.array([[1.,0],[0,0],[-1,0],[1e-20,0]])
        cos,_,_,status=cosine_rows(a,np.ones_like(a))
        self.assertGreater(cos[0],0);self.assertLess(cos[2],0)
        self.assertTrue(np.isnan(cos[1]) and np.isnan(cos[3]));self.assertEqual(status[1],'zero_gradient')

    def test_uncertainty_not_forced_to_sign_and_alpha_no_clipping(self):
        eps,_=calibrate_epsilon(.3,[.3,.30000001]);self.assertEqual(direction(eps,eps),'indeterminate')
        self.assertEqual(direction(None,eps),'undefined');self.assertEqual(dose_alpha(.01,.1)[1],'alpha_cap_exceeded')
        self.assertEqual(DOSES,(.01,.03,.10))

    def test_correlations_constant_and_case_bootstrap_pairing(self):
        self.assertEqual(rho([1,1,1],[1,2,3])[1],'constant_response')
        ids=np.random.default_rng(142).integers(0,4,(2000,4));a=np.array([.1,.2,-.2,.3]);b_=a+.1
        self.assertAlmostEqual(bootstrap(b_-a,ids)['mean'],.1)
        self.assertLess(np.diff(bootstrap(b_-a,ids)['ci'])[0],1e-12)

    def test_discovery_plan_only_reads_discovery_and_frozen_signed_scores(self):
        cases=['a','b','c','d'];table=[]
        for s in STAGES:
            for n in PRIMARY:
                for r in ('B','I'):
                    for index in range(8):
                        for c in range(8):table.append(dict(stage=s,channel=c,objective=n,support=r,index=index,case=cases[index//2],unit_response=c-3,
                            max_dose_status='valid',gradient_status='valid'))
        with tempfile.TemporaryDirectory() as d:
            p=candidate_plan({'contributions':table},cases,Path(d))
            self.assertFalse(p['probe_cases_opened']);self.assertEqual(p['scores']['D1']['B']['transition']['0'],-3.)
            self.assertEqual(p['discovery_cases'],cases)

    def test_only_C1_outputs_allowed(self):
        self.assertEqual(output_path(ROOT/'runs/unit'),ROOT/'runs/unit')
        for p in (C0/'runs/new',b.ROOT/'runs/new',PROJECT/'Results',ROOT/'runs'):
            with self.assertRaises(ValueError):output_path(p)

if __name__=='__main__':unittest.main(verbosity=2)
