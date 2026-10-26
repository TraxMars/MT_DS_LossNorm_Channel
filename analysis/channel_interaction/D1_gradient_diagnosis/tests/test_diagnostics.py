"""Artificial-tensor tests only. Real validation always runs both datasets."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import h5py
import numpy as np
import torch

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]/'code'))
from diagnostic import (Capture,extract,gradient_validity,measure_unit,pack,preserve,read_image_only,
                        signed_cosine,stage_parameters)
from common import ROOT,SPECS,new_output,write_json
from networks import VNetDS

class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def models(self):
        s=VNetDS(n_channels=1,n_classes=2,n_filters=2,normalization='batchnorm',has_dropout=True).eval()
        t=VNetDS(n_channels=1,n_classes=2,n_filters=2,normalization='batchnorm',has_dropout=True).eval()
        t.requires_grad_(False)
        return s,t

    def test_datasets_lock_patches_and_splits(self):
        self.assertEqual((SPECS['LA']['labeled'],SPECS['LA']['unlabeled'],SPECS['LA']['patch']),(8,72,[112,112,80]))
        self.assertEqual((SPECS['Pancreas']['labeled'],SPECS['Pancreas']['unlabeled'],SPECS['Pancreas']['patch']),(6,56,[96,96,96]))

    def test_c0_identity_without_boundary_imports(self):
        s,_=self.models()
        with Capture(s) as capture:
            outputs,features=s(torch.randn(1,1,32,32,32),deepsup=True,return_features=True)
            rows=capture.verify(features)
        self.assertTrue(all(r['same_object'] for r in rows))
        self.assertFalse(any('stage1b' in str(getattr(m,'__file__','')) for m in list(sys.modules.values()) if m))
        self.assertFalse(any(s._forward_hooks for s in s.modules()))

    def test_train_restore_bn_rng_modes_and_existing_grad(self):
        s,t=self.models();s.block_three.eval()
        p=next(s.parameters());p.grad=torch.ones_like(p)
        before=torch.get_rng_state().clone()
        with preserve((s,t),train=True,seed=82) as report:
            s(torch.randn(2,1,32,32,32));t(torch.randn(2,1,32,32,32))
        self.assertTrue(report['changed_buffers_before_restore'])
        self.assertTrue(report['grad_slots_unchanged'])
        self.assertTrue(torch.equal(torch.get_rng_state(),before))
        self.assertTrue(torch.equal(p.grad,torch.ones_like(p)))
        self.assertFalse(s.block_three.training)

    def test_restore_even_when_forward_raises(self):
        s,t=self.models()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with preserve((s,t),train=True,seed=9) as report:
                s(torch.randn(2,1,32,32,32));raise RuntimeError('injected')
        self.assertTrue(report['state_restored'] and report['rng_restored'])

    def test_real_losses_parameter_reconstruction_eval_and_train(self):
        s,t=self.models();mapping={k:v for k,v in stage_parameters(s).items() if k in ('E3','D1')}
        x=torch.randn(2,1,32,32,32);y=torch.randint(0,2,(2,32,32,32))
        for train in (False,True):
            with preserve((s,t),train=train,seed=71):
                for target in (y,None):
                    raw,meta=extract(s,t,x,target,mapping,dict(normalization_eps=1e-8,consistency=.3))
                    self.assertEqual(len(meta['checks']['gradient_reconstruction']),2)
                    self.assertTrue(np.isnan(pack(raw)['D1'][:3]).all())
                    self.assertTrue(np.isfinite(pack(raw)['E3']).all())

    def test_train_dropout_replay_is_exact(self):
        s,t=self.models();x=torch.randn(2,1,32,32,32);outputs=[]
        for _ in range(2):
            with preserve((s,t),train=True,seed=91): outputs.append(s(x).detach())
        self.assertTrue(torch.equal(*outputs))

    def test_precision_retry_replays_both_roles_and_restores_original_float32(self):
        import diagnostic
        s,t=self.models();before=[{k:v.clone() for k,v in m.state_dict().items()} for m in (s,t)]
        x=torch.randn(2,1,32,32,32);y=torch.randint(0,2,(2,32,32,32))
        mapping={k:v for k,v in stage_parameters(s).items() if k in ('E3','D1')}
        calls=[];original=diagnostic.extract
        def injected(*args,**kwargs):
            calls.append(args[2].dtype)
            if len(calls)==2:raise AssertionError('LossNorm gradient reconstruction D1: injected')
            return original(*args,**kwargs)
        with mock.patch.object(diagnostic,'extract',injected):
            result,report=measure_unit(s,t,{'L':(x,y),'U':(x,None)},mapping,
                                       dict(normalization_eps=1e-8,consistency=.3),train=True,seed=21)
        self.assertEqual(calls,[torch.float32,torch.float32,torch.float64,torch.float64])
        self.assertEqual(report['accepted_precision'],'torch.float64')
        self.assertEqual(len(report['failed_attempts']),1)
        self.assertTrue(all(meta['gradient_precision']=='torch.float64' for _,meta in result.values()))
        for model,state in zip((s,t),before):
            self.assertTrue(all(torch.equal(value,state[k]) and value.dtype==state[k].dtype for k,value in model.state_dict().items()))

    def test_U_loader_does_not_access_missing_or_trap_label(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'U.h5'
            with h5py.File(path,'w') as f: f['image']=np.ones((9,10,11),np.float32)
            access=[];image=read_image_only(path,access,'unlabeled','case')
            self.assertEqual(image.shape,(9,10,11));self.assertEqual([r['key'] for r in access],['image'])
            original=h5py.Group.__getitem__
            def trap(group,key):
                if key!='image': raise AssertionError('U GT access')
                return original(group,key)
            with mock.patch.object(h5py.Group,'__getitem__',trap): read_image_only(path,[],'unlabeled','case')

    def test_invalid_cosines_stay_undefined_and_scaling_preserves_sign(self):
        a=np.array([[1.,2.],[0.,0.],[1e-20,0.],[np.nan,1.]])
        b=-a
        cosine,valid,_,_=signed_cosine(a,b)
        self.assertEqual(valid.tolist(),[True,False,False,False])
        self.assertAlmostEqual(cosine[0],-1.)
        self.assertTrue(np.isnan(cosine[1:]).all())
        np.testing.assert_allclose(signed_cosine(a*11,b*3)[0],cosine,equal_nan=True)

    def test_output_confinement_and_nonfinite_serialization(self):
        with self.assertRaises(ValueError): new_output('D1_gradient_diagnosis','/tmp/not_a_D1_result')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'x.json';write_json(p,dict(nan=np.nan,inf=np.inf))
            self.assertEqual(json.loads(p.read_text()),dict(nan=None,inf=None))

    def test_no_optimizer_or_intervention_entry(self):
        for p in (ROOT/'D1_gradient_diagnosis/code').glob('*.py'):
            tree=ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('step','backward','update_ema','attenuate'))

if __name__=='__main__': unittest.main(verbosity=2)
