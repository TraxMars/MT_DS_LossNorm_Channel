"""Validate diagnostic mapping, loss algebra, graph status and image-only input."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis/stage1a_gradients"))
from core import (SCALES, deterministic_crop, gradient_validity, independent_losses,
                  parameter_gradients, read_image_only, signed_cosine, stage_parameters)
from networks import VNetDS
from objectives import consistency_loss, normalize_loss, supervised_loss
from stats_tools import (bootstrap_pair_mean, observation_patterns, pairwise_cosine,
                         profile_pca_and_clusters)


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(143)

    def model(self, **kwargs):
        return VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True,
                      **kwargs).eval()

    def test_discovered_parameter_names_axes_and_reachability(self):
        model = self.model()
        mapping = stage_parameters(model)
        expected = {
            "E1": ("block_one.conv.0.weight", 16, 0),
            "E2": ("block_two.conv.3.weight", 32, 0),
            "E3": ("block_three.conv.6.weight", 64, 0),
            "E4": ("block_four.conv.6.weight", 128, 0),
            "E5": ("block_five.conv.6.weight", 256, 0),
            "D4": ("block_five_up.conv.0.weight", 128, 1),
            "D3": ("block_six_up.conv.0.weight", 64, 1),
            "D2": ("block_seven_up.conv.0.weight", 32, 1),
            "D1": ("block_nine.conv.0.weight", 16, 0),
        }
        for stage, spec in mapping.items():
            self.assertEqual((spec.name, spec.channels, spec.output_axis), expected[stage])
        no_bn = VNetDS(n_channels=1, normalization="none")
        self.assertEqual(stage_parameters(no_bn)["E3"].name, "block_three.conv.4.weight")

    def test_transpose_convolution_output_channel_vectorization(self):
        mapping = stage_parameters(self.model(n_filters=2))
        spec = mapping["D4"]
        gradient = torch.arange(spec.parameter.numel()).reshape(spec.parameter.shape).float()
        vectors = spec.channel_vectors(gradient)
        for channel in range(spec.channels):
            torch.testing.assert_close(vectors[channel], gradient[:, channel].reshape(-1), atol=0, rtol=0)
        self.assertFalse(torch.equal(vectors, gradient.reshape(spec.channels, -1)))

    def test_four_scale_losses_and_original_aggregate_lossnorm(self):
        labels = torch.randint(0, 2, (2, 16, 16, 16))
        outputs = [torch.randn(2, 2, size, size, size, requires_grad=True) for size in (2, 4, 8, 16)]
        terms, total, _ = independent_losses(outputs, labels=labels)
        original = supervised_loss(outputs, labels)[0]
        torch.testing.assert_close(total, original, atol=0, rtol=0)
        torch.testing.assert_close(torch.stack(terms).mean(), original)
        targets = [torch.randn_like(output, requires_grad=True) for output in outputs]
        mt, mt_total, _ = independent_losses(outputs, teacher_outputs=targets)
        torch.testing.assert_close(mt_total, consistency_loss(outputs, targets), atol=0, rtol=0)
        g = torch.autograd.grad(normalize_loss(mt_total), outputs, retain_graph=True)
        raw = torch.autograd.grad(torch.stack(mt).mean(), outputs)
        for actual, expected in zip(g, raw):
            torch.testing.assert_close(actual, expected / mt_total.detach().clamp_min(1e-8))
        self.assertTrue(all(t.grad is None for t in targets))

    def test_autograd_reachability_shapes_existing_grad_slots_and_state(self):
        for domain in ("L", "U"):
            model = self.model(n_filters=2)
            mapping = stage_parameters(model)
            initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            first = next(model.parameters())
            first.grad = torch.full_like(first, 0.123)
            prior = first.grad.clone()
            rng = torch.get_rng_state().clone()
            outputs = model(torch.ones(2, 1, 16, 16, 16), deepsup=True)
            teacher = [output.detach() + torch.tensor([0.1, -0.1])[None, :, None, None, None]
                       for output in outputs]
            labels = torch.randint(0, 2, (2, 16, 16, 16))
            losses, total, _ = independent_losses(outputs, labels=labels if domain == "L" else None,
                                                   teacher_outputs=teacher if domain == "U" else None)
            gradients, check = parameter_gradients(losses, total, mapping, weight=1 if domain == "L" else 0.3)
            for stage, spec in mapping.items():
                for scale, gradient in zip(SCALES, gradients[stage]):
                    self.assertEqual(gradient is not None, scale in spec.reachable)
                    if gradient is not None:
                        self.assertEqual(gradient.shape, (spec.channels, spec.parameter.numel() // spec.channels))
            torch.testing.assert_close(first.grad, prior, atol=0, rtol=0)
            self.assertTrue(all(p.grad is None for p in list(model.parameters())[1:]))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
            self.assertEqual(check["normalized_scale_factor"], (1 if domain == "L" else 0.3) /
                             (4 * max(float(total.detach()), 1e-8)))
            # Only label generation consumed RNG; forward/autograd used no stochastic layers.
            self.assertFalse(torch.equal(rng, torch.get_rng_state()))

    def test_signed_cosine_invalid_zero_tiny_and_nonfinite(self):
        left = np.array([[3., 4.], [0, 0], [1e-16, 0], [np.nan, 0]])
        right = np.array([[-3., -4.], [1, 0], [1, 0], [1, 0]])
        scores, valid, statuses, _ = signed_cosine(left, right)
        self.assertAlmostEqual(scores[0], -1)
        self.assertEqual(valid.tolist(), [True, False, False, False])
        self.assertEqual(statuses.tolist(), ["valid", "exact_zero", "below_threshold", "nonfinite"])
        self.assertTrue(np.isnan(scores[1:]).all())

    def test_unlabeled_loader_forbids_label_and_crops_ignore_label_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image_only.h5"
            image = np.arange(20*24*28, dtype=np.float32).reshape(20, 24, 28)
            with h5py.File(path, "w") as handle:
                handle["image"] = image
            original = h5py.File.__getitem__
            def guarded(handle, key):
                if key != "image":
                    raise AssertionError("Unlabeled loader accessed a forbidden key")
                return original(handle, key)
            access = []
            with mock.patch.object(h5py.File, "__getitem__", guarded):
                actual = read_image_only(path, access, "unlabeled", "synthetic")
            np.testing.assert_array_equal(actual, image)
            self.assertEqual([row["key"] for row in access], ["image"])
            x1, y1, crop1 = deterministic_crop(image, None, (16, 16, 16), 142, "synthetic", 0)
            x2, _, crop2 = deterministic_crop(image, np.ones_like(image, dtype=np.int64),
                                              (16, 16, 16), 142, "synthetic", 0)
            self.assertIsNone(y1)
            self.assertEqual(crop1, crop2)
            torch.testing.assert_close(x1, x2, atol=0, rtol=0)

    def test_case_pair_cosines_negative_sign_invalid_masks_and_bootstrap(self):
        left = np.array([[[1.,0.],[0.,0.]],[[2.,0.],[0.,0.]]])
        right = np.array([[[-1.,0.],[0.,0.]],[[-3.,0.],[0.,0.]]])
        scores, valid, _, _ = pairwise_cosine(left,right)
        np.testing.assert_array_equal(scores[0], -np.ones((2,2)))
        self.assertTrue(valid[0].all() and not valid[1].any())
        self.assertTrue(np.isnan(scores[1]).all())
        bounds = bootstrap_pair_mean(scores,draws=10)
        np.testing.assert_array_equal(bounds[:,0],[-1.,-1.])
        self.assertTrue(np.isnan(bounds[:,1]).all())

    def test_single_reachable_head_has_no_interpretable_profile_pca(self):
        result, _, scores, labels = profile_pca_and_clusters(
            np.ones((3,8,1)),np.ones((4,8,1)),np.ones(8),np.ones(8),bootstrap_draws=2)
        self.assertEqual(result["status"],"CONSTANT_PROFILE")
        self.assertEqual(result["rank"],0)
        self.assertEqual(result["algebraic_rank_upper_bound"],0)
        self.assertIsNone(scores)
        self.assertIsNone(labels)

    def test_profile_simplex_rank_constraint_is_explicit(self):
        generator=np.random.default_rng(142)
        left=generator.dirichlet(np.ones(4),size=(3,16))
        right=generator.dirichlet(np.ones(4),size=(5,16))
        result,_,_,_=profile_pca_and_clusters(left,right,np.ones(16),np.ones(16),bootstrap_draws=3)
        self.assertEqual(result["status"],"OK")
        self.assertLessEqual(result["rank"],6)
        self.assertEqual(result["algebraic_rank_upper_bound"],6)

    def test_rank_one_can_describe_conflicting_heads_not_cooperation(self):
        array=np.full((1,4,2,2),np.nan)
        array[0,0]=[[1.,0.],[1.,0.]]
        array[0,1]=[[1.,0.],[-1.,0.]]
        _,cross,rank,cancellation=observation_patterns(array,[0,1],1e-12,1e-6)
        np.testing.assert_allclose(rank[0],[1.,1.])
        np.testing.assert_allclose(cancellation[0],[1.,0.])
        self.assertEqual(cross[0,1,0,1],-1.)


if __name__ == "__main__":
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests))
    report = dict(status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun,
                  failures=[(str(t), s) for t, s in result.failures], errors=[(str(t), s) for t, s in result.errors],
                  elapsed_seconds=time.monotonic()-started)
    path = PROJECT / "analysis/stage1a_gradients/TEST_RESULTS.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
