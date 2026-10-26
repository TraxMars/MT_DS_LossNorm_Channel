"""Stage 0 compatibility checks; stdlib unittest, no training entry point.

Run on CPU by default. --gpu-smoke reads one real batch per dataset, performs
train/eval forwards and backward comparisons, and never steps an optimizer.
"""

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import unittest
from unittest import mock

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from boundary import describe_stage_features, get_stage_feature, stage_feature_info
from boundary.features import STAGES
from networks import VNetDS
from objectives import consistency_loss, normalize_loss, supervised_loss, update_ema

REFERENCE = PROJECT / "analysis/stage_features/reference"
OBSERVATIONS = {}
BASELINE_SHA256 = "57f29fc308d484c09cd2f080c5e283aea7a59e99eb4b0bf5612a25bd0e1e41b2"


def load_class(filename, name):
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


Baseline = load_class(REFERENCE / "vnet_ds_baseline.py", "VNetDS")
Upstream = load_class(REFERENCE / "upstream_vnet.py", "DuCiSC")


def new_model(cls=VNetDS, device="cpu", dropout=True):
    return cls(n_channels=1, n_classes=2, normalization="batchnorm",
               has_dropout=dropout).to(device)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(item) for item in value)
    return copy.deepcopy(value)


def assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), (left.keys(), right.keys())
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right, (left, right)


def rng_state(device="cpu"):
    return snapshot(dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch=torch.get_rng_state(), cuda=(
                             torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None)))


def restore_rng(state, device="cpu"):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def compare_gpu_results(actual, expected):
    """Require exact network behavior; expose native CUDA loss reduction noise.

    The original CUDA CE forward uses a nondeterministic reduction. That scalar
    is also the LossNorm denominator, so combined gradients can differ by a few
    float32 rounding units even for an unchanged Baseline repeated against itself.
    Raw objective gradients, predictions, BN and RNG remain strict comparisons.
    """
    strict = ("supervised_outputs", "unlabeled_outputs", "teacher_outputs", "sup_grad",
              "mt_grad", "student_buffers", "teacher_buffers", "final_rng")
    for key in strict:
        assert_exact(actual[key], expected[key])
    summary = dict(exact_keys=list(strict), scalar_rtol=1e-6, scalar_atol=1e-7,
                   normalized_gradient_rtol=1e-6, normalized_gradient_atol=1e-7)
    for key in ("losses", "combined_grad"):
        differences = []
        for left, right in zip(actual[key], expected[key]):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
            differences.append(float((left - right).abs().max()))
        summary[key] = dict(exact=all(x == 0 for x in differences),
                            max_abs_difference=max(differences))
        if key == "losses":
            summary[key]["per_scalar_abs_difference"] = differences
    return summary


def bn_buffers(model):
    return snapshot({k: v for k, v in model.named_buffers()})


def compare_pipeline(cls, initial, images, labels, training, feature_flag, weight, rng,
                     teacher_initial=None):
    """Mirror the three Baseline forwards; collect raw and normalized gradients."""
    device = str(images.device)
    student = new_model(cls, device)
    student.load_state_dict(initial, strict=True)
    teacher = copy.deepcopy(student).requires_grad_(False)
    if teacher_initial is not None:
        teacher.load_state_dict(teacher_initial, strict=True)
    student.train(training)
    teacher.train(training)
    initial_buffers = bn_buffers(student)
    initial_teacher_buffers = bn_buffers(teacher)
    restore_rng(rng, device)

    def forward(model, x):
        if not feature_flag:
            return model(x, deepsup=True)
        outputs, features = model(x, deepsup=True, return_features=True)
        # Metadata reads must not change computation, RNG, or retained tensors.
        describe_stage_features(features)
        return outputs

    supervised_outputs = forward(student, images[:2])
    unlabeled_outputs = forward(student, images[2:])
    with torch.no_grad():
        teacher_outputs = forward(teacher, images[2:])
    sup, ce, dice = supervised_loss(supervised_outputs, labels)
    mt = consistency_loss(unlabeled_outputs, teacher_outputs)
    params = tuple(student.parameters())
    sup_grad = snapshot(torch.autograd.grad(sup, params, retain_graph=True, allow_unused=True))
    mt_grad = snapshot(torch.autograd.grad(mt, params, retain_graph=True, allow_unused=True))
    loss = normalize_loss(sup)
    if weight > 0:
        loss = loss + weight * normalize_loss(mt)
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    for p in params:
        assert p.grad is not None and torch.isfinite(p.grad).all()
    result = snapshot(dict(supervised_outputs=supervised_outputs,
                           unlabeled_outputs=unlabeled_outputs, teacher_outputs=teacher_outputs,
                           losses=(sup, ce, dice, mt, loss), sup_grad=sup_grad, mt_grad=mt_grad,
                           combined_grad=tuple(p.grad for p in params),
                           student_buffers=bn_buffers(student), teacher_buffers=bn_buffers(teacher),
                           final_rng=rng_state(device)))
    # Student has two forwards per update, teacher one; eval has none.
    for name, before in initial_buffers.items():
        if name.endswith("num_batches_tracked"):
            assert_exact(result["student_buffers"][name], before + (2 if training else 0))
            assert_exact(result["teacher_buffers"][name],
                         initial_teacher_buffers[name] + (1 if training else 0))
    if not training:
        assert_exact(result["student_buffers"], initial_buffers)
        assert_exact(result["teacher_buffers"], initial_teacher_buffers)
        assert_exact(result["final_rng"], rng)
    elif student.has_dropout:
        key = "cuda" if device.startswith("cuda") else "torch"
        assert not torch.equal(result["final_rng"][key], rng[key])
    return result


class StageFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.use_deterministic_algorithms(True)

    def test_baseline_and_schema(self):
        self.assertEqual(hashlib.sha256((REFERENCE / "vnet_ds_baseline.py").read_bytes()).hexdigest(),
                         BASELINE_SHA256)
        old, new, upstream = new_model(Baseline), new_model(), new_model(Upstream)
        schema = lambda m: {k: (tuple(v.shape), str(v.dtype)) for k, v in m.state_dict().items()}
        self.assertEqual(schema(old), schema(new))
        self.assertEqual(schema(old), schema(upstream))
        self.assertEqual([(k, type(v).__name__) for k, v in old.named_modules()],
                         [(k, type(v).__name__) for k, v in new.named_modules()])
        new.load_state_dict(old.state_dict(), strict=True)
        OBSERVATIONS["schema"] = dict(state_entries=len(schema(new)),
                                      parameter_tensors=len(list(new.parameters())),
                                      parameter_count=sum(p.numel() for p in new.parameters()),
                                      baseline_sha256=BASELINE_SHA256)

    def test_default_and_upstream_outputs_train_eval(self):
        images = torch.randn(2, 1, 16, 32, 16)
        for training in (True, False):
            for dropout in (True, False):
                with self.subTest(training=training, dropout=dropout):
                    old = new_model(Baseline, dropout=dropout).train(training)
                    initial, rng = snapshot(old.state_dict()), rng_state()
                    restore_rng(rng)
                    expected = snapshot(old(images, deepsup=True))
                    expected_rng, expected_bn = rng_state(), bn_buffers(old)
                    variants = ((VNetDS, False, False), (VNetDS, True, False),
                                (VNetDS, False, True), (VNetDS, True, True),
                                (Upstream, True, False))
                    for cls, deepsup, features in variants:
                        model = new_model(cls, dropout=dropout).train(training)
                        model.load_state_dict(initial, strict=True)
                        restore_rng(rng)
                        kwargs = dict(deepsup=deepsup)
                        if features:
                            kwargs["return_features"] = True
                        actual = model(images, **kwargs)
                        if features or cls is Upstream:
                            actual = actual[0]
                        assert_exact(snapshot(actual), expected if deepsup else expected[-1])
                        assert_exact(rng_state(), expected_rng)
                        assert_exact(bn_buffers(model), expected_bn)

    def test_nine_stages_identity_shapes_no_recompute_and_gradients(self):
        for training in (True, False):
            model = new_model().train(training)
            calls, tensors, handles = Counter(), {}, []
            for name, module in model.named_modules():
                if name and not list(module.children()):
                    def count(m, args, output, name=name):
                        calls[name] += 1
                    handles.append(module.register_forward_hook(count))
            for name, stage in (("block_one", "E1"), ("block_two", "E2"),
                                ("block_three", "E3"), ("block_four", "E4")):
                def save(m, args, output, stage=stage):
                    tensors[stage] = output
                handles.append(getattr(model, name).register_forward_hook(save))
            for name, stage in (("out_conv_4", "D4"), ("out_conv_3", "D3"),
                                ("out_conv_2", "D2"), ("out_conv", "D1")):
                def save_input(m, args, stage=stage):
                    tensors[stage] = args[0]
                handles.append(getattr(model, name).register_forward_pre_hook(save_input))
            def save_bottleneck(m, args):
                tensors["E5"] = args[0]
            handles.append(model.block_five_up.register_forward_pre_hook(save_bottleneck))
            with mock.patch.object(model, "encoder", wraps=model.encoder) as encoder, \
                 mock.patch.object(model, "decoder", wraps=model.decoder) as decoder:
                outputs, features = model(torch.randn(2, 1, 32, 32, 32),
                                          deepsup=True, return_features=True)
                self.assertEqual(encoder.call_count, 1)
                self.assertEqual(decoder.call_count, 1)
            for handle in handles:
                handle.remove()
            for name, count in calls.items():
                self.assertEqual(count, 2 if name == "dropout" else 1, name)
            self.assertEqual([tuple(x.shape[2:]) for x in outputs],
                             [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)])
            blocks = ("block_one", "block_two", "block_three", "block_four", "block_five",
                      "block_five_up", "block_six_up", "block_seven_up", "block_nine")
            expected = [(16, 32), (32, 16), (64, 8), (128, 4), (256, 2),
                        (128, 4), (64, 8), (32, 16), (16, 32)]
            index = 0
            for group, stages in STAGES.items():
                self.assertEqual(tuple(features[group]), stages)
                for stage in stages:
                    tensor = get_stage_feature(features, group, stage)
                    channels, size = expected[index]
                    self.assertIs(tensor, tensors[stage])
                    self.assertEqual(tuple(tensor.shape), (2, channels, size, size, size))
                    self.assertTrue(tensor.requires_grad)
                    self.assertIsNotNone(tensor.grad_fn)
                    tensor.retain_grad()
                    # Each stage independently reaches its producing block.
                    weight = next(getattr(model, blocks[index]).parameters())
                    grad, = torch.autograd.grad(tensor.sum(), weight, retain_graph=True)
                    self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
                    index += 1
            loss, _, _ = supervised_loss(outputs, torch.randint(0, 2, (2, 32, 32, 32)))
            loss.backward()
            for stages in features.values():
                for tensor in stages.values():
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0)
            OBSERVATIONS[f"stages_{'train' if training else 'eval'}"] = describe_stage_features(features)
            OBSERVATIONS["leaf_module_calls_per_forward"] = dict(calls)
            with torch.no_grad():
                _, inference_features = model(torch.randn(2, 1, 16, 16, 16), return_features=True)
            self.assertTrue(all(not t.requires_grad for stages in inference_features.values()
                                for t in stages.values()))

    def test_loss_and_parameter_gradients_train_eval_warmup(self):
        images, labels = torch.randn(4, 1, 16, 16, 16), torch.randint(0, 2, (2, 16, 16, 16))
        initial, rng = snapshot(new_model().state_dict()), rng_state()
        teacher_initial = snapshot(initial)
        # Different targets exercise nonzero eval-mode consistency gradients too.
        teacher_initial["out_conv.bias"].add_(torch.tensor([-0.1, 0.1]))
        for training in (True, False):
            for weight in (0.0, 0.3):
                with self.subTest(training=training, weight=weight):
                    expected = compare_pipeline(Baseline, initial, images, labels, training, False,
                                                weight, rng, teacher_initial)
                    for flag in (False, True):
                        actual = compare_pipeline(VNetDS, initial, images, labels, training, flag,
                                                  weight, rng, teacher_initial)
                        assert_exact(actual, expected)
                    OBSERVATIONS[f"pipeline_{training}_{weight}"] = dict(
                        losses=[float(x) for x in expected["losses"]],
                        equality="exact atol=rtol=0", raw_supervised_gradient=True,
                        raw_consistency_gradient=True, normalized_combined_gradient=True)

    def test_all_six_original_best_checkpoints_strict(self):
        records = []
        directories = sorted((PROJECT / "Results/seed_42").iterdir())
        self.assertEqual(len(directories), 6)
        old, new = new_model(Baseline).eval(), new_model().eval()
        images = torch.randn(2, 1, 16, 16, 16)
        for directory in directories:
            path = directory / "best_model.pth"
            weights = torch.load(path, map_location="cpu", weights_only=True)
            for model in (old, new):
                result = model.load_state_dict(weights, strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
            with torch.no_grad():
                expected = old(images, deepsup=True)
                actual, features = new(images, deepsup=True, return_features=True)
            assert_exact(actual, expected)
            self.assertTrue(all(not t.requires_grad for stages in features.values() for t in stages.values()))
            records.append(dict(run=directory.name, path=str(path), strict=True,
                                state_entries=len(weights), predictions_exact=True))
        OBSERVATIONS["best_checkpoints"] = records

    def test_teacher_bn_dropout_ema_and_normalization(self):
        model = new_model().train()
        teacher = copy.deepcopy(model).requires_grad_(False).train()
        before = bn_buffers(teacher)
        rng = rng_state()
        with torch.no_grad():
            teacher(torch.randn(2, 1, 16, 16, 16), deepsup=True, return_features=True)
        after = bn_buffers(teacher)
        self.assertFalse(torch.equal(rng["torch"], rng_state()["torch"]))
        self.assertTrue(teacher.training and teacher.dropout.training)
        for k in before:
            if k.endswith("num_batches_tracked"):
                assert_exact(after[k], before[k] + 1)
        with torch.no_grad():
            next(model.parameters()).add_(0.125)
        update_ema(model, teacher, 0.99, 0)
        assert_exact(tuple(model.parameters()), tuple(teacher.parameters()))
        assert_exact(bn_buffers(teacher), after)
        old_params = snapshot(tuple(teacher.parameters()))
        with torch.no_grad():
            next(model.parameters()).add_(0.125)
        update_ema(model, teacher, 0.99, 9)
        for previous, student, target in zip(old_params, model.parameters(), teacher.parameters()):
            expected = previous.mul(0.9).add(student.detach(), alpha=0.1)
            torch.testing.assert_close(target, expected, atol=1e-7, rtol=1e-6)
        assert_exact(bn_buffers(teacher), after)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        for value in (0.0, 1e-10, 2.0):
            loss = torch.tensor(value, requires_grad=True)
            output = normalize_loss(loss)
            output.backward()
            torch.testing.assert_close(loss.grad, torch.tensor(1.0 / max(value, 1e-8)))
            torch.testing.assert_close(output, torch.tensor(value / max(value, 1e-8)))

    def test_helpers_reject_bad_inputs_and_preserve_tensor(self):
        tensor = torch.randn(2, 16, 4, 5, 6, requires_grad=True)
        features = {"encoder": {"E1": tensor}}
        rng = rng_state()
        self.assertIs(get_stage_feature(features, "encoder", "E1"), tensor)
        info = stage_feature_info(features, "encoder", "E1")
        self.assertEqual(info["channels"], 16)
        self.assertEqual(info["spatial_shape"], (4, 5, 6))
        self.assertTrue(info["requires_grad"])
        assert_exact(rng_state(), rng)
        cases = [(ValueError, features, "bad", "E1"),
                 (ValueError, features, "encoder", "D1"),
                 (KeyError, {}, "encoder", "E1"),
                 (KeyError, features, "encoder", "E2"),
                 (TypeError, [], "encoder", "E1"),
                 (TypeError, {"encoder": []}, "encoder", "E1"),
                 (TypeError, {"encoder": {"E1": 1}}, "encoder", "E1"),
                 (ValueError, {"encoder": {"E1": tensor[0]}}, "encoder", "E1")]
        for error, candidate, group, stage in cases:
            with self.assertRaises(error):
                get_stage_feature(candidate, group, stage)


def gpu_smoke(args):
    if not torch.cuda.is_available():
        return dict(status="SKIPPED", reason="CUDA unavailable in this execution environment")
    from data import SPECS, VolumeDataset, split_manifest
    device = f"cuda:{args.gpu_index}"
    torch.cuda.set_device(args.gpu_index)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Match train.py. PyTorch CUDA NLL forward rejects global strict determinism.
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    report = dict(status="PASS", device=device, device_name=torch.cuda.get_device_name(device),
                  torch=torch.__version__, cudnn_deterministic=True, cudnn_benchmark=False,
                  global_strict_determinism=False, datasets={})
    for dataset in ("LA", "Pancreas", "BraTS19"):
        spec = SPECS[dataset]
        root = args.data_root / spec["directory"]
        volume_dataset = VolumeDataset(root, dataset, spec["patch"])
        manifest = split_manifest(root, dataset, spec["labels"])
        indices = [0, 1, spec["labels"], spec["labels"] + 1]
        batch = [volume_dataset[i] for i in indices]
        images = torch.stack([x["image"] for x in batch]).to(device)
        labels = torch.stack([x["label"] for x in batch[:2]]).to(device)
        initial = torch.load(PROJECT / f"Results/seed_42/{dataset}_{spec['labels']}l/best_model.pth",
                             map_location="cpu", weights_only=True)
        teacher_initial = torch.load(
            PROJECT / f"Results/seed_42/{dataset}_{spec['labels']}l/teacher_at_best_student.pth",
            map_location="cpu", weights_only=True)
        rng = rng_state(device)
        torch.cuda.reset_peak_memory_stats(device)
        rows = {}
        for training in (True, False):
            expected = compare_pipeline(Baseline, initial, images, labels, training, False,
                                        0.3, rng, teacher_initial)
            repeated = compare_pipeline(Baseline, initial, images, labels, training, False,
                                        0.3, rng, teacher_initial)
            comparisons = {"baseline_repeat": compare_gpu_results(repeated, expected)}
            del repeated
            for flag in (False, True):
                actual = compare_pipeline(VNetDS, initial, images, labels, training, flag,
                                          0.3, rng, teacher_initial)
                comparisons["features" if flag else "default"] = compare_gpu_results(actual, expected)
                del actual
            rows["train" if training else "eval"] = dict(
                losses=[float(x) for x in expected["losses"]], comparisons=comparisons)
            del expected
        model = new_model(device=device).train()
        model.load_state_dict(initial, strict=True)
        _, features = model(images[:2], deepsup=True, return_features=True)
        metadata = describe_stage_features(features)
        report["datasets"][dataset] = dict(
            batch_shape=list(images.shape), case_ids=[volume_dataset.ids[i] for i in indices],
            labeled_ids=manifest["labeled_ids"][:2], modes=rows, stages=metadata,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            optimizer_steps=0, checkpoint_writes=0)
        print(f"PASS GPU real batch {dataset}: {list(images.shape)} train/eval raw losses, "
              "outputs/raw gradients/RNG/BN exact; CE/normalized gradients checked with "
              "rtol=1e-6 atol=1e-7 and unchanged-Baseline repeat control", flush=True)
        del model, features, images, labels, initial, teacher_initial
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parent / "Dataset")
    args = parser.parse_args()
    start = time.monotonic()
    if args.gpu_smoke:
        report = gpu_smoke(args)
        successful = report["status"] != "FAIL"
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(StageFeaturesTests)
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
        successful = result.wasSuccessful()
        report = dict(status="PASS" if successful else "FAIL", tests_run=result.testsRun,
                      failures=[(str(t), s) for t, s in result.failures],
                      errors=[(str(t), s) for t, s in result.errors],
                      skipped=[(str(t), reason) for t, reason in result.skipped],
                      torch=torch.__version__, observations=OBSERVATIONS)
    report["elapsed_seconds"] = time.monotonic() - start
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
