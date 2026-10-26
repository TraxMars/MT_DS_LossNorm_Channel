"""Frozen-model parameter-gradient diagnostics. No optimizer or training entry."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from objectives import consistency_loss, normalize_loss, supervised_loss

SCALES = ("D4", "D3", "D2", "D1")
STAGES = ("E1", "E2", "E3", "E4", "E5", "D4", "D3", "D2", "D1")


@dataclass
class StageParameter:
    stage: str
    name: str
    module: nn.Module
    output_axis: int
    reachable: tuple
    relationship: str

    @property
    def parameter(self):
        return self.module.weight

    @property
    def channels(self):
        return self.module.out_channels

    def channel_vectors(self, gradient):
        if gradient is None:
            return None
        if gradient.shape != self.parameter.shape:
            raise AssertionError(f"Wrong parameter gradient shape: {self.name}")
        return gradient.movedim(self.output_axis, 0).reshape(self.channels, -1)

    def describe(self):
        return dict(stage=self.stage, parameter=self.name,
                    module_type=type(self.module).__name__, weight_shape=list(self.parameter.shape),
                    output_channel_axis=self.output_axis, channels=self.channels,
                    values_per_channel=self.parameter.numel() // self.channels,
                    reachable_scales=list(self.reachable), relationship=self.relationship)


def stage_parameters(model):
    """Discover actual convolution modules by type/role, never fixed list indices."""
    result = {}
    for stage, block in zip(STAGES[:5], ("block_one", "block_two", "block_three", "block_four", "block_five")):
        convolutions = [(name, module) for name, module in getattr(model, block).named_modules()
                        if isinstance(module, nn.Conv3d)]
        name, module = convolutions[-1]
        result[stage] = StageParameter(stage, f"{block}.{name}.weight", module, 0, SCALES,
                                       "Last Conv3d in encoder block; followed by BN/ReLU and optional E5 dropout")
    for stage, block, first_scale in (("D4", "block_five_up", 0), ("D3", "block_six_up", 1),
                                      ("D2", "block_seven_up", 2)):
        found = [(name, module) for name, module in getattr(model, block).named_modules()
                 if isinstance(module, nn.ConvTranspose3d)]
        if len(found) != 1 or found[0][1].groups != 1:
            raise ValueError("Expected one ungrouped ConvTranspose3d in each upsampling block")
        name, module = found[0]
        result[stage] = StageParameter(stage, f"{block}.{name}.weight", module, 1, SCALES[first_scale:],
                                       "Upsampling branch before BN/ReLU and skip addition; does not parameterize the skip branch")
    found = [(name, module) for name, module in model.block_nine.named_modules()
             if isinstance(module, nn.Conv3d)]
    name, module = found[-1]
    result["D1"] = StageParameter("D1", f"block_nine.{name}.weight", module, 0, ("D1",),
                                   "Final feature-producing convolution before BN/ReLU/dropout and out_conv")
    for spec in result.values():
        if spec.module.groups != 1:
            raise ValueError("Grouped convolution output-channel mapping needs a separate definition")
        if dict(model.named_parameters())[spec.name] is not spec.parameter:
            raise AssertionError("Parameter identity mismatch")
    return result


def independent_losses(outputs, labels=None, teacher_outputs=None):
    """Reuse original loss functions at each head; preserve aggregate LossNorm."""
    if len(outputs) != 4:
        raise ValueError("Four logits heads are required")
    if (labels is None) == (teacher_outputs is None):
        raise ValueError("Provide labeled GT or teacher logits, exclusively")
    if labels is not None:
        terms = [supervised_loss([output], labels) for output in outputs]
        losses = [term[0] for term in terms]
        aggregate = supervised_loss(outputs, labels)[0]
        details = [dict(ce=float(term[1].detach()), dice=float(term[2].detach())) for term in terms]
    else:
        losses = [consistency_loss([output], [teacher]) for output, teacher in zip(outputs, teacher_outputs)]
        aggregate = consistency_loss(outputs, teacher_outputs)
        details = [{} for _ in losses]
    torch.testing.assert_close(torch.stack(losses).mean(), aggregate, atol=1e-7, rtol=1e-6)
    return losses, aggregate, details


def parameter_gradients(losses, aggregate, mapping, eps=1e-8, weight=1.0):
    """Use autograd.grad exclusively, preserving every leaf's .grad state.

    A normalized *scale contribution* is weight*g_s/(4*max(aggregate,eps)).
    Applying normalize_loss independently to each head would change the baseline.
    """
    params = tuple(spec.parameter for spec in mapping.values())
    before = [None if p.grad is None else p.grad.detach().clone() for p in params]
    raw, checks = {stage: [] for stage in mapping}, []
    for scale, loss in zip(SCALES, losses):
        gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        for (stage, spec), gradient in zip(mapping.items(), gradients):
            reachable = scale in spec.reachable
            if (gradient is not None) != reachable:
                raise AssertionError(f"Reachability mismatch {stage}/{scale}")
            vectors = spec.channel_vectors(gradient)
            raw[stage].append(None if vectors is None else vectors.detach().cpu().numpy().copy())
    full_raw = torch.autograd.grad(aggregate, params, retain_graph=True, allow_unused=True)
    normalized = torch.autograd.grad(weight * normalize_loss(aggregate, eps), params, allow_unused=True)
    denominator = max(float(aggregate.detach()), eps)
    factor = weight / (4 * denominator)
    for (stage, spec), expected, actual, prior in zip(mapping.items(), full_raw, normalized, before):
        if (prior is None) != (spec.parameter.grad is None):
            raise AssertionError("autograd.grad changed a .grad slot")
        if prior is not None and not torch.equal(prior, spec.parameter.grad):
            raise AssertionError("autograd.grad changed existing .grad values")
        summed = sum(item for item in raw[stage] if item is not None) / 4
        expected_vectors = spec.channel_vectors(expected).detach().cpu().numpy()
        normalized_vectors = spec.channel_vectors(actual).detach().cpu().numpy()
        raw_error = np.linalg.norm((summed - expected_vectors).astype(np.float64))
        raw_reference = np.linalg.norm(expected_vectors.astype(np.float64))
        norm_error = np.linalg.norm((summed * (weight / denominator) - normalized_vectors).astype(np.float64))
        norm_reference = np.linalg.norm(normalized_vectors.astype(np.float64))
        # Separate head backpropagations sum in a different floating-point order
        # from one aggregate backpropagation. Use vector-relative reconstruction
        # error rather than relative error of near-zero individual coefficients.
        raw_relative = raw_error / max(raw_reference, 1e-30)
        normalized_relative = norm_error / max(norm_reference, 1e-30)
        if raw_error > 1e-9 and raw_relative > 1e-4:
            raise AssertionError(f"Raw gradient reconstruction {stage}: relative L2 error {raw_relative}")
        if norm_error > 1e-9 and normalized_relative > 1e-4:
            raise AssertionError(f"LossNorm gradient reconstruction {stage}: relative L2 error {normalized_relative}")
        checks.append(dict(stage=stage, raw_max_abs_error=float(np.max(np.abs(summed - expected_vectors))),
                           raw_relative_l2_error=float(raw_relative), normalized_relative_l2_error=float(normalized_relative),
                           normalized_max_abs_error=float(np.max(np.abs(
                               summed * (weight / denominator) - normalized_vectors)))))
    return raw, dict(lossnorm_denominator=denominator, normalized_scale_factor=factor,
                     normalized_scalar=float(weight * normalize_loss(aggregate.detach(), eps)),
                     gradient_reconstruction=checks)


def read_image_only(path, access_log, role, case):
    """Unlabeled loader intentionally has no label lookup or label return value."""
    with h5py.File(path, "r") as handle:
        image = handle["image"][:].astype(np.float32)
    access_log.append(dict(role=role, case=case, path=str(path), key="image"))
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 3D image")
    return image


def read_labeled_case(path, access_log, case):
    image = read_image_only(path, access_log, "labeled", case)
    with h5py.File(path, "r") as handle:
        label = handle["label"][:].astype(np.int64)
    access_log.append(dict(role="labeled", case=case, path=str(path), key="label"))
    if label.shape != image.shape or not np.isin(label, [0, 1]).all():
        raise ValueError("Labeled GT must be matching binary 3D data")
    return image, label


def deterministic_crop(image, label, patch, seed, case, repeat):
    """No foreground search, label-based crop choice, augmentation or U GT."""
    padding = [(max((p - s) // 2 + 3, 0),) * 2 for s, p in zip(image.shape, patch)] \
        if any(s <= p for s, p in zip(image.shape, patch)) else [(0, 0)] * 3
    padded = np.pad(image, padding)
    seed_bytes = hashlib.sha256(f"{seed}:{case}:{repeat}".encode()).digest()[:8]
    generator = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    starts = [int(generator.integers(0, s - p + 1)) for s, p in zip(padded.shape, patch)]
    slices = tuple(slice(s, s + p) for s, p in zip(starts, patch))
    x = torch.from_numpy(np.ascontiguousarray(padded[slices][None, None]))
    y = None if label is None else torch.from_numpy(np.ascontiguousarray(np.pad(label, padding)[slices][None]))
    return x, y, dict(original_shape=list(image.shape), padding=padding, starts=starts, patch=list(patch))


def gradient_validity(vectors, absolute_floor=1e-12, relative_floor=1e-6):
    """Per-channel status; undefined and negligible vectors never become zero scores."""
    if vectors is None:
        raise ValueError("Handle graph-unreachable gradients before calling gradient_validity")
    norms = np.sqrt(np.sum(np.asarray(vectors, dtype=np.float64) ** 2, axis=-1))
    finite = np.isfinite(vectors).all(axis=-1) & np.isfinite(norms)
    maximum = np.max(np.where(finite, norms, 0), axis=-1, keepdims=True)
    floor = np.maximum(absolute_floor, relative_floor * maximum)
    statuses = np.full(norms.shape, "valid", dtype="U20")
    statuses[~finite] = "nonfinite"
    statuses[finite & (norms == 0)] = "exact_zero"
    statuses[finite & (norms > 0) & (norms <= floor)] = "below_threshold"
    return norms, statuses, np.broadcast_to(floor, norms.shape)


def signed_cosine(left, right, absolute_floor=1e-12, relative_floor=1e-6):
    """Same parameter coordinates only. No spatial feature-gradient comparison."""
    ln, ls, _ = gradient_validity(left, absolute_floor, relative_floor)
    rn, rs, _ = gradient_validity(right, absolute_floor, relative_floor)
    valid = (ls == "valid") & (rs == "valid")
    scores = np.full(ln.shape, np.nan)
    dot = np.sum(np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64), axis=-1)
    np.divide(dot, ln * rn, out=scores, where=valid)
    return np.clip(scores, -1, 1), valid, ls, rs


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()
