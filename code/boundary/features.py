"""Validate and inspect live [batch, channels, H, W, D] stage tensors.

No tensor copies, detach, forward calls, hooks, or model state are introduced.
Do not modify tensors returned by get_stage_feature in place.
"""

from collections.abc import Mapping
from types import MappingProxyType

import torch


STAGES = MappingProxyType({
    "encoder": ("E1", "E2", "E3", "E4", "E5"),
    "decoder": ("D4", "D3", "D2", "D1"),
})
SOURCES = MappingProxyType({
    "E1": "block_one",
    "E2": "block_two",
    "E3": "block_three",
    "E4": "block_four",
    "E5": "block_five -> optional dropout",
    "D4": "block_five_up + E4 -> out_conv_4 input",
    "D3": "block_six -> block_six_up + E3 -> out_conv_3 input",
    "D2": "block_seven -> block_seven_up + E2 -> out_conv_2 input",
    "D1": "block_nine -> optional dropout -> out_conv input",
})


def get_stage_feature(features, group, stage):
    """Return the original tensor, rejecting invalid names and malformed inputs."""
    if group not in STAGES:
        raise ValueError(f"Unknown feature group {group!r}; expected {tuple(STAGES)}")
    if stage not in STAGES[group]:
        raise ValueError(f"Unknown {group} stage {stage!r}; expected {STAGES[group]}")
    if not isinstance(features, Mapping):
        raise TypeError("features must be the nested mapping returned by VNetDS")
    if group not in features:
        raise KeyError(f"Missing feature group {group!r}")
    stages = features[group]
    if not isinstance(stages, Mapping):
        raise TypeError(f"features[{group!r}] must be a mapping")
    if stage not in stages:
        raise KeyError(f"Missing stage {group}/{stage}")
    tensor = stages[stage]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{group}/{stage} must be a torch.Tensor")
    if tensor.ndim != 5 or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{group}/{stage} must have nonempty [B,C,H,W,D] shape")
    return tensor


def stage_feature_info(features, group, stage):
    """Return shape/gradient metadata without retaining another tensor reference."""
    tensor = get_stage_feature(features, group, stage)
    return dict(stage=stage, group=group, channels=tensor.shape[1],
                spatial_shape=tuple(tensor.shape[2:]), batch_size=tensor.shape[0],
                requires_grad=tensor.requires_grad, source=SOURCES[stage],
                dtype=str(tensor.dtype), device=str(tensor.device))


def describe_stage_features(features):
    """Describe all nine stages in encoder E1..E5, decoder D4..D1 order."""
    return {group: {stage: stage_feature_info(features, group, stage)
                    for stage in stages} for group, stages in STAGES.items()}
