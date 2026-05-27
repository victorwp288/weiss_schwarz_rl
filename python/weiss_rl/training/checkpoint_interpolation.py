from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def interpolate_model_state_dicts(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    second_weight: float,
) -> dict[str, Any]:
    """Return a state dict linearly interpolated between two compatible models."""

    alpha = float(second_weight)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("second_weight must be in [0, 1]")
    first_keys = set(first)
    second_keys = set(second)
    if first_keys != second_keys:
        missing = sorted(first_keys.symmetric_difference(second_keys))[:5]
        raise ValueError(f"state dict keys do not match; first mismatches: {missing}")

    mixed: dict[str, Any] = {}
    for key in sorted(first_keys):
        left = first[key]
        right = second[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            if left != right:
                raise ValueError(f"non-tensor state value differs for key {key!r}")
            mixed[key] = left
            continue
        if left.shape != right.shape:
            raise ValueError(f"tensor shape mismatch for key {key!r}: {tuple(left.shape)} != {tuple(right.shape)}")
        if left.dtype != right.dtype:
            raise ValueError(f"tensor dtype mismatch for key {key!r}: {left.dtype} != {right.dtype}")
        if left.is_floating_point() or left.is_complex():
            mixed[key] = torch.lerp(left.detach().cpu(), right.detach().cpu(), alpha)
        else:
            if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                raise ValueError(f"non-floating tensor differs for key {key!r}")
            mixed[key] = left.detach().cpu().clone()
    return mixed


__all__ = ["interpolate_model_state_dicts"]
