from __future__ import annotations

import pytest
import torch

from weiss_rl.training.checkpoint_interpolation import interpolate_model_state_dicts


def test_interpolate_model_state_dicts_lerps_float_tensors_and_copies_equal_int_tensors() -> None:
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor([7], dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor([7], dtype=torch.int64),
    }

    mixed = interpolate_model_state_dicts(first, second, second_weight=0.25)

    assert mixed["weight"].tolist() == pytest.approx([2.0, 4.0])
    assert mixed["counter"].tolist() == [7]
    assert mixed["counter"] is not first["counter"]


def test_interpolate_model_state_dicts_rejects_incompatible_keys() -> None:
    with pytest.raises(ValueError, match="state dict keys do not match"):
        interpolate_model_state_dicts(
            {"left": torch.tensor([1.0])},
            {"right": torch.tensor([1.0])},
            second_weight=0.5,
        )


def test_interpolate_model_state_dicts_rejects_changed_nonfloating_tensor() -> None:
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        interpolate_model_state_dicts(
            {"counter": torch.tensor([1], dtype=torch.int64)},
            {"counter": torch.tensor([2], dtype=torch.int64)},
            second_weight=0.5,
        )
