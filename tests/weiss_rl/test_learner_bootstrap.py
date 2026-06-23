from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from weiss_rl.learners.bootstrap import (
    current_model_bootstrap_value,
    has_raw_vtrace_inputs,
    resolve_vtrace_bootstrap_value,
)


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _reference() -> Tensor:
    return torch.zeros((), dtype=torch.float64)


class _ValueSeatAwareModel:
    def value_seat_aware(self, obs: Tensor, actor: Tensor, hidden: Tensor) -> Tensor:
        return obs[:, 0] + actor.to(dtype=obs.dtype) + hidden[:, 0, 0]


class _ForwardSeatAwareModel:
    def forward_seat_aware(self, obs: Tensor, actor: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return obs.new_zeros((obs.shape[0], 2)), obs[:, 0] - actor.to(dtype=obs.dtype), hidden


class _UnsupportedModel:
    pass


def _bootstrap_batch() -> dict[str, Any]:
    return {
        "bootstrap_obs": torch.tensor([[2.0, 0.0], [4.0, 1.0]], dtype=torch.float32),
        "bootstrap_actor": torch.tensor([1, 0], dtype=torch.long),
        "final_hidden_state": torch.tensor([[[0.5], [0.0]], [[1.5], [0.0]]], dtype=torch.float32),
        "bootstrap_value": torch.tensor([123.0, 456.0], dtype=torch.float32),
    }


def test_has_raw_vtrace_inputs_accepts_stored_or_current_model_bootstrap_fields() -> None:
    common = {
        "rewards": object(),
        "discounts": object(),
        "behavior_logp": object(),
    }

    assert not has_raw_vtrace_inputs({"rewards": object()}, batch_value=_batch_value)
    assert has_raw_vtrace_inputs({**common, "bootstrap_value": object()}, batch_value=_batch_value)
    assert has_raw_vtrace_inputs(
        {
            **common,
            "bootstrap_obs": object(),
            "bootstrap_actor": object(),
            "final_hidden_state": object(),
        },
        batch_value=_batch_value,
    )
    assert not has_raw_vtrace_inputs({**common, "bootstrap_obs": object()}, batch_value=_batch_value)


def test_current_model_bootstrap_value_prefers_value_seat_aware_path() -> None:
    values = current_model_bootstrap_value(
        _bootstrap_batch(),
        batch_size=2,
        like=torch.zeros((2,), dtype=torch.float64),
        model=_ValueSeatAwareModel(),
        compiled_model=None,
        reference_parameter=_reference,
        batch_value=_batch_value,
    )

    assert values is not None
    torch.testing.assert_close(values, torch.tensor([3.5, 5.5], dtype=torch.float64))


def test_current_model_bootstrap_value_uses_forward_fallback() -> None:
    values = current_model_bootstrap_value(
        _bootstrap_batch(),
        batch_size=2,
        like=torch.zeros((2,), dtype=torch.float64),
        model=_ForwardSeatAwareModel(),
        compiled_model=None,
        reference_parameter=_reference,
        batch_value=_batch_value,
    )

    assert values is not None
    torch.testing.assert_close(values, torch.tensor([1.0, 4.0], dtype=torch.float64))


def test_current_model_bootstrap_value_returns_none_for_missing_or_unsupported_inputs() -> None:
    assert (
        current_model_bootstrap_value(
            {},
            batch_size=2,
            like=torch.zeros((2,)),
            model=_ValueSeatAwareModel(),
            compiled_model=None,
            reference_parameter=_reference,
            batch_value=_batch_value,
        )
        is None
    )
    assert (
        current_model_bootstrap_value(
            _bootstrap_batch(),
            batch_size=2,
            like=torch.zeros((2,)),
            model=None,
            compiled_model=None,
            reference_parameter=_reference,
            batch_value=_batch_value,
        )
        is None
    )
    assert (
        current_model_bootstrap_value(
            _bootstrap_batch(),
            batch_size=2,
            like=torch.zeros((2,)),
            model=_UnsupportedModel(),
            compiled_model=None,
            reference_parameter=_reference,
            batch_value=_batch_value,
        )
        is None
    )
    bad_hidden = {**_bootstrap_batch(), "final_hidden_state": torch.zeros((2, 2))}
    assert (
        current_model_bootstrap_value(
            bad_hidden,
            batch_size=2,
            like=torch.zeros((2,)),
            model=_ValueSeatAwareModel(),
            compiled_model=None,
            reference_parameter=_reference,
            batch_value=_batch_value,
        )
        is None
    )


def test_current_model_bootstrap_value_validates_bootstrap_shapes() -> None:
    with pytest.raises(ValueError, match=r"bootstrap_obs must have shape \(2, observation\)"):
        current_model_bootstrap_value(
            {**_bootstrap_batch(), "bootstrap_obs": torch.zeros((1, 2))},
            batch_size=2,
            like=torch.zeros((2,)),
            model=_ValueSeatAwareModel(),
            compiled_model=None,
            reference_parameter=_reference,
            batch_value=_batch_value,
        )
    with pytest.raises(ValueError, match="final_hidden_state seat mismatch"):
        current_model_bootstrap_value(
            {**_bootstrap_batch(), "final_hidden_state": torch.zeros((2, 3, 1))},
            batch_size=2,
            like=torch.zeros((2,)),
            model=_ValueSeatAwareModel(),
            compiled_model=None,
            reference_parameter=_reference,
            batch_value=_batch_value,
        )


def test_resolve_vtrace_bootstrap_value_prefers_current_model_over_stored_values() -> None:
    values = resolve_vtrace_bootstrap_value(
        _bootstrap_batch(),
        batch_size=2,
        like=torch.zeros((2,), dtype=torch.float64),
        model=_ValueSeatAwareModel(),
        compiled_model=None,
        reference_parameter=_reference,
        batch_value=_batch_value,
    )

    torch.testing.assert_close(values, torch.tensor([3.5, 5.5], dtype=torch.float64))


def test_resolve_vtrace_bootstrap_value_falls_back_to_stored_batch_value_and_validates_shape() -> None:
    values = resolve_vtrace_bootstrap_value(
        {"bootstrap_value": torch.tensor([1.0, 2.0], dtype=torch.float32)},
        batch_size=2,
        like=torch.zeros((2,), dtype=torch.float64),
        model=_UnsupportedModel(),
        compiled_model=None,
        reference_parameter=_reference,
        batch_value=_batch_value,
    )

    torch.testing.assert_close(values, torch.tensor([1.0, 2.0], dtype=torch.float64))
    with pytest.raises(ValueError, match=r"bootstrap_value must have shape \(2,\), got \(1,\)"):
        resolve_vtrace_bootstrap_value(
            {"bootstrap_value": torch.tensor([1.0], dtype=torch.float32)},
            batch_size=2,
            like=torch.zeros((2,), dtype=torch.float64),
            model=_UnsupportedModel(),
            compiled_model=None,
            reference_parameter=_reference,
            batch_value=_batch_value,
        )
