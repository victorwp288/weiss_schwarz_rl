from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from weiss_rl.learners.impala.loss_batch_inputs import resolve_impala_loss_batch_inputs


def test_resolve_impala_loss_batch_inputs_prefers_compiled_forward_model_and_expected_shape() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.ones((2, 1), dtype=torch.long)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    model = object()
    compiled_model = object()
    batch = {
        "vtrace_result": "vtrace",
        "obs": "raw_obs",
        "actions": "raw_actions",
    }
    calls: list[tuple[str, Any]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        calls.append(("batch_value", key))
        return source_batch.get(key)

    learner = SimpleNamespace(
        model=model,
        compiled_model=compiled_model,
        _require_obs=lambda value: calls.append(("require_obs", value)) or obs,
        _require_actions=lambda value, *, expected_shape: (
            calls.append(("require_actions", (value, expected_shape))) or actions
        ),
        _resolve_packed_legal_actions_with_meta=lambda source_batch, *, expected_shape: (
            calls.append(("packed_legal", (source_batch, expected_shape))) or packed_legal
        ),
    )

    result = resolve_impala_loss_batch_inputs(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
    )

    assert result.vtrace_result == "vtrace"
    assert result.obs is obs
    assert result.actions is actions
    assert result.packed_legal is packed_legal
    assert result.forward_model is compiled_model
    assert calls == [
        ("batch_value", "vtrace_result"),
        ("batch_value", "obs"),
        ("require_obs", "raw_obs"),
        ("batch_value", "actions"),
        ("require_actions", ("raw_actions", torch.Size((2, 1)))),
        ("packed_legal", (batch, torch.Size((2, 1)))),
    ]


def test_resolve_impala_loss_batch_inputs_falls_back_to_base_model() -> None:
    obs = torch.zeros((1, 1, 2), dtype=torch.float32)
    actions = torch.zeros((1, 1), dtype=torch.long)
    model = object()
    learner = SimpleNamespace(
        model=model,
        compiled_model=None,
        _require_obs=lambda _value: obs,
        _require_actions=lambda _value, *, expected_shape: actions,
        _resolve_packed_legal_actions_with_meta=lambda _batch, *, expected_shape: None,
    )

    result = resolve_impala_loss_batch_inputs(
        learner=learner,
        batch={"obs": object(), "actions": object()},
        batch_value=lambda source_batch, key: source_batch.get(key),
    )

    assert result.forward_model is model
    assert result.packed_legal is None
