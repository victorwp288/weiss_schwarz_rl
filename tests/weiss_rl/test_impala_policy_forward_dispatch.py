from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from weiss_rl.learners.impala.losses.loss_policy_forward import evaluate_impala_policy_forward


def test_evaluate_impala_policy_forward_uses_factorized_path_without_dense_forward() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.as_tensor([[0], [1]], dtype=torch.long)
    loss_mask = torch.ones((2, 1), dtype=torch.float32)
    retention_active = torch.as_tensor([[False], [True]], dtype=torch.bool)
    original_packed = (torch.as_tensor([0, 1]), torch.as_tensor([0, 2]), None)
    resolved_packed = (torch.as_tensor([1]), torch.as_tensor([0, 1]), None)
    factorized_result = SimpleNamespace(values=torch.as_tensor([[0.25], [0.5]], dtype=torch.float32))
    calls: list[tuple[str, Any]] = []

    def should_use_factorized(forward_model: object, *, packed_legal: object) -> bool:
        calls.append(("should_use_factorized", (forward_model, packed_legal)))
        return True

    def evaluate_factorized(
        source_batch: object,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        extra_active_mask: torch.Tensor | None,
    ) -> tuple[SimpleNamespace, tuple[torch.Tensor, torch.Tensor, None]]:
        calls.append(("evaluate_factorized", (source_batch, obs, actions, extra_active_mask)))
        return factorized_result, resolved_packed

    learner = SimpleNamespace(
        _should_use_factorized_legal_policy=should_use_factorized,
        _evaluate_factorized_time_major=evaluate_factorized,
    )
    forward_model = object()
    batch = object()

    result = evaluate_impala_policy_forward(
        learner=learner,
        batch=batch,
        batch_value=lambda _source, key: pytest.fail(f"unexpected batch_value({key})"),
        forward_model=forward_model,
        obs=obs,
        actions=actions,
        packed_legal=original_packed,
        loss_mask=loss_mask,
        reset_before_step=None,
        trajectory_retention_active=retention_active,
        restrict_packed_policy_rows=True,
    )

    assert result.factorized_result is factorized_result
    assert result.packed_legal is resolved_packed
    assert result.logits is None
    assert result.packed_logits is None
    assert result.values is factorized_result.values
    assert result.forward_observation_context is None
    assert calls == [
        ("should_use_factorized", (forward_model, original_packed)),
        ("evaluate_factorized", (batch, obs, actions, retention_active)),
    ]


def test_evaluate_impala_policy_forward_forwards_dense_kwargs_and_restricts_rows() -> None:
    obs = torch.zeros((3, 1, 2), dtype=torch.float32)
    actions = torch.as_tensor([[0], [1], [0]], dtype=torch.long)
    loss_mask = torch.as_tensor([[1.0], [0.0], [0.0]], dtype=torch.float32)
    retention_active = torch.as_tensor([[False], [True], [False]], dtype=torch.bool)
    reset_before_step = torch.as_tensor([[False], [True], [False]], dtype=torch.bool)
    logits = torch.zeros((3, 1, 4), dtype=torch.float32)
    packed_logits = torch.zeros((5,), dtype=torch.float32)
    values = torch.as_tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    observation_context = {"rows": obs.reshape(-1, obs.shape[-1])}
    batch = {
        "initial_hidden_state": "hidden",
        "to_play_seat": "seat",
        "actor": "actor",
        "legal_actions": "legal",
        "opponent_context_index": "opponent",
    }
    calls: list[tuple[str, Any]] = []

    def batch_value(source_batch: Mapping[str, object], key: str) -> object:
        calls.append(("batch_value", key))
        return source_batch[key]

    def forward_time_major(
        forward_obs: torch.Tensor,
        *,
        initial_hidden_state: object,
        to_play_seat: object,
        actor: object,
        legal_actions: object,
        policy_train_mask: torch.Tensor | None,
        reset_before_step: torch.Tensor | None,
        opponent_context_index: object,
    ) -> SimpleNamespace:
        calls.append(
            (
                "forward",
                (
                    forward_obs,
                    initial_hidden_state,
                    to_play_seat,
                    actor,
                    legal_actions,
                    policy_train_mask,
                    reset_before_step,
                    opponent_context_index,
                ),
            )
        )
        return SimpleNamespace(
            logits=logits,
            packed_logits=packed_logits,
            values=values,
            observation_context=observation_context,
        )

    learner = SimpleNamespace(
        _should_use_factorized_legal_policy=lambda _forward_model, *, packed_legal: False,
        _forward_time_major=forward_time_major,
    )
    packed_legal = (torch.as_tensor([0, 1]), torch.as_tensor([0, 2]), None)

    result = evaluate_impala_policy_forward(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
        forward_model=object(),
        obs=obs,
        actions=actions,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
        reset_before_step=reset_before_step,
        trajectory_retention_active=retention_active,
        restrict_packed_policy_rows=True,
    )

    forward_call = calls[-1]
    assert forward_call[0] == "forward"
    forwarded_mask = forward_call[1][5]
    assert isinstance(forwarded_mask, torch.Tensor)
    assert forwarded_mask.dtype == loss_mask.dtype
    assert forwarded_mask.tolist() == [[1.0], [1.0], [0.0]]
    assert forward_call[1][0] is obs
    assert forward_call[1][1:5] == ("hidden", "seat", "actor", "legal")
    assert forward_call[1][6] is reset_before_step
    assert forward_call[1][7] == "opponent"
    assert calls[:5] == [
        ("batch_value", "initial_hidden_state"),
        ("batch_value", "to_play_seat"),
        ("batch_value", "actor"),
        ("batch_value", "legal_actions"),
        ("batch_value", "opponent_context_index"),
    ]
    assert result.factorized_result is None
    assert result.packed_legal is packed_legal
    assert result.logits is logits
    assert result.packed_logits is packed_logits
    assert result.values is values
    assert result.forward_observation_context is observation_context
