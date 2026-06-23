from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import weiss_rl.learners.impala.loss_teacher_stage as impala_loss_teacher_stage
from weiss_rl.learners.impala.loss_teacher_stage import apply_impala_teacher_auxiliary_stage


def test_apply_impala_teacher_auxiliary_stage_maps_loss_inputs_and_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_loss = torch.tensor(2.0, dtype=torch.float32)
    resolved_mask = torch.ones((2, 1, 7), dtype=torch.bool)
    resolver_calls: list[tuple[Any, torch.Size, int]] = []

    def resolve_legal_mask(source_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        resolver_calls.append((source_batch, expected_shape, action_dim))
        return resolved_mask

    learner = SimpleNamespace(_resolve_legal_mask=resolve_legal_mask)
    batch = {"teacher_stage_batch": True}
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    logits = torch.zeros((2, 1, 7), dtype=torch.float32)
    legal_mask = torch.ones((2, 1, 7), dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    packed_view = object()
    factorized_result = object()
    public_targets = torch.zeros((2, 1, 7), dtype=torch.float32)
    values = torch.zeros((2, 1), dtype=torch.float32)
    inputs = SimpleNamespace(
        context=context,
        teacher_aux_active=True,
        logits=logits,
        legal_mask=legal_mask,
        loss_mask=loss_mask,
        values=values,
        packed_legal=packed_legal,
        teacher_aux_packed_view=packed_view,
        factorized_result=factorized_result,
        public_heuristic_target_logits=public_targets,
    )
    batch_values: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_values.append((source_batch, key))
        return None

    captured: dict[str, Any] = {}

    def fake_apply_impala_teacher_auxiliary(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        assert kwargs["resolve_legal_mask"](batch, torch.Size((2, 1)), 7) is resolved_mask
        return SimpleNamespace(
            total_loss=kwargs["total_loss"] + torch.tensor(0.25, dtype=torch.float32),
            teacher_metrics={"teacher_valid_fraction": 0.5},
        )

    monkeypatch.setattr(
        impala_loss_teacher_stage,
        "apply_impala_teacher_auxiliary",
        fake_apply_impala_teacher_auxiliary,
    )

    result = apply_impala_teacher_auxiliary_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        action_catalog="catalog",
        batch_value=batch_value,
    )

    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["total_loss"] is total_loss
    assert captured["context"] is context
    assert captured["teacher_aux_active"] is True
    assert captured["logits"] is logits
    assert captured["legal_mask"] is legal_mask
    assert captured["loss_mask"] is loss_mask
    assert captured["action_catalog"] == "catalog"
    assert captured["expected_shape"] == values.shape
    assert captured["packed_legal"] is packed_legal
    assert captured["packed_view"] is packed_view
    assert captured["factorized_result"] is factorized_result
    assert captured["public_heuristic_target_logits"] is public_targets
    assert captured["batch_value"] is batch_value
    assert resolver_calls == [(batch, torch.Size((2, 1)), 7)]
    assert batch_values == []
    torch.testing.assert_close(result.total_loss, torch.tensor(2.25))
    assert result.teacher_metrics == {"teacher_valid_fraction": 0.5}
