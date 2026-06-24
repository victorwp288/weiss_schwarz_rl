from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import weiss_rl.learners.impala.losses.loss_metrics_stage as impala_loss_metrics_stage
from weiss_rl.learners.impala.losses.loss_metrics_stage import assemble_impala_loss_core_metrics
from weiss_rl.learners.impala.losses.objective_loss import compute_impala_objective_losses
from weiss_rl.learners.impala.support.metrics_assembly import ImpalaMetricAssemblyRequest


def test_compute_impala_objective_losses_uses_current_logp_for_retention_and_policy_logp_for_pg() -> None:
    result = compute_impala_objective_losses(
        policy_action_logp=torch.zeros((2, 1), dtype=torch.float32),
        retention_action_logp=torch.as_tensor([[-0.25], [-2.0]], dtype=torch.float32),
        actions=torch.as_tensor([[0], [1]], dtype=torch.long),
        advantages=torch.ones((2, 1), dtype=torch.float32),
        values=torch.zeros((2, 1), dtype=torch.float32),
        targets=torch.zeros((2, 1), dtype=torch.float32),
        entropy=torch.zeros((2, 1), dtype=torch.float32),
        loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
        value_loss_mask=None,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        trajectory_retention_valid=torch.as_tensor([[False], [True]], dtype=torch.bool),
        trajectory_retention_coef=0.5,
    )

    assert result.policy_loss.item() == pytest.approx(0.0)
    assert result.value_loss.item() == pytest.approx(0.0)
    assert result.trajectory_retention_metrics["trajectory_retention_loss"] == pytest.approx(2.0)
    assert result.trajectory_retention_metrics["trajectory_retention_weighted_loss"] == pytest.approx(1.0)
    assert result.total_loss.item() == pytest.approx(1.0)
    assert result.value_loss_mask.tolist() == [[1.0], [1.0]]


def test_compute_impala_objective_losses_respects_explicit_value_mask_and_entropy_term() -> None:
    result = compute_impala_objective_losses(
        policy_action_logp=torch.as_tensor([[-0.5], [-4.0]], dtype=torch.float32),
        retention_action_logp=torch.as_tensor([[-0.5], [-4.0]], dtype=torch.float32),
        actions=torch.as_tensor([[0], [1]], dtype=torch.long),
        advantages=torch.as_tensor([[2.0], [10.0]], dtype=torch.float32),
        values=torch.as_tensor([[0.0], [3.0]], dtype=torch.float32),
        targets=torch.as_tensor([[2.0], [1.0]], dtype=torch.float32),
        entropy=torch.as_tensor([[0.25], [99.0]], dtype=torch.float32),
        loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
        value_loss_mask=torch.as_tensor([[0.0], [1.0]], dtype=torch.float32),
        value_loss_coef=0.5,
        entropy_coef=0.1,
        trajectory_retention_valid=None,
        trajectory_retention_coef=0.0,
    )

    assert result.policy_loss.item() == pytest.approx(1.0)
    assert result.value_loss.item() == pytest.approx(4.0)
    assert result.entropy_mean.item() == pytest.approx(0.25)
    assert result.total_loss.item() == pytest.approx(2.975)
    assert result.value_loss_mask.tolist() == [[0.0], [1.0]]
    assert result.trajectory_retention_metrics == {}


def test_assemble_impala_loss_core_metrics_maps_stage_outputs_to_metric_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_loss = torch.tensor(3.0, dtype=torch.float32)
    policy_loss = torch.tensor(0.5, dtype=torch.float32)
    value_loss = torch.tensor(1.25, dtype=torch.float32)
    entropy_mean = torch.tensor(0.125, dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    value_loss_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    actions = torch.tensor([[2], [3]], dtype=torch.long)
    action_logp = torch.tensor([[-0.2], [-0.7]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-0.1], [-0.6]], dtype=torch.float32)
    rewards = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    advantages = torch.tensor([[0.25], [-0.5]], dtype=torch.float32)
    targets = torch.tensor([[0.75], [-0.25]], dtype=torch.float32)
    rhos = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    legal_mask = torch.ones((2, 1, 5), dtype=torch.bool)
    packed_legal = (
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    packed_view = object()
    factorized_result = object()
    batch = {"metric_stage_batch": True}
    resolved_mask = torch.ones_like(legal_mask)
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    timing_calls: list[tuple[str, float]] = []

    def resolve_legal_mask(source_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        resolver_calls.append((source_batch, expected_shape, action_dim))
        return resolved_mask

    def record_timing(name: str, duration: float) -> None:
        timing_calls.append((name, duration))

    learner = SimpleNamespace(
        entropy_scope="family",
        pass_action_id=4,
        _resolve_legal_mask=resolve_legal_mask,
        _record_timing_ms=record_timing,
    )
    inputs = SimpleNamespace(
        obs=torch.zeros((2, 1, 3), dtype=torch.float32),
        loss_mask=loss_mask,
        actions=actions,
        emit_structured_metrics=True,
        logits=logits,
        legal_mask=legal_mask,
        packed_legal=packed_legal,
        packed_view=packed_view,
        factorized_result=factorized_result,
    )
    objective_losses = SimpleNamespace(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        value_loss_mask=value_loss_mask,
        trajectory_retention_metrics={"trajectory_retention_rows": 2.0},
    )
    policy_anchor_stage = SimpleNamespace(policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25})
    teacher_finalization = SimpleNamespace(teacher_metrics={"teacher_aux_loss": 0.5})
    resolved_vtrace = SimpleNamespace(
        behavior_logp_for_mask=behavior_logp,
        rewards_for_metrics=rewards,
        advantages=advantages,
        targets=targets,
        rhos_for_metrics=rhos,
    )
    clip_config = SimpleNamespace(rho_bar=1.5, c_bar=1.25)
    batch_values: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_values.append((source_batch, key))
        return None

    captured: dict[str, Any] = {}

    def fake_assemble_impala_loss_metrics(
        request: ImpalaMetricAssemblyRequest,
        *,
        batch_value: Any,
        record_timing_ms: Any,
    ) -> dict[str, float]:
        captured["request"] = request
        captured["batch_value"] = batch_value
        captured["record_timing_ms"] = record_timing_ms
        assert request.resolve_legal_mask is not None
        assert request.resolve_legal_mask(batch, torch.Size((2, 1)), 5) is resolved_mask
        return {"loss": 3.0, "metric_stage": 1.0}

    monkeypatch.setattr(
        impala_loss_metrics_stage,
        "assemble_impala_loss_metrics",
        fake_assemble_impala_loss_metrics,
    )

    metrics = assemble_impala_loss_core_metrics(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        objective_losses=cast(Any, objective_losses),
        policy_anchor_stage=cast(Any, policy_anchor_stage),
        teacher_finalization=cast(Any, teacher_finalization),
        resolved_vtrace=resolved_vtrace,
        clip_config=clip_config,
        action_logp=action_logp,
        action_catalog="catalog",
        batch_value=batch_value,
    )

    request = cast(ImpalaMetricAssemblyRequest, captured["request"])
    assert metrics == {"loss": 3.0, "metric_stage": 1.0}
    assert captured["batch_value"] is batch_value
    assert captured["record_timing_ms"] is record_timing
    assert request.total_loss is total_loss
    assert request.policy_loss is policy_loss
    assert request.value_loss is value_loss
    assert request.entropy_mean is entropy_mean
    assert request.entropy_scope == "family"
    assert request.loss_mask is loss_mask
    assert request.value_loss_mask is value_loss_mask
    assert request.actions is actions
    assert request.action_logp is action_logp
    assert request.behavior_logp_for_mask is behavior_logp
    assert request.rewards_for_metrics is rewards
    assert request.advantages is advantages
    assert request.targets is targets
    assert request.rhos_for_metrics is rhos
    assert request.rho_bar == pytest.approx(1.5)
    assert request.c_bar == pytest.approx(1.25)
    assert request.action_catalog == "catalog"
    assert request.pass_action_id == 4
    assert request.trajectory_retention_metrics == {"trajectory_retention_rows": 2.0}
    assert request.policy_anchor_metrics == {"policy_anchor_weighted_loss": 0.25}
    assert request.teacher_metrics == {"teacher_aux_loss": 0.5}
    assert request.emit_structured_metrics is True
    assert request.logits is logits
    assert request.legal_mask is legal_mask
    assert request.packed_legal is packed_legal
    assert request.packed_view is packed_view
    assert request.factorized_result is factorized_result
    assert request.batch is batch
    assert request.expected_shape == torch.Size((2, 1))
    assert request.action_dim == 5
    assert resolver_calls == [(batch, torch.Size((2, 1)), 5)]
    assert batch_values == []
    assert timing_calls == []
