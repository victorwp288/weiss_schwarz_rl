from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import weiss_rl.learners.impala.loss_vtrace_stage as impala_loss_vtrace_stage
from weiss_rl.learners.impala.loss_vtrace_stage import compute_impala_vtrace_stage


def test_compute_impala_vtrace_stage_resolves_targets_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_logp = torch.tensor([[-0.2], [-0.7]], dtype=torch.float32, requires_grad=True)
    resolved_action_logp = torch.tensor([[-0.3], [-0.8]], dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 1), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True)
    context: dict[str, Any] = {}
    inputs = SimpleNamespace(
        vtrace_result="vtrace-result",
        values=values,
        loss_mask=loss_mask,
        context=context,
    )
    batch = {"vtrace_rho_bar": 2.0, "vtrace_c_bar": 0.5}
    float_target = object()
    resolve_bootstrap_value = object()
    learner = SimpleNamespace(
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        _float_target=float_target,
        _resolve_vtrace_bootstrap_value=resolve_bootstrap_value,
    )
    resolved_vtrace = SimpleNamespace(
        action_logp=resolved_action_logp,
        behavior_logp_for_mask=torch.zeros((2, 1), dtype=torch.float32),
        targets=torch.ones((2, 1), dtype=torch.float32, requires_grad=True),
        advantages=torch.full((2, 1), 2.0, dtype=torch.float32, requires_grad=True),
        rhos_for_metrics=torch.full((2, 1), 3.0, dtype=torch.float32, requires_grad=True),
        rewards_for_metrics=torch.full((2, 1), 4.0, dtype=torch.float32, requires_grad=True),
    )
    captured: dict[str, Any] = {}

    def fake_resolve_impala_vtrace_targets(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return resolved_vtrace

    monkeypatch.setattr(
        impala_loss_vtrace_stage,
        "resolve_impala_vtrace_targets",
        fake_resolve_impala_vtrace_targets,
    )
    batch_value_calls: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_value_calls.append((source_batch, key))
        return source_batch.get(key)

    stage = compute_impala_vtrace_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        action_logp=action_logp,
        batch_value=batch_value,
    )

    assert stage.retention_action_logp is action_logp
    assert stage.action_logp is resolved_action_logp
    assert stage.clip_config.rho_bar == pytest.approx(2.0)
    assert stage.clip_config.c_bar == pytest.approx(0.5)
    assert stage.resolved_vtrace is resolved_vtrace
    assert captured["batch"] is batch
    assert captured["vtrace_result"] == "vtrace-result"
    assert captured["values"] is values
    assert captured["action_logp"] is action_logp
    assert captured["loss_mask"] is loss_mask
    assert captured["rho_bar"] == pytest.approx(2.0)
    assert captured["c_bar"] == pytest.approx(0.5)
    assert captured["float_target"] is float_target
    assert captured["resolve_bootstrap_value"] is resolve_bootstrap_value
    assert captured["batch_value"] is batch_value
    assert batch_value_calls == [(batch, "vtrace_rho_bar"), (batch, "vtrace_c_bar")]
    torch.testing.assert_close(context["targets"], resolved_vtrace.targets)
    torch.testing.assert_close(context["advantages"], resolved_vtrace.advantages)
    torch.testing.assert_close(context["vtrace_rhos"], resolved_vtrace.rhos_for_metrics)
    torch.testing.assert_close(context["rewards"], resolved_vtrace.rewards_for_metrics)
    torch.testing.assert_close(context["policy_train_mask"], loss_mask)
    assert context["targets"].requires_grad is False
    assert context["advantages"].requires_grad is False
    assert context["vtrace_rhos"].requires_grad is False
    assert context["rewards"].requires_grad is False
    assert context["policy_train_mask"].requires_grad is False
