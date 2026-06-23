from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import weiss_rl.learners.impala.paired_outcome_update as impala_paired_outcome_update


def test_run_impala_paired_outcome_preference_optimizer_update_forwards_casted_replay_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"preference": True}
    loss = torch.tensor(1.75)
    loss_metrics = {"preference_metric": 1.75}
    loss_context = {"preference_context": torch.tensor(3.0)}
    calls: list[tuple[str, Any]] = []

    def paired_outcome_loss(
        source_batch: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("preference_loss", (source_batch, kwargs)))
        return loss, loss_metrics, loss_context

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["spec"].missing_model_message == (
            "ImpalaLearner requires a model to run a paired outcome preference optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_paired_outcome_preference_loss_and_metrics"
        built_loss, built_metrics, built_context = kwargs["build_loss"]()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        return {"loss": float(built_loss.item()), **built_metrics}

    learner = SimpleNamespace(
        model=object(),
        _paired_outcome_preference_loss_and_metrics=paired_outcome_loss,
    )
    monkeypatch.setattr(
        impala_paired_outcome_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_outcome_update.run_impala_paired_outcome_preference_optimizer_update(
        learner=learner,
        batch=batch,
        beta="0.7",
        coef="0.25",
        aggregation=123,
        group_balance=1,
        retention_coef="0.5",
        retention_margin="0.125",
        retention_role=456,
        retention_reference_top_only=1,
        top_action_retention_coef="0.75",
        top_action_retention_margin="0.875",
        top_action_retention_role=789,
        top_action_retention_reference_top_only=1,
    )

    assert result == {"loss": pytest.approx(1.75), "preference_metric": 1.75}
    assert [name for name, _payload in calls] == ["scoped", "preference_loss"]
    assert calls[1][1][0] is batch
    assert calls[1][1][1] == {
        "beta": 0.7,
        "coef": 0.25,
        "aggregation": "123",
        "group_balance": True,
        "retention_coef": 0.5,
        "retention_margin": 0.125,
        "retention_role": "456",
        "retention_reference_top_only": True,
        "top_action_retention_coef": 0.75,
        "top_action_retention_margin": 0.875,
        "top_action_retention_role": "789",
        "top_action_retention_reference_top_only": True,
    }


def test_run_impala_paired_outcome_preference_optimizer_update_uses_default_replay_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"preference": True}
    captured_kwargs: dict[str, Any] = {}
    learner = SimpleNamespace(
        model=object(),
        _paired_outcome_preference_loss_and_metrics=lambda source_batch, **kwargs: (
            captured_kwargs.update(kwargs) or torch.tensor(0.5),
            {"preference_metric": 0.5},
            {},
        ),
    )

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        loss, metrics, _context = kwargs["build_loss"]()
        return {"loss": float(loss.item()), **metrics}

    monkeypatch.setattr(
        impala_paired_outcome_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_outcome_update.run_impala_paired_outcome_preference_optimizer_update(
        learner=learner,
        batch=batch,
        beta=0.3,
        coef=0.2,
    )

    assert result == {"loss": pytest.approx(0.5), "preference_metric": 0.5}
    assert captured_kwargs == {
        "beta": 0.3,
        "coef": 0.2,
        "aggregation": "mean",
        "group_balance": False,
        "retention_coef": 0.0,
        "retention_margin": 0.0,
        "retention_role": "preferred",
        "retention_reference_top_only": False,
        "top_action_retention_coef": 0.0,
        "top_action_retention_margin": 0.0,
        "top_action_retention_role": "all",
        "top_action_retention_reference_top_only": False,
    }
