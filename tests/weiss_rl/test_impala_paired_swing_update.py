from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import weiss_rl.learners.impala.updates.paired_swing_update as impala_paired_swing_update


def test_run_impala_paired_swing_optimizer_update_validates_full_surface_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    learner = SimpleNamespace(model=object())

    monkeypatch.setattr(
        impala_paired_swing_update,
        "run_scoped_impala_optimizer_update",
        lambda **_kwargs: calls.append("scoped"),
    )

    with pytest.raises(ValueError, match="full_surface_top_action_retention_coef must be >= 0"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_coef=-0.1,
        )
    with pytest.raises(ValueError, match="full_surface_top_action_retention_margin must be >= 0"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_margin=-0.1,
        )
    with pytest.raises(ValueError, match="full_surface_retention_batch is required"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_coef=0.5,
        )

    assert calls == []


def test_run_impala_paired_swing_optimizer_update_composes_full_surface_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"paired": True}
    retention_batch = {"retention": True}
    swing_loss = torch.tensor(2.0)
    retention_loss = torch.tensor(0.5)
    calls: list[tuple[str, Any]] = []

    def paired_swing_loss(source_batch: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("swing", (source_batch, kwargs)))
        return swing_loss, {"swing_metric": 2.0}, {"swing_context": "base"}

    def full_surface_retention(
        source_batch: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("retention", (source_batch, kwargs)))
        return retention_loss, {"retention_metric": 0.5}, {"retention_context": "extra"}

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["spec"].missing_model_message == (
            "ImpalaLearner requires a model to run a paired-swing optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_paired_swing_loss_and_metrics"
        loss, metrics, context = kwargs["build_loss"]()
        assert loss.item() == pytest.approx(2.5)
        assert metrics == {"swing_metric": 2.0, "retention_metric": 0.5}
        assert context == {"swing_context": "base", "retention_context": "extra"}
        return {"loss": float(loss.item()), **metrics}

    learner = SimpleNamespace(
        model=object(),
        _paired_swing_loss_and_metrics=paired_swing_loss,
        _paired_swing_full_surface_top_action_retention_loss_and_metrics=full_surface_retention,
    )
    monkeypatch.setattr(
        impala_paired_swing_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
        learner=learner,
        batch=batch,
        margin=1,
        coef=0.75,
        positive_action_source="teacher_positive",
        negative_action_source="learner_negative",
        loss_scope="span",
        compare_to="baseline",
        margin_retention_coef=0.25,
        margin_retention_margin=0.5,
        top_action_retention_coef=0.125,
        top_action_retention_margin=0.75,
        full_surface_retention_batch=retention_batch,
        full_surface_top_action_retention_coef=0.4,
        full_surface_top_action_retention_margin=0.6,
        full_surface_top_action_retention_mode="target_action",
    )

    assert result == {"loss": pytest.approx(2.5), "swing_metric": 2.0, "retention_metric": 0.5}
    assert [name for name, _payload in calls] == ["scoped", "swing", "retention"]
    assert calls[1][0] == "swing"
    assert calls[1][1][0] is batch
    assert calls[1][1][1] == {
        "margin": 1.0,
        "coef": 0.75,
        "positive_action_source": "teacher_positive",
        "negative_action_source": "learner_negative",
        "loss_scope": "span",
        "compare_to": "baseline",
        "margin_retention_coef": 0.25,
        "margin_retention_margin": 0.5,
        "top_action_retention_coef": 0.125,
        "top_action_retention_margin": 0.75,
    }
    assert calls[2] == (
        "retention",
        (
            retention_batch,
            {
                "coef": 0.4,
                "margin": 0.6,
                "mode": "target_action",
            },
        ),
    )
