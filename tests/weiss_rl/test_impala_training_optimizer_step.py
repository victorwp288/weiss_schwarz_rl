from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import weiss_rl.learners.impala.update_training_step as impala_update_training_step
from weiss_rl.learners.impala.update_training_step import run_impala_training_optimizer_step


def test_run_impala_training_optimizer_step_validates_builds_loss_and_scales_nonfinite_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"training_step_batch": True}
    learner = SimpleNamespace(model=object())
    loss = torch.tensor(2.0)
    loss_metrics = {"loss": 2.0}
    loss_context = {"context": torch.tensor(1.0)}
    calls: list[str] = []

    def fake_validate_impala_training_inputs(*, learner: Any, batch: Any) -> None:
        assert learner is training_learner
        assert batch is training_batch
        calls.append("validate")

    def fake_build_scoped_impala_loss(*, learner: Any, loss_timer_name: str, build_loss: Any) -> SimpleNamespace:
        assert learner is training_learner
        assert loss_timer_name == "learner_loss_and_metrics"
        built_loss, built_metrics, built_context = build_loss()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        calls.append("build")
        return SimpleNamespace(loss=built_loss, metrics=built_metrics, context=built_context)

    def fake_run_impala_optimizer_step(**kwargs: Any) -> dict[str, float]:
        assert kwargs["learner"] is training_learner
        assert kwargs["batch"] is training_batch
        assert kwargs["loss"] is loss
        assert kwargs["base_metrics"] is loss_metrics
        assert kwargs["context"] is loss_context
        assert kwargs["scale_loss_on_nonfinite_gradients"] is True
        calls.append("optimizer")
        return {"loss": 2.0, "grad_norm": 0.5}

    training_learner = learner
    training_batch = batch
    learner._loss_and_metrics_with_context = lambda source_batch: (
        calls.append("loss") or loss,
        loss_metrics,
        loss_context,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "validate_impala_training_inputs",
        fake_validate_impala_training_inputs,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "build_scoped_impala_loss",
        fake_build_scoped_impala_loss,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "run_impala_optimizer_step",
        fake_run_impala_optimizer_step,
    )

    metrics = run_impala_training_optimizer_step(learner=learner, batch=batch)

    assert calls == ["validate", "loss", "build", "optimizer"]
    assert metrics == {"loss": 2.0, "grad_norm": 0.5}


def test_run_impala_training_optimizer_step_rejects_missing_model_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    learner = SimpleNamespace(model=None)

    monkeypatch.setattr(
        impala_update_training_step,
        "validate_impala_training_inputs",
        lambda *, learner, batch: calls.append("validate"),
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "build_scoped_impala_loss",
        lambda **_kwargs: calls.append("build"),
    )

    with pytest.raises(ValueError, match="ImpalaLearner requires a model to run an optimizer step"):
        run_impala_training_optimizer_step(learner=learner, batch={})

    assert calls == ["validate"]
