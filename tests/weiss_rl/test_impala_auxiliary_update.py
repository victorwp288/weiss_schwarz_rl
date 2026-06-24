from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import weiss_rl.learners.impala.updates.auxiliary_update as impala_auxiliary_update

from .impala_test_support import _simple_training_batch


def test_run_impala_auxiliary_optimizer_update_uses_auxiliary_loss_and_scoped_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"auxiliary": True}
    loss = torch.tensor(1.25)
    loss_metrics = {"auxiliary_metric": 1.25}
    loss_context = {"auxiliary_context": torch.tensor(2.0)}
    calls: list[tuple[str, Any]] = []

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert (
            kwargs["spec"].missing_model_message == "ImpalaLearner requires a model to run an auxiliary optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_auxiliary_loss_and_metrics"
        built_loss, built_metrics, built_context = kwargs["build_loss"]()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        return {"loss": float(built_loss.item()), **built_metrics}

    learner = SimpleNamespace(
        model=object(),
        _auxiliary_loss_and_metrics=lambda source_batch: (
            calls.append(("auxiliary_loss", source_batch)) or loss,
            loss_metrics,
            loss_context,
        ),
    )
    monkeypatch.setattr(
        impala_auxiliary_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_auxiliary_update.run_impala_auxiliary_optimizer_update(learner=learner, batch=batch)

    assert result == {"loss": pytest.approx(1.25), "auxiliary_metric": 1.25}
    assert [name for name, _payload in calls] == ["scoped", "auxiliary_loss"]
    assert calls[1] == ("auxiliary_loss", batch)


def test_run_impala_auxiliary_optimizer_update_rejects_missing_model_before_auxiliary_loss() -> None:
    learner = SimpleNamespace(
        model=None,
        _auxiliary_loss_and_metrics=lambda _batch: pytest.fail("auxiliary loss should not be built"),
    )

    with pytest.raises(ValueError, match="ImpalaLearner requires a model to run an auxiliary optimizer step"):
        impala_auxiliary_update.run_impala_auxiliary_optimizer_update(
            learner=learner,
            batch=_simple_training_batch(),
        )
