from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import weiss_rl.learners.impala.normal_update as impala_normal_update


def test_run_impala_normal_update_runs_training_step_diagnostics_logging_and_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = SimpleNamespace(name="learner")
    batch = {"training": True}
    scope_metrics = {"loss": 0.0, "throughput": 10.0}
    vtrace_result = object()
    calls: list[tuple[str, Any]] = []

    def fake_begin_scope(**kwargs: Any) -> SimpleNamespace:
        calls.append(("begin", kwargs))
        assert kwargs == {
            "learner": learner,
            "batch": batch,
            "count_learner_update": True,
            "include_training_metrics": True,
            "checkpoint_on_interval": True,
        }
        return SimpleNamespace(started_at=12.5, metrics=scope_metrics)

    def fake_training_step(**kwargs: Any) -> dict[str, float]:
        calls.append(("training", kwargs))
        assert kwargs == {"learner": learner, "batch": batch}
        return {"loss": 1.5, "grad_norm": 0.25}

    def fake_summarize(**kwargs: Any) -> dict[str, float]:
        calls.append(("summarize", kwargs))
        assert kwargs == {"learner": learner, "batch": batch, "vtrace_result": vtrace_result}
        return {"vtrace_rho_p50": 0.75}

    def fake_log(**kwargs: Any) -> bool:
        calls.append(("log", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["metrics"] is scope_metrics
        assert kwargs["metrics"] == {
            "loss": 1.5,
            "throughput": 10.0,
            "grad_norm": 0.25,
            "vtrace_rho_p50": 0.75,
        }
        return True

    def fake_finalize(**kwargs: Any) -> dict[str, float]:
        calls.append(("finalize", kwargs))
        assert kwargs == {"learner": learner, "metrics": scope_metrics, "started_at": 12.5}
        return {"final_loss": scope_metrics["loss"], "final_vtrace": scope_metrics["vtrace_rho_p50"]}

    monkeypatch.setattr(impala_normal_update, "begin_impala_update_scope", fake_begin_scope)
    monkeypatch.setattr(impala_normal_update, "resolve_impala_update_vtrace_result", lambda source_batch: vtrace_result)
    monkeypatch.setattr(impala_normal_update, "has_impala_training_inputs", lambda source_batch: True)
    monkeypatch.setattr(impala_normal_update, "run_impala_training_optimizer_step", fake_training_step)
    monkeypatch.setattr(impala_normal_update, "summarize_precomputed_vtrace_update_metrics", fake_summarize)
    monkeypatch.setattr(impala_normal_update, "log_impala_update_metrics_if_due", fake_log)
    monkeypatch.setattr(impala_normal_update, "finalize_impala_update_scope", fake_finalize)

    result = impala_normal_update.run_impala_normal_update(learner=learner, batch=batch)

    assert result == {"final_loss": 1.5, "final_vtrace": 0.75}
    assert [name for name, _payload in calls] == ["begin", "training", "summarize", "log", "finalize"]


def test_run_impala_normal_update_skips_training_step_without_training_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = SimpleNamespace(name="learner")
    batch = {"metadata_only": True}
    scope_metrics = {"loss": 0.0}
    calls: list[str] = []

    monkeypatch.setattr(
        impala_normal_update,
        "begin_impala_update_scope",
        lambda **kwargs: calls.append("begin") or SimpleNamespace(started_at=3.0, metrics=scope_metrics),
    )
    monkeypatch.setattr(
        impala_normal_update,
        "resolve_impala_update_vtrace_result",
        lambda source_batch: calls.append("vtrace") or None,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "has_impala_training_inputs",
        lambda source_batch: calls.append("has_training") or False,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "run_impala_training_optimizer_step",
        lambda **_kwargs: pytest.fail("training optimizer step should be skipped"),
    )
    monkeypatch.setattr(
        impala_normal_update,
        "summarize_precomputed_vtrace_update_metrics",
        lambda **kwargs: calls.append("summarize") or {"vtrace_rows": 0.0},
    )
    monkeypatch.setattr(
        impala_normal_update,
        "log_impala_update_metrics_if_due",
        lambda **kwargs: calls.append("log") or False,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "finalize_impala_update_scope",
        lambda **kwargs: calls.append("finalize") or dict(kwargs["metrics"]),
    )

    result = impala_normal_update.run_impala_normal_update(learner=learner, batch=batch)

    assert result == {"loss": 0.0, "vtrace_rows": 0.0}
    assert calls == ["begin", "vtrace", "has_training", "summarize", "log", "finalize"]
