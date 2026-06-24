from __future__ import annotations

from pathlib import Path

import pytest
import torch
from weiss_rl.diagnostics.logging.training_logger import TrainingLogger
from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.learners.vtrace import compute_vtrace_targets
from weiss_rl.model import PolicyValueModel

from .vtrace_test_support import fixture_array, load_vtrace_fixture, synthetic_vtrace_training_batch, tiny_model_config


def test_impala_learner_update_exposes_vtrace_metrics() -> None:
    fixture = load_vtrace_fixture()
    learner = ImpalaLearner()
    result = compute_vtrace_targets(
        fixture_array(fixture["rewards"]),
        fixture_array(fixture["values"]),
        fixture_array(fixture["discounts"]),
        fixture_array(fixture["behavior_logp"]),
        fixture_array(fixture["target_logp"]),
        rho_bar=fixture["rho_bar"],
        c_bar=fixture["c_bar"],
    )

    metrics = learner.update(
        {
            "vtrace_result": result,
            "vtrace_rho_bar": fixture["rho_bar"],
            "vtrace_c_bar": fixture["c_bar"],
        }
    )

    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["vtrace_rho_p95"] == pytest.approx(1.6487212955951691)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(0.5)


def test_impala_learner_update_reduces_fixed_batch_loss_on_synthetic_targets() -> None:
    torch.manual_seed(0)

    model = PolicyValueModel(observation_dim=4, action_dim=4, config=tiny_model_config())
    learner = ImpalaLearner(
        model=model,
        learning_rate=0.05,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        grad_norm_clip=10.0,
    )
    batch = synthetic_vtrace_training_batch(learner)

    before_loss, before_metrics = learner._loss_and_metrics(batch)
    update_metrics = learner.update(batch)
    after_loss, after_metrics = learner._loss_and_metrics(batch)

    assert update_metrics["loss"] == pytest.approx(float(before_loss.detach()))
    assert update_metrics["policy_loss"] == pytest.approx(before_metrics["policy_loss"])
    assert update_metrics["value_loss"] == pytest.approx(before_metrics["value_loss"])
    assert update_metrics["entropy"] > 0.0
    assert update_metrics["grad_norm"] > 0.0
    assert after_metrics["loss"] < before_metrics["loss"]
    assert float(after_loss.detach()) == pytest.approx(after_metrics["loss"])


def test_impala_learner_logging_persists_returned_loss_metrics(tmp_path: Path) -> None:
    torch.manual_seed(0)

    model = PolicyValueModel(observation_dim=4, action_dim=4, config=tiny_model_config())
    learner = ImpalaLearner(
        model=model,
        learning_rate=0.05,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        grad_norm_clip=10.0,
        logs_dir=tmp_path / "logs",
        logging_interval_updates=1,
    )
    batch = synthetic_vtrace_training_batch(learner)

    update_metrics = learner.update(batch)

    [record] = TrainingLogger.read_jsonl(tmp_path / "logs" / "training_metrics.jsonl")
    assert record["loss"] == pytest.approx(update_metrics["loss"])
    assert record["value_loss"] == pytest.approx(update_metrics["value_loss"])
    assert record["actor_loss"] == pytest.approx(update_metrics["policy_loss"])
    assert record["entropy"] == pytest.approx(update_metrics["entropy"])
    assert record["vtrace_rho_p50"] == pytest.approx(update_metrics["vtrace_rho_p50"])
    assert record["vtrace_rho_p90"] == pytest.approx(update_metrics["vtrace_rho_p90"])
    assert record["vtrace_rho_p99"] == pytest.approx(update_metrics["vtrace_rho_p99"])
    assert record["vtrace_clip_rate"] == pytest.approx(update_metrics["vtrace_rho_clip_rate"])
    assert record["vtrace_c_clipped_rate"] == pytest.approx(update_metrics["vtrace_c_clip_rate"])
    assert record["vtrace_rho_mean"] == pytest.approx(update_metrics["vtrace_rho_mean"])
    assert record["kl_divergence"] is None
    assert record["custom_metrics"]["vtrace_batch_metrics_available"] == 0.0
    assert record["custom_metrics"]["vtrace_rho_p95"] == pytest.approx(update_metrics["vtrace_rho_p95"])
