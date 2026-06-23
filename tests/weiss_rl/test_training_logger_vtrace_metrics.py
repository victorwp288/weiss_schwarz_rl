from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.learners.vtrace import VtraceMetrics, compute_vtrace_metrics


def _masked_batch(*, time_steps: int = 4, batch_size: int = 2, action_space: int = 5) -> dict[str, np.ndarray]:
    return {
        "logits": np.random.randn(time_steps, batch_size, action_space),
        "behavior_logits": np.random.randn(time_steps, batch_size, action_space),
        "actions": np.random.randint(0, action_space, size=(time_steps, batch_size)),
        "legal_mask": np.ones((time_steps, batch_size, action_space), dtype=bool),
        "rewards": np.random.randn(time_steps, batch_size),
    }


def test_vtrace_metrics_ignore_illegal_logits_when_masked() -> None:
    batch = {
        "logits": np.array([[[0.0, 1.0, 100.0, -100.0]]], dtype=np.float32),
        "behavior_logits": np.array([[[0.0, 1.0, -50.0, 50.0]]], dtype=np.float32),
        "actions": np.array([[1]], dtype=np.int64),
        "legal_mask": np.array([[[True, True, False, False]]]),
    }

    metrics = compute_vtrace_metrics(batch)

    assert isinstance(metrics, VtraceMetrics)
    assert metrics.rho_mean == pytest.approx(1.0)
    assert metrics.kl_divergence == pytest.approx(0.0)
    expected_entropy = -(
        (math.e / (math.e + 1.0)) * math.log(math.e / (math.e + 1.0))
        + (1.0 / (math.e + 1.0)) * math.log(1.0 / (math.e + 1.0))
    )
    assert metrics.entropy == pytest.approx(expected_entropy)


def test_vtrace_metrics_support_packed_legal_ids() -> None:
    batch = {
        "logits": np.array([[[1.0, 9.0, 0.0, 9.0]]], dtype=np.float32),
        "behavior_logits": np.array([[[0.0, -9.0, 0.0, -9.0]]], dtype=np.float32),
        "actions": np.array([[2]], dtype=np.int64),
        "legal_ids": np.array([0, 2], dtype=np.int64),
        "legal_offsets": np.array([0, 2], dtype=np.int64),
    }

    metrics = compute_vtrace_metrics(batch)

    expected_rho = (1.0 / (math.e + 1.0)) / 0.5
    expected_entropy = -(
        (math.e / (math.e + 1.0)) * math.log(math.e / (math.e + 1.0))
        + (1.0 / (math.e + 1.0)) * math.log(1.0 / (math.e + 1.0))
    )
    expected_kl = 0.5 * math.log(0.5 / (math.e / (math.e + 1.0))) + 0.5 * math.log(0.5 / (1.0 / (math.e + 1.0)))

    assert metrics.rho_mean == pytest.approx(expected_rho)
    assert metrics.rho_p90 == pytest.approx(expected_rho)
    assert metrics.kl_divergence == pytest.approx(expected_kl)
    assert metrics.entropy == pytest.approx(expected_entropy)


def test_vtrace_metrics_require_legality_surface() -> None:
    metrics = compute_vtrace_metrics(
        {
            "logits": np.zeros((1, 1, 2), dtype=np.float32),
            "behavior_logits": np.zeros((1, 1, 2), dtype=np.float32),
            "actions": np.zeros((1, 1), dtype=np.int64),
        }
    )

    assert math.isnan(metrics.rho_mean)
    assert math.isnan(metrics.rho_p50)
    assert math.isnan(metrics.rho_p90)
    assert math.isnan(metrics.rho_p99)
    assert math.isnan(metrics.clip_rate)
    assert math.isnan(metrics.c_clipped_rate)
    assert math.isnan(metrics.kl_divergence)
    assert math.isnan(metrics.entropy)


def test_vtrace_metrics_support_object_batches() -> None:
    batch = SimpleNamespace(**_masked_batch(time_steps=1, batch_size=1, action_space=3))

    metrics = compute_vtrace_metrics(batch)

    assert isinstance(metrics, VtraceMetrics)
    assert math.isfinite(metrics.rho_mean)
    assert math.isfinite(metrics.kl_divergence)
    assert math.isfinite(metrics.entropy)
