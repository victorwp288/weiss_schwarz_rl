from __future__ import annotations

import numpy as np
import pytest
import weiss_rl.eval.simulator.harness as eval_harness
from weiss_rl.eval import EvalSamplerAnomalies, sample_action_pinned
from weiss_rl.eval.simulator.harness import _normalize_cdf_probs

from .eval_sampler_test_support import StubFloatRng, expected_single_row_logp


def test_normalize_cdf_probs_counts_renormalization_anomaly() -> None:
    probs64 = np.array([0.6, 0.400002], dtype=np.float64)
    anomalies = EvalSamplerAnomalies()

    normalized = _normalize_cdf_probs(probs64, anomalies=anomalies)

    assert float(np.sum(normalized, dtype=np.float64)) == pytest.approx(1.0)
    assert anomalies.cdf_renormalizations == 1


def test_sample_action_pinned_plumbs_renormalization_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = np.array([0.0, 0.5], dtype=np.float32)
    legal_ids = np.array([0, 1], dtype=np.uint32)
    anomalies = EvalSamplerAnomalies()

    def _fake_legal_probs_for_cdf(
        logits: np.ndarray,
        legal_ids: np.ndarray,
        *,
        anomalies: EvalSamplerAnomalies | None = None,
    ) -> np.ndarray:
        del logits, legal_ids
        return _normalize_cdf_probs(np.array([0.6, 0.400002], dtype=np.float64), anomalies=anomalies)

    monkeypatch.setattr(eval_harness, "_legal_probs_for_cdf", _fake_legal_probs_for_cdf)

    action, logp = sample_action_pinned(logits, legal_ids, rng=StubFloatRng(0.75), anomalies=anomalies)

    assert action == 1
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert anomalies.cdf_renormalizations == 1
