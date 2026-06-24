from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.learners.impala.updates.update_training_inputs import (
    has_impala_training_inputs,
    missing_impala_training_input_fields,
    resolve_impala_update_vtrace_result,
    summarize_precomputed_vtrace_update_metrics,
    validate_impala_training_inputs,
)
from weiss_rl.learners.vtrace import VTraceTargets


def test_impala_update_training_input_helpers_preserve_missing_field_contract() -> None:
    learner = SimpleNamespace(
        _has_legal_actions=lambda batch: False,
        _has_raw_vtrace_inputs=lambda batch: False,
    )
    batch = {"obs": np.zeros((1, 1, 2), dtype=np.float32)}

    assert has_impala_training_inputs(batch) is True
    assert resolve_impala_update_vtrace_result(batch) is None
    assert missing_impala_training_input_fields(learner=learner, batch=batch) == [
        "actions",
        "legal_actions",
        "vtrace_result_or_raw_inputs",
    ]
    with pytest.raises(
        ValueError,
        match=(
            "batch must include obs, actions, legality, and either vtrace_result or raw vtrace inputs "
            "for learner updates; missing actions, legal_actions, vtrace_result_or_raw_inputs"
        ),
    ):
        validate_impala_training_inputs(learner=learner, batch=batch)


def test_impala_update_training_input_helpers_accept_raw_vtrace_and_summarize_precomputed_targets() -> None:
    learner = SimpleNamespace(
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        _has_legal_actions=lambda batch: True,
        _has_raw_vtrace_inputs=lambda batch: True,
    )
    vtrace_result = VTraceTargets(
        vs=np.zeros((2, 1), dtype=np.float32),
        pg_advantages=np.zeros((2, 1), dtype=np.float32),
        rhos=np.asarray([[0.5], [2.0]], dtype=np.float32),
    )
    batch = {
        "obs": np.zeros((2, 1, 2), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.int64),
        "vtrace_result": vtrace_result,
        "vtrace_rho_bar": 1.5,
        "vtrace_c_bar": 0.75,
    }

    assert missing_impala_training_input_fields(learner=learner, batch=batch) == []
    validate_impala_training_inputs(learner=learner, batch=batch)
    assert (
        summarize_precomputed_vtrace_update_metrics(
            learner=learner,
            batch=batch,
            vtrace_result=None,
        )
        == {}
    )

    metrics = summarize_precomputed_vtrace_update_metrics(
        learner=learner,
        batch=batch,
        vtrace_result=vtrace_result,
    )

    assert metrics["vtrace_rho_p50"] == pytest.approx(1.25)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(0.5)
