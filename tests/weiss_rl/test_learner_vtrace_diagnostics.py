from __future__ import annotations

import numpy as np
import pytest
import weiss_rl.learners.impala as impala_root
from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.learners.vtrace_diagnostics import summarize_vtrace_diagnostics

from .vtrace_test_support import fixture_array, load_vtrace_fixture


def test_vtrace_diagnostics_helper_reports_clip_rates() -> None:
    result = VTraceTargets(
        vs=np.zeros((2, 2), dtype=np.float32),
        pg_advantages=np.ones((2, 2), dtype=np.float32),
        rhos=np.asarray([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
    )

    metrics = summarize_vtrace_diagnostics(result, rho_bar=1.5, c_bar=0.75)

    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(0.75)


def test_summarize_vtrace_diagnostics_reports_percentiles_and_clip_rates() -> None:
    fixture = load_vtrace_fixture()
    result = VTraceTargets(
        vs=fixture_array(fixture["expected_vs"]),
        pg_advantages=fixture_array(fixture["expected_pg_advantages"]),
        rhos=fixture_array(fixture["expected_rhos"]),
    )

    metrics = summarize_vtrace_diagnostics(result, rho_bar=fixture["rho_bar"], c_bar=fixture["c_bar"])

    assert metrics == pytest.approx(
        {
            "vtrace_rho_p50": 0.9811104834079742,
            "vtrace_rho_p90": 1.6487212538719178,
            "vtrace_rho_p95": 1.6487212955951691,
            "vtrace_rho_p99": 1.64872132897377,
            "vtrace_rho_clip_rate": 0.5,
            "vtrace_c_clip_rate": 0.5,
        }
    )


def test_impala_root_does_not_reexport_vtrace_diagnostics_helpers() -> None:
    assert not hasattr(impala_root, "summarize_vtrace_diagnostics")
    assert not hasattr(impala_root, "VTRACE_RHO_PERCENTILES")
