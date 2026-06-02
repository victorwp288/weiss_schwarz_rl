from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.learners.impala import summarize_vtrace_diagnostics as impala_summarize_vtrace_diagnostics
from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.learners.vtrace_diagnostics import summarize_vtrace_diagnostics


def test_vtrace_diagnostics_helper_matches_impala_wrapper() -> None:
    result = VTraceTargets(
        vs=np.zeros((2, 2), dtype=np.float32),
        pg_advantages=np.ones((2, 2), dtype=np.float32),
        rhos=np.asarray([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
    )

    direct_metrics = summarize_vtrace_diagnostics(result, rho_bar=1.5, c_bar=0.75)
    wrapper_metrics = impala_summarize_vtrace_diagnostics(result, rho_bar=1.5, c_bar=0.75)

    assert impala_summarize_vtrace_diagnostics is not summarize_vtrace_diagnostics
    assert wrapper_metrics == pytest.approx(direct_metrics)
    assert direct_metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert direct_metrics["vtrace_c_clip_rate"] == pytest.approx(0.75)
