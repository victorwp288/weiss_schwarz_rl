from __future__ import annotations

from pathlib import Path

import pytest

from .learning_progress_test_support import build_overheated_training_summary


def test_learning_progress_diagnostic_flags_large_vtrace_rho(tmp_path: Path) -> None:
    summary = build_overheated_training_summary(tmp_path)

    assert summary["off_policy"]["max_vtrace_rho_mean"] == 2636.0
    assert summary["off_policy"]["max_vtrace_rho_p99"] == 4100.0
    assert summary["off_policy"]["max_vtrace_train_rho_mean"] == 4108.0
    assert summary["off_policy"]["max_vtrace_train_rho_p95"] == 4096.0
    assert summary["off_policy"]["max_vtrace_train_rho_p99"] == 4097.0
    assert summary["off_policy"]["max_vtrace_clip_rate"] == 0.75
    assert summary["off_policy"]["max_target_behavior_train_logp_delta_abs_mean"] == pytest.approx(0.4)
    assert summary["off_policy"]["max_target_behavior_train_logp_delta_abs_p99"] == pytest.approx(1.5)
    assert any("vtrace_rho_mean exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_rho_p99 exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_train_rho_mean exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_train_rho_p95 exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_train_rho_p99 exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_clip_rate exceeded 0.5" in warning for warning in summary["warnings"])
    assert any("target_behavior_train_logp_delta_abs_p99 exceeded" in warning for warning in summary["warnings"])
