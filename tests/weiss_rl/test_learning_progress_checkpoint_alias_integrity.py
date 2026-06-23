from __future__ import annotations

from pathlib import Path

from .learning_progress_test_support import build_stale_checkpoint_alias_summary


def test_learning_progress_diagnostic_warns_when_latest_alias_mismatches_tracker_source(tmp_path: Path) -> None:
    summary = build_stale_checkpoint_alias_summary(tmp_path)

    assert summary["checkpoint_alias_integrity"]["latest_alias_matches_source"] is False
    assert summary["checkpoint_alias_integrity"]["observed_best_alias_matches_source"] is False
    assert any("latest checkpoint alias file does not match" in warning for warning in summary["warnings"])
    assert any("observed_best checkpoint alias file does not match" in warning for warning in summary["warnings"])
