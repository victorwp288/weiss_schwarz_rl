from __future__ import annotations

from weiss_rl.diagnostics import learning_progress as learning_progress_module
from weiss_rl.diagnostics.learning_progress import evaluate_league_guard
from weiss_rl.diagnostics.learning_progress_artifacts import (
    _final_eval_matrix_summary as artifact_final_eval_matrix_summary,
)
from weiss_rl.diagnostics.learning_progress_artifacts import (
    _periodic_dev_eval_trend as artifact_periodic_dev_eval_trend,
)
from weiss_rl.diagnostics.learning_progress_artifacts import _promotion_gate_summary as artifact_promotion_gate_summary
from weiss_rl.diagnostics.learning_progress_guard import (
    DEFAULT_LEAGUE_GUARD_ANCHORS as guard_default_league_guard_anchors,
)
from weiss_rl.diagnostics.learning_progress_guard import (
    evaluate_league_guard as guard_evaluate_league_guard,
)
from weiss_rl.diagnostics.learning_progress_metrics import (
    _window_summary as metric_window_summary,
)
from weiss_rl.diagnostics.learning_progress_metrics import (
    build_training_log_summary_sections as metric_build_training_log_summary_sections,
)


def test_learning_progress_artifact_helpers_are_package_owned() -> None:
    assert learning_progress_module._final_eval_matrix_summary is artifact_final_eval_matrix_summary
    assert learning_progress_module._periodic_dev_eval_trend is artifact_periodic_dev_eval_trend
    assert learning_progress_module._promotion_gate_summary is artifact_promotion_gate_summary
    assert artifact_final_eval_matrix_summary.__module__ == "weiss_rl.diagnostics.learning_progress_artifacts"


def test_learning_progress_guard_helpers_are_package_owned() -> None:
    assert evaluate_league_guard is guard_evaluate_league_guard
    assert learning_progress_module.DEFAULT_LEAGUE_GUARD_ANCHORS is guard_default_league_guard_anchors
    assert guard_evaluate_league_guard.__module__ == "weiss_rl.diagnostics.learning_progress_guard"


def test_learning_progress_metric_sections_are_package_owned() -> None:
    assert learning_progress_module._window_summary is metric_window_summary
    assert learning_progress_module.build_training_log_summary_sections is metric_build_training_log_summary_sections
    assert metric_build_training_log_summary_sections.__module__ == "weiss_rl.diagnostics.learning_progress_metrics"
