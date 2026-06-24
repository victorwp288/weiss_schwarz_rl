from __future__ import annotations

from weiss_rl.diagnostics.progress import learning_progress as learning_progress_module
from weiss_rl.diagnostics.progress.learning_progress import evaluate_league_guard
from weiss_rl.diagnostics.progress.learning_progress_artifacts import (
    _final_eval_matrix_summary as artifact_final_eval_matrix_summary,
)
from weiss_rl.diagnostics.progress.learning_progress_artifacts import (
    _periodic_dev_eval_trend as artifact_periodic_dev_eval_trend,
)
from weiss_rl.diagnostics.progress.learning_progress_artifacts import (
    _promotion_gate_summary as artifact_promotion_gate_summary,
)
from weiss_rl.diagnostics.progress.learning_progress_guard import (
    DEFAULT_LEAGUE_GUARD_ANCHORS as guard_default_league_guard_anchors,
)
from weiss_rl.diagnostics.progress.learning_progress_guard import (
    evaluate_league_guard as guard_evaluate_league_guard,
)
from weiss_rl.diagnostics.progress.learning_progress_metrics import (
    _window_summary as metric_window_summary,
)
from weiss_rl.diagnostics.progress.learning_progress_metrics import (
    build_training_log_summary_sections as metric_build_training_log_summary_sections,
)
from weiss_rl.diagnostics.progress.learning_progress_sections import (
    LEARNING_PROGRESS_DIAGNOSTIC_SECTIONS as section_plan,
)
from weiss_rl.diagnostics.progress.learning_progress_sections import (
    learning_progress_diagnostic_plan_payload as section_plan_payload,
)

from .learning_progress_test_support import build_overheated_training_summary


def test_learning_progress_artifact_helpers_are_package_owned() -> None:
    assert learning_progress_module._final_eval_matrix_summary is artifact_final_eval_matrix_summary
    assert learning_progress_module._periodic_dev_eval_trend is artifact_periodic_dev_eval_trend
    assert learning_progress_module._promotion_gate_summary is artifact_promotion_gate_summary
    assert artifact_final_eval_matrix_summary.__module__ == "weiss_rl.diagnostics.progress.learning_progress_artifacts"


def test_learning_progress_guard_helpers_are_package_owned() -> None:
    assert evaluate_league_guard is guard_evaluate_league_guard
    assert learning_progress_module.DEFAULT_LEAGUE_GUARD_ANCHORS is guard_default_league_guard_anchors
    assert guard_evaluate_league_guard.__module__ == "weiss_rl.diagnostics.progress.learning_progress_guard"


def test_learning_progress_metric_sections_are_package_owned() -> None:
    assert learning_progress_module._window_summary is metric_window_summary
    assert learning_progress_module.build_training_log_summary_sections is metric_build_training_log_summary_sections
    assert (
        metric_build_training_log_summary_sections.__module__
        == "weiss_rl.diagnostics.progress.learning_progress_metrics"
    )


def test_learning_progress_diagnostic_plan_is_package_owned() -> None:
    assert learning_progress_module.LEARNING_PROGRESS_DIAGNOSTIC_SECTIONS is section_plan
    assert learning_progress_module.learning_progress_diagnostic_plan_payload is section_plan_payload
    assert section_plan_payload()[0] == {
        "key": "loss",
        "question": "Is the optimizer reducing the learner objective?",
        "evidence": ["training_metrics.jsonl loss windows"],
    }


def test_learning_progress_summary_includes_reader_order(tmp_path) -> None:
    summary = build_overheated_training_summary(tmp_path)

    assert summary["diagnostic_plan"][0]["key"] == "loss"
    assert {section["key"] for section in summary["diagnostic_plan"]} >= {
        "reward_scale",
        "chosen_action_learning",
        "off_policy",
        "promotion_gate",
        "final_eval_matrix",
    }
