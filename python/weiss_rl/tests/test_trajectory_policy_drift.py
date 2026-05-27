from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from weiss_rl.diagnostics.trajectory_policy_drift import (
    summarize_policy_drift,
    summarize_policy_drift_by_group,
    summarize_policy_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY_NAMES = ("pass", "play", "attack")


def test_trajectory_policy_drift_requires_fixed_pythonhashseed() -> None:
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run(
        [
            sys.executable,
            "python/scripts/trajectory_policy_drift.py",
            "--stack-config",
            "missing.yaml",
            "--dataset",
            "missing.npz",
            "--policy",
            "direct|runs/missing|training/checkpoints/latest.pt",
            "--output-json",
            "diagnostics/missing.json",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr


def test_summarize_policy_scores_reports_target_alignment_by_family() -> None:
    summary = summarize_policy_scores(
        label="policy",
        top_actions=np.asarray([1, 2, 3, 4]),
        target_actions=np.asarray([1, 3, 3, 4]),
        target_probabilities=np.asarray([0.8, 0.2, 0.7, 0.6]),
        target_log_probs=np.log(np.asarray([0.8, 0.2, 0.7, 0.6])),
        top_families=np.asarray([1, 1, 2, 2]),
        target_families=np.asarray([1, 2, 2, 2]),
        row_mask=np.asarray([True, True, True, False]),
        family_names=FAMILY_NAMES,
        values=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )

    assert summary["row_count"] == 3
    assert summary["top_action_matches_target_rate"] == pytest.approx(2 / 3)
    assert summary["top_family_matches_target_rate"] == pytest.approx(2 / 3)
    assert summary["mean_probability_on_target_action"] == pytest.approx((0.8 + 0.2 + 0.7) / 3)
    assert summary["value_percentiles"]["mean"] == pytest.approx(0.2)
    assert summary["target_family_summaries"][0]["family"] == "attack"


def test_summarize_policy_drift_identifies_lost_target_top_actions_and_drops() -> None:
    summary = summarize_policy_drift(
        reference_label="direct",
        candidate_label="update10",
        reference_top_actions=np.asarray([1, 3, 5, 7]),
        candidate_top_actions=np.asarray([2, 3, 4, 7]),
        reference_target_probabilities=np.asarray([0.7, 0.6, 0.8, 0.1]),
        candidate_target_probabilities=np.asarray([0.1, 0.7, 0.3, 0.1]),
        reference_top_families=np.asarray([1, 2, 2, 0]),
        candidate_top_families=np.asarray([1, 2, 1, 0]),
        target_actions=np.asarray([1, 3, 5, 7]),
        target_families=np.asarray([1, 2, 2, 0]),
        row_mask=np.asarray([True, True, True, False]),
        family_names=FAMILY_NAMES,
        candidate_target_log_probs=np.asarray([-1.0, -0.5, -2.0, -0.1]),
        candidate_top_log_probs=np.asarray([-0.9999995, -0.5, -1.99995, -0.1]),
        reference_values=np.asarray([0.6, 0.5, 0.4, 0.3]),
        candidate_values=np.asarray([0.2, 0.6, 0.1, 0.3]),
        row_coordinates=[
            {"row_index": 0, "step_index": 0},
            {"row_index": 1, "step_index": 1},
            {"row_index": 2, "step_index": 2},
            {"row_index": 3, "step_index": 3},
        ],
        max_examples=1,
    )

    assert summary["row_count"] == 3
    assert summary["top_action_changed_rate"] == pytest.approx(2 / 3)
    assert summary["top_family_changed_rate"] == pytest.approx(1 / 3)
    assert summary["lost_target_top_action_rate"] == pytest.approx(2 / 3)
    assert summary["gained_target_top_action_rate"] == pytest.approx(0.0)
    assert summary["mean_target_action_probability_delta"] == pytest.approx((-0.6 + 0.1 - 0.5) / 3)
    assert summary["mean_value_delta"] == pytest.approx((-0.4 + 0.1 - 0.3) / 3)
    assert summary["lost_target_top_action_same_family_rate"] == pytest.approx(0.5)
    assert summary["lost_target_top_action_abs_probability_delta_lte_1e-5_count"] == 0
    margin_summary = summary["lost_target_top_action_candidate_top_over_target_margin"]
    assert margin_summary["count"] == 2
    assert margin_summary["near_tie_thresholds"][0]["count"] == 1
    assert margin_summary["near_tie_thresholds"][2]["count"] == 2
    assert summary["largest_target_probability_drops"][0]["row_index"] == 0
    assert summary["lost_target_top_action_examples"][0]["candidate_top_over_target_logp_margin"] == pytest.approx(
        5e-5
    )


def test_summarize_policy_drift_by_group_splits_rows_by_label() -> None:
    summaries = summarize_policy_drift_by_group(
        group_name="role",
        group_labels=np.asarray(["preferred", "preferred", "rejected", ""]),
        reference_label="direct",
        candidate_label="update10",
        reference_top_actions=np.asarray([1, 3, 5, 7]),
        candidate_top_actions=np.asarray([2, 3, 5, 7]),
        reference_target_probabilities=np.asarray([0.7, 0.6, 0.8, 0.1]),
        candidate_target_probabilities=np.asarray([0.1, 0.7, 0.9, 0.1]),
        reference_top_families=np.asarray([1, 2, 2, 0]),
        candidate_top_families=np.asarray([1, 2, 2, 0]),
        target_actions=np.asarray([1, 3, 5, 7]),
        target_families=np.asarray([1, 2, 2, 0]),
        row_mask=np.asarray([True, True, True, True]),
        family_names=FAMILY_NAMES,
    )

    by_role = {str(summary["role"]): summary for summary in summaries}
    assert set(by_role) == {"preferred", "rejected"}
    assert by_role["preferred"]["row_count"] == 2
    assert by_role["preferred"]["lost_target_top_action_rate"] == pytest.approx(0.5)
    assert by_role["rejected"]["row_count"] == 1
    assert by_role["rejected"]["mean_target_action_probability_delta"] == pytest.approx(0.1)
