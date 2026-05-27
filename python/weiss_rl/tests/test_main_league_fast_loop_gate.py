from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.main_league_fast_loop_gate import (
    MainLeagueFastLoopGateConfig,
    evaluate_main_league_fast_loop_gate,
)


def _write_mechanistic_gate(path: Path, *, passed: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "paired_swing_mechanistic_gate_v1",
                "passed": passed,
                "failures": [] if passed else ["row_worsened_fraction_above:0.3>0.15"],
                "summary": {"row_count": 10},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_scorecard(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "main_league_frontier_scorecard_v1",
                "entries": entries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_drift_gate(path: Path, *, passed: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "trajectory_policy_drift_gate_v1",
                "passed": passed,
                "failures": [] if passed else ["lost_target_top_action_rate_above:0.1>0"],
                "summary": {"lost_target_top_action_rate": 0.0 if passed else 0.1},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_live_progress_gate(path: Path, *, passed: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "main_league_live_probe_gate_v1",
                "passed": passed,
                "failures": []
                if passed
                else [{"reason": "hard_negative_exposure_below_min", "value": 0, "threshold": 1}],
                "summary": {
                    "exposure_totals": {"pfsp_champion_envs": 4, "pfsp_hard_negative_envs": 0 if not passed else 3}
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_target_gate(path: Path, *, passed: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "paired_flip_targets_gate_v1",
                "passed": passed,
                "failures": [] if passed else ["target_count_below:1<2"],
                "summary": {"target_count": 2 if passed else 1},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _entry(candidate_label: str, decision: str) -> dict:
    return {
        "candidate_label": candidate_label,
        "panel_kind": "sentinel",
        "paired_seeds": 64,
        "escalation": {"decision": decision, "reason": "test"},
    }


def test_fast_loop_gate_allows_sentinel_after_mechanistic_pass(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    drift = _write_drift_gate(tmp_path / "drift.json", passed=True)

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(stage="sentinel", mechanistic_gate_json=mechanistic, drift_gate_json=drift)
    )

    assert report["passed"] is True
    assert report["required_decision"] is None
    assert report["target_gate"]["passed"] is True
    assert report["drift_gate"]["passed"] is True


def test_fast_loop_gate_blocks_sentinel_without_drift_or_live_progress_gate(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(stage="sentinel", mechanistic_gate_json=mechanistic)
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "missing_drift_or_live_progress_gate_json"


def test_fast_loop_gate_allows_sentinel_after_live_progress_gate_passes(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    live_progress = _write_live_progress_gate(tmp_path / "live_progress.json", passed=True)

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="sentinel",
            mechanistic_gate_json=mechanistic,
            live_progress_gate_json=live_progress,
        )
    )

    assert report["passed"] is True
    assert report["live_progress_gate"]["passed"] is True


def test_fast_loop_gate_blocks_sentinel_when_live_progress_gate_fails(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    live_progress = _write_live_progress_gate(tmp_path / "live_progress.json", passed=False)

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="sentinel",
            mechanistic_gate_json=mechanistic,
            live_progress_gate_json=live_progress,
        )
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "live_progress_gate_failed"


def test_fast_loop_gate_blocks_sentinel_when_mechanistic_fails(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=False)
    drift = _write_drift_gate(tmp_path / "drift.json", passed=True)

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(stage="sentinel", mechanistic_gate_json=mechanistic, drift_gate_json=drift)
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "mechanistic_gate_failed"


def test_fast_loop_gate_blocks_sentinel_when_drift_gate_fails(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    drift = _write_drift_gate(tmp_path / "drift.json", passed=False)

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="sentinel",
            mechanistic_gate_json=mechanistic,
            drift_gate_json=drift,
        )
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "drift_gate_failed"


def test_fast_loop_gate_blocks_sentinel_when_target_gate_fails(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    target = _write_target_gate(tmp_path / "target.json", passed=False)
    drift = _write_drift_gate(tmp_path / "drift.json", passed=True)

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="sentinel",
            mechanistic_gate_json=mechanistic,
            target_gate_json=target,
            drift_gate_json=drift,
        )
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "target_gate_failed"


def test_fast_loop_gate_requires_sentinel_scorecard_before_full_confirm64(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    scorecard = _write_scorecard(tmp_path / "scorecard.json", [_entry("candidate", "run_full_confirm64")])

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="full_confirm64",
            mechanistic_gate_json=mechanistic,
            frontier_scorecard_json=scorecard,
        )
    )

    assert report["passed"] is True
    assert report["scorecard_entry"]["candidate_label"] == "candidate"


def test_fast_loop_gate_blocks_confirm128_without_full_confirm64_pass(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    scorecard = _write_scorecard(tmp_path / "scorecard.json", [_entry("candidate", "run_full_confirm64")])

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="confirm128",
            mechanistic_gate_json=mechanistic,
            frontier_scorecard_json=scorecard,
        )
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "wrong_escalation_decision"
    assert report["failures"][0]["required_decision"] == "run_confirm128"


def test_fast_loop_gate_selects_scorecard_entry_by_candidate_label(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    scorecard = _write_scorecard(
        tmp_path / "scorecard.json",
        [
            _entry("failed_candidate", "stop"),
            _entry("survivor", "run_confirm256"),
        ],
    )

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="confirm256",
            mechanistic_gate_json=mechanistic,
            frontier_scorecard_json=scorecard,
            candidate_label="survivor",
        )
    )

    assert report["passed"] is True
    assert report["scorecard_entry"]["candidate_label"] == "survivor"


def test_fast_loop_gate_rejects_ambiguous_scorecard_without_candidate_label(tmp_path: Path) -> None:
    mechanistic = _write_mechanistic_gate(tmp_path / "mechanistic.json", passed=True)
    scorecard = _write_scorecard(
        tmp_path / "scorecard.json",
        [
            _entry("candidate_a", "run_confirm128"),
            _entry("candidate_b", "run_confirm128"),
        ],
    )

    report = evaluate_main_league_fast_loop_gate(
        MainLeagueFastLoopGateConfig(
            stage="confirm128",
            mechanistic_gate_json=mechanistic,
            frontier_scorecard_json=scorecard,
        )
    )

    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "scorecard_entry_not_found"
