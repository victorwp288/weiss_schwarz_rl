from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import (
    EvalGameRecord,
    build_matchup_export,
    load_eval_game_records,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)

_CONFIG_HASH256 = "ab" * 32
_SPEC_HASH256 = "cd" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


def _pair(pair_index: int, outcome_a: OutcomeToken, outcome_b: OutcomeToken) -> list[EvalGameRecord]:
    episode_seed = pair_index + 100
    return [
        _record(pair_index, 0, outcome_a, episode_seed=episode_seed),
        _record(pair_index, 1, outcome_b, episode_seed=episode_seed),
    ]


def _record(
    pair_index: int,
    swap_index: int,
    outcome: OutcomeToken,
    *,
    episode_seed: int,
    focal_policy_id: str = "champion",
    opponent_policy_id: str = "baseline",
    run_id256: str | None = None,
) -> EvalGameRecord:
    if swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1

    episode_index = pair_index * 2 + swap_index
    episode_key64 = episode_index + 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=swap_index,
        episode_index=episode_index,
        episode_seed=episode_seed,
        episode_key=f"{episode_key64:064x}",
        episode_key64=episode_key64,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
        outcome=outcome,
        terminated=outcome != "T",
        truncated=outcome == "T",
        engine_status=0,
        run_id256=run_id256,
    )


def _write_jsonl(path: Path, records: list[EvalGameRecord]) -> None:
    path.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_eval_game_records_round_trips_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    records = [*_pair(0, "W", "L"), *_pair(1, "D", "T")]
    records[0] = replace(records[0], run_id256="12" * 32)
    _write_jsonl(path, records)

    loaded = load_eval_game_records(path)

    assert loaded == tuple(records)


def test_build_matchup_export_and_write_outputs(tmp_path: Path) -> None:
    payload = build_matchup_export(
        [*_pair(0, "W", "W"), *_pair(1, "W", "W")],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        max_paired_seeds=8,
        scheme="S0",
        sample_count=16,
        seed=7,
    )

    assert payload["focal_policy_id"] == "champion"
    assert payload["opponent_policy_id"] == "baseline"
    assert payload["deck_context"] == {"focal_deck": None, "opponent_deck": None}
    assert payload["summary"] == {
        "games": 4,
        "wins": 4,
        "losses": 0,
        "draws": 0,
        "truncations": 0,
        "engine_errors": 0,
        "natural_timeouts": 0,
        "no_progress_timeouts": 0,
        "decision_limit_timeouts": 0,
        "tick_limit_timeouts": 0,
        "timeout_unknown": 0,
        "total_actions": 0,
        "pass_actions": 0,
        "main_move_actions": 0,
        "pass_with_nonpass_available": 0,
        "max_consecutive_main_moves": 0,
    }
    assert payload["stop_reason"] == "decisive"
    assert payload["should_stop"] is True
    assert payload["observed_paired_seeds"] == 2
    assert payload["excluded_paired_seeds"] == 0
    assert payload["has_payoff_samples"] is True
    assert payload["uncertainty"]["paired_seed_count"] == 2

    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    write_matchup_summary_json(json_path, payload)
    write_matchup_summary_csv(csv_path, payload)

    written_json = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert written_json["stop_reason"] == "decisive"
    assert rows == [
        {
            "focal_policy_id": "champion",
            "opponent_policy_id": "baseline",
            "focal_deck": "",
            "opponent_deck": "",
            "scheme": "S0",
            "paired_seeds": "2",
            "max_paired_seeds": "8",
            "observed_paired_seeds": "2",
            "excluded_paired_seeds": "0",
            "has_payoff_samples": "True",
            "stop_reason": "decisive",
            "should_stop": "True",
            "games": "4",
            "wins": "4",
            "losses": "0",
            "draws": "0",
            "truncations": "0",
            "engine_errors": "0",
            "natural_timeouts": "0",
            "no_progress_timeouts": "0",
            "decision_limit_timeouts": "0",
            "tick_limit_timeouts": "0",
            "timeout_unknown": "0",
            "total_actions": "0",
            "pass_actions": "0",
            "main_move_actions": "0",
            "pass_with_nonpass_available": "0",
            "max_consecutive_main_moves": "0",
            "mean": "1.0",
            "ci_low": "1.0",
            "ci_high": "1.0",
            "ci_half_width": "0.0",
            "prob_gt_half": "1.0",
            "prob_lt_half": "0.0",
            "sample_count": "16",
        }
    ]


def test_build_matchup_export_rejects_mixed_matchups() -> None:
    with pytest.raises(ValueError, match="exactly one focal/opponent matchup"):
        build_matchup_export(
            [*_pair(0, "W", "L"), _record(1, 0, "W", episode_seed=101, opponent_policy_id="other")],
            stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
            max_paired_seeds=8,
        )


def test_build_matchup_export_rejects_mixed_config_hashes() -> None:
    records = [*_pair(0, "W", "L"), *_pair(1, "D", "T")]
    records[-1] = replace(records[-1], config_hash256="ef" * 32)

    with pytest.raises(ValueError, match="exactly one config/spec contract"):
        build_matchup_export(
            records,
            stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
            max_paired_seeds=8,
        )


def test_build_matchup_export_rejects_mixed_spec_hashes() -> None:
    records = [*_pair(0, "W", "L"), *_pair(1, "D", "T")]
    records[-1] = replace(records[-1], spec_hash256="01" * 32)

    with pytest.raises(ValueError, match="exactly one config/spec contract"):
        build_matchup_export(
            records,
            stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
            max_paired_seeds=8,
        )
