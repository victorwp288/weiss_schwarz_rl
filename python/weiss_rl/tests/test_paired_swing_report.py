from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS
from weiss_rl.experiments.paired_swing_report import (
    PairedSwingReportConfig,
    build_paired_swing_report,
)
from weiss_rl.experiments.paired_swing_report_reporting import (
    paired_swing_report_output_payload,
)
from weiss_rl.experiments.paired_swing_report_runtime import paired_swing_report_config_from_args


def _write_compare_json(path: Path) -> Path:
    payload = {
        "baseline": {"label": "selected"},
        "candidate": {"label": "candidate"},
        "rows": [
            {
                "opponent_policy_id": "B2 HeuristicPublic",
                "status": "ok",
                "shared_games": 2,
                "baseline_wins": 1,
                "candidate_wins": 2,
                "delta_wins": 1,
                "baseline_nonwin_candidate_win": 1,
                "baseline_win_candidate_nonwin": 0,
                "changed_outcome": 1,
                "examples": [
                    {
                        "pair_index": 1,
                        "swap_index": 0,
                        "episode_seed": 101,
                        "baseline_outcome": "L",
                        "candidate_outcome": "W",
                        "baseline_decision_count": 20,
                        "candidate_decision_count": 22,
                        "baseline_pass_actions": 3,
                        "candidate_pass_actions": 2,
                        "baseline_pass_with_nonpass_available": 1,
                        "candidate_pass_with_nonpass_available": 0,
                    }
                ],
            },
            {
                "opponent_policy_id": "seed_imported_policy_000001",
                "status": "ok",
                "shared_games": 2,
                "baseline_wins": 1,
                "candidate_wins": 0,
                "delta_wins": -1,
                "baseline_nonwin_candidate_win": 0,
                "baseline_win_candidate_nonwin": 1,
                "changed_outcome": 1,
                "examples": [
                    {
                        "pair_index": 2,
                        "swap_index": 1,
                        "episode_seed": 202,
                        "baseline_outcome": "W",
                        "candidate_outcome": "L",
                        "baseline_decision_count": 30,
                        "candidate_decision_count": 29,
                        "baseline_pass_actions": 5,
                        "candidate_pass_actions": 7,
                        "baseline_pass_with_nonpass_available": 1,
                        "candidate_pass_with_nonpass_available": 2,
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_paired_swing_report_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_swing_report,
        paired_swing_report_cli,
        paired_swing_report_entrypoint,
        paired_swing_report_runtime,
    )

    assert paired_swing_report_entrypoint._build_parser is paired_swing_report_cli.build_paired_swing_report_parser
    assert paired_swing_report_entrypoint.run_paired_swing_report is paired_swing_report_runtime.run_paired_swing_report
    assert paired_swing_report_entrypoint.PairedSwingReportConfig is paired_swing_report.PairedSwingReportConfig
    assert paired_swing_report_entrypoint.build_paired_swing_report is paired_swing_report.build_paired_swing_report
    assert paired_swing_report_entrypoint.write_paired_swing_report is paired_swing_report.write_paired_swing_report


def test_paired_swing_report_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_swing_report_cli import build_paired_swing_report_parser

    args = build_paired_swing_report_parser().parse_args(
        [
            "--compare-json",
            str(tmp_path / "compare.json"),
            "--output-json",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.compare_json == [tmp_path / "compare.json"]
    assert args.opponent_pool_jsonl == []
    assert args.fixed_opponent is None
    assert args.learned_opponent == []
    assert args.max_examples_per_bucket == 24
    assert args.notes == ""
    assert args.output_json == tmp_path / "report.json"


def test_paired_swing_report_runtime_maps_args_and_resolves_inputs(tmp_path: Path) -> None:
    args = SimpleNamespace(
        compare_json=[tmp_path / "compare.json"],
        opponent_pool_jsonl=[tmp_path / "pool.jsonl"],
        fixed_opponent=None,
        learned_opponent=["policy_000004"],
        max_examples_per_bucket=9,
        notes="unit",
    )

    config = paired_swing_report_config_from_args(args)

    assert config == PairedSwingReportConfig(
        compare_jsons=((tmp_path / "compare.json").resolve(),),
        opponent_pool_jsonls=((tmp_path / "pool.jsonl").resolve(),),
        fixed_opponents=FIXED_THESIS_OPPONENTS,
        learned_opponents=("policy_000004",),
        max_examples_per_bucket=9,
        notes="unit",
    )


def test_paired_swing_report_reporting_preserves_compact_console_payload(tmp_path: Path) -> None:
    report = {
        "aggregate": {
            "groups": {
                "all": {"delta_wins": 2},
                "fixed": {"delta_wins": 1},
                "learned": {"delta_wins": -1},
                "hard_negative": {"delta_wins": -2},
            }
        }
    }

    assert paired_swing_report_output_payload(output_json=tmp_path / "report.json", report=report) == {
        "output_json": (tmp_path / "report.json").as_posix(),
        "all_delta_wins": 2,
        "fixed_delta_wins": 1,
        "learned_delta_wins": -1,
        "hard_negative_delta_wins": -2,
    }


def test_paired_swing_report_summarizes_fixed_learned_and_pool_tags(tmp_path: Path) -> None:
    compare_json = _write_compare_json(tmp_path / "compare.json")
    pool_jsonl = tmp_path / "opponent_pool.jsonl"
    pool_jsonl.write_text(
        json.dumps(
            {
                "hard_negative_ids": ["seed_imported_policy_000001"],
                "champion_ids": ["seed_imported_policy_000001"],
                "recent_ids": ["policy_000003"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_paired_swing_report(
        PairedSwingReportConfig(
            compare_jsons=(compare_json,),
            opponent_pool_jsonls=(pool_jsonl,),
            fixed_opponents=("B2 HeuristicPublic",),
            learned_opponents=("seed_imported_policy_000001",),
        )
    )

    aggregate = report["aggregate"]["groups"]
    assert aggregate["all"]["delta_wins"] == 0
    assert aggregate["fixed"]["delta_wins"] == 1
    assert aggregate["learned"]["delta_wins"] == -1
    assert aggregate["hard_negative"]["delta_wins"] == -1
    assert aggregate["champion"]["delta_wins"] == -1
    assert report["pool_tags"]["seed_imported_policy_000001"] == ["champion", "hard_negative"]

    targets = report["comparisons"][0]["repair_targets"]
    assert targets["fixed_gain"][0]["episode_seed"] == 101
    assert targets["hard_negative_regression"][0]["episode_seed"] == 202
    assert targets["hard_negative_regression"][0]["pass_actions_delta"] == 2
    seed_plan = report["comparisons"][0]["replay_seed_plan"]
    assert seed_plan["fixed_gain"]["B2 HeuristicPublic"] == [101]
    assert seed_plan["hard_negative_regression"]["seed_imported_policy_000001"] == [202]


def test_paired_swing_report_matches_seed_wrapped_pool_tags(tmp_path: Path) -> None:
    payload = {
        "baseline": {"label": "selected"},
        "candidate": {"label": "candidate"},
        "rows": [
            {
                "opponent_policy_id": "seed_outer_seed_inner_policy_000002",
                "status": "ok",
                "shared_games": 2,
                "baseline_wins": 1,
                "candidate_wins": 0,
                "delta_wins": -1,
                "baseline_nonwin_candidate_win": 0,
                "baseline_win_candidate_nonwin": 1,
                "changed_outcome": 1,
                "examples": [
                    {
                        "pair_index": 3,
                        "swap_index": 0,
                        "episode_seed": 303,
                        "baseline_outcome": "W",
                        "candidate_outcome": "L",
                        "baseline_decision_count": 24,
                        "candidate_decision_count": 25,
                        "baseline_pass_actions": 2,
                        "candidate_pass_actions": 4,
                        "baseline_pass_with_nonpass_available": 0,
                        "candidate_pass_with_nonpass_available": 1,
                    }
                ],
            }
        ],
    }
    compare_json = tmp_path / "compare.json"
    compare_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pool_jsonl = tmp_path / "opponent_pool.jsonl"
    pool_jsonl.write_text(
        json.dumps({"hard_negative_ids": ["seed_inner_policy_000002"]}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_paired_swing_report(
        PairedSwingReportConfig(
            compare_jsons=(compare_json,),
            opponent_pool_jsonls=(pool_jsonl,),
            learned_opponents=("seed_inner_policy_000002",),
        )
    )

    aggregate = report["aggregate"]["groups"]
    assert aggregate["learned"]["delta_wins"] == -1
    assert aggregate["hard_negative"]["delta_wins"] == -1

    targets = report["comparisons"][0]["repair_targets"]
    assert targets["hard_negative_regression"][0]["episode_seed"] == 303
    seed_plan = report["comparisons"][0]["replay_seed_plan"]
    assert seed_plan["hard_negative_regression"]["seed_outer_seed_inner_policy_000002"] == [303]
