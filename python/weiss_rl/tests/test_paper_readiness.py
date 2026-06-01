from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.eval import (
    build_paper_readiness_summary,
    paper_readiness,
    paper_readiness_check_cli,
    paper_readiness_check_entrypoint,
    paper_readiness_check_reporting,
    paper_readiness_check_runtime,
    paper_readiness_contracts,
    paper_readiness_final_eval_summary,
    paper_readiness_guardrails,
)
from weiss_rl.eval.policy_set import RANDOM_LEGAL_POLICY_ID

SeatResultInput = tuple[int, int] | dict[str, int]


def _normalize_seat_result(value: SeatResultInput) -> dict[str, int]:
    if isinstance(value, tuple):
        seat0_wins, seat1_wins = value
        draws = 0
        truncations = 0
        engine_errors = 0
    else:
        seat0_wins = int(value.get("seat0_wins", 0))
        seat1_wins = int(value.get("seat1_wins", 0))
        draws = int(value.get("draws", 0))
        truncations = int(value.get("truncations", 0))
        engine_errors = int(value.get("engine_errors", 0))
    decisive_games = seat0_wins + seat1_wins
    return {
        "seat0_wins": seat0_wins,
        "seat1_wins": seat1_wins,
        "draws": draws,
        "truncations": truncations,
        "engine_errors": engine_errors,
        "decisive_games": decisive_games,
        "total_games": decisive_games + draws + truncations,
    }


def _write_final_eval_fixture(
    tmp_path: Path,
    *,
    candidate_samples: list[float],
    seat_results: list[SeatResultInput],
    games_matrix: list[list[int]] | None = None,
    truncation_matrix: list[list[int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = [RANDOM_LEGAL_POLICY_ID, "policy_000300"]
    candidate_mean = sum(candidate_samples) / len(candidate_samples) if candidate_samples else None
    candidate_ci_low = min(candidate_samples) if candidate_samples else None
    candidate_ci_high = max(candidate_samples) if candidate_samples else None
    matchups = [
        (0, 0, policies[0], policies[0]),
        (0, 1, policies[0], policies[1]),
        (1, 0, policies[1], policies[0]),
        (1, 1, policies[1], policies[1]),
    ]
    normalized_seat_results = [_normalize_seat_result(value) for value in seat_results]

    for (focal_index, opponent_index, focal_policy_id, opponent_policy_id), seat_result in zip(
        matchups,
        normalized_seat_results,
        strict=True,
    ):
        diagnostics_path = (
            final_eval_dir
            / "matchups"
            / f"{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
            / f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
            / "diagnostics.json"
        )
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(
                {"seat_results": seat_result},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    summary_payload = {
        "policy_ids": policies,
        "metadata": metadata or {"selection": {"mode": "deterministic_v1"}},
        "matrices": {
            "games": {
                "policy_ids": policies,
                "values": games_matrix
                or [
                    [normalized_seat_results[0]["total_games"], normalized_seat_results[1]["total_games"]],
                    [normalized_seat_results[2]["total_games"], normalized_seat_results[3]["total_games"]],
                ],
            },
            "truncations": {
                "policy_ids": policies,
                "values": truncation_matrix
                or [
                    [normalized_seat_results[0]["truncations"], normalized_seat_results[1]["truncations"]],
                    [normalized_seat_results[2]["truncations"], normalized_seat_results[3]["truncations"]],
                ],
            },
            "mean": {
                "policy_ids": policies,
                "values": [[0.5, 0.0], [candidate_mean, 0.5]],
            },
            "ci_low": {
                "policy_ids": policies,
                "values": [[0.5, 0.0], [candidate_ci_low, 0.5]],
            },
            "ci_high": {
                "policy_ids": policies,
                "values": [[0.5, 0.0], [candidate_ci_high, 0.5]],
            },
            "has_payoff_samples": {
                "policy_ids": policies,
                "values": [[True, True], [bool(candidate_samples), True]],
            },
            "paired_seed_count": {
                "policy_ids": policies,
                "values": [[1, 1], [2 if candidate_samples else 0, 1]],
            },
            "stop_reason": {
                "policy_ids": policies,
                "values": [["precision", "precision"], ["precision", "precision"]],
            },
        },
        "posterior_samples": {
            "policy_ids": policies,
            "sample_count": len(candidate_samples),
            "values": [[[], []], [candidate_samples, []]],
        },
        "matchups": [
            {
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
                "focal_policy_index": focal_index,
                "opponent_policy_index": opponent_index,
                "diagnostics_path": (
                    f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__/"
                    f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                ),
            }
            for focal_index, opponent_index, focal_policy_id, opponent_policy_id in matchups
        ],
    }
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    (final_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_eval_dir


def _write_multi_policy_final_eval_fixture(tmp_path: Path, *, metadata: dict[str, Any] | None = None) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = [RANDOM_LEGAL_POLICY_ID, "policy_000300", "policy_000400"]
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    matchups: list[dict[str, Any]] = []
    for focal_index, focal_policy_id in enumerate(policies):
        for opponent_index, opponent_policy_id in enumerate(policies):
            diagnostics_path = (
                final_eval_dir
                / "matchups"
                / f"{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                / f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
                / "diagnostics.json"
            )
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "seat_results": {
                            "seat0_wins": 1,
                            "seat1_wins": 1,
                            "draws": 0,
                            "truncations": 0,
                            "engine_errors": 0,
                            "decisive_games": 2,
                            "total_games": 2,
                        }
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            matchups.append(
                {
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                    "focal_policy_index": focal_index,
                    "opponent_policy_index": opponent_index,
                    "diagnostics_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__/"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                    ),
                }
            )

    values_float: list[list[float | None]] = [
        [0.5, 0.0, 0.0],
        [0.91, 0.5, 0.48],
        [0.94, 0.52, 0.5],
    ]
    summary_payload = {
        "policy_ids": policies,
        "metadata": metadata or {"selection": {"mode": "deterministic_v1"}},
        "matrices": {
            "games": {"policy_ids": policies, "values": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]},
            "truncations": {"policy_ids": policies, "values": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
            "mean": {"policy_ids": policies, "values": values_float},
            "ci_low": {"policy_ids": policies, "values": [[0.5, 0.0, 0.0], [0.88, 0.5, 0.45], [0.9, 0.5, 0.5]]},
            "ci_high": {"policy_ids": policies, "values": [[0.5, 0.0, 0.0], [0.95, 0.5, 0.51], [0.97, 0.55, 0.5]]},
            "has_payoff_samples": {
                "policy_ids": policies,
                "values": [[True, True, True], [True, True, True], [True, True, True]],
            },
            "paired_seed_count": {"policy_ids": policies, "values": [[1, 1, 1], [2, 1, 1], [2, 1, 1]]},
            "stop_reason": {
                "policy_ids": policies,
                "values": [
                    ["precision", "precision", "precision"],
                    ["precision", "precision", "precision"],
                    ["precision", "precision", "precision"],
                ],
            },
        },
        "posterior_samples": {
            "policy_ids": policies,
            "sample_count": 4,
            "values": [
                [[], [], []],
                [[0.88, 0.9, 0.92, 0.95], [], [0.45, 0.48, 0.5, 0.51]],
                [[0.9, 0.93, 0.95, 0.97], [0.5, 0.51, 0.53, 0.55], []],
            ],
        },
        "matchups": matchups,
    }
    (final_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_eval_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_dir_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_ready"
    policies = [RANDOM_LEGAL_POLICY_ID, "policy_000300"]
    manifest_payload = {
        "run_id256": "ab" * 32,
        "run_id64": "0123456789abcdef",
        "start_nonce": 7,
        "git_commit": "deadbeef" * 5,
        "git_dirty": False,
        "spec_hash256": "cd" * 32,
        "config_hash256": "ef" * 32,
        "simulator": {
            "version": "0.7.0",
            "compatibility_hash": "feedfacecafebeef",
        },
        "spec_bundle": {"version": 1, "cards": []},
        "config_canonical": {"stack": {"name": "synthetic"}},
        "seed_files": {
            "final_eval": {
                "path": "configs/seeds/report_eval_seeds.txt",
                "sha256": "12" * 32,
            }
        },
        "hardware": {"platform": "test", "cpu": "synthetic"},
        "evaluation_pinning": {"eval_sampling_algorithm": "pinned_cdf_pcg_v1"},
        "policy_set_selection": list(policies),
        "policy_set_selection_details": {"mode": "deterministic_v1"},
    }
    _write_json(run_dir / "manifest.json", manifest_payload)
    _write_json(run_dir / "spec_bundle.json", cast(dict[str, Any], manifest_payload["spec_bundle"]))
    _write_json(run_dir / "config_canonical.json", cast(dict[str, Any], manifest_payload["config_canonical"]))
    _write_json(
        run_dir / "environment.json",
        {
            "kind": "environment_manifest_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "run_id256": manifest_payload["run_id256"],
            "run_id64": manifest_payload["run_id64"],
        },
    )
    _write_json(
        run_dir / "run_summary.json",
        {
            "kind": "run_summary_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "runtime_mode": "train_ordered",
            "policy_set_selection_mode": "deterministic_v1",
        },
    )
    _write_json(
        run_dir / "determinism_report.json",
        {
            "kind": "determinism_report_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "policy_selection_mode": "deterministic_v1",
            "replay_verification": {
                "path": "eval/diagnostics/replay_verification.json",
                "status": "pending",
            },
        },
    )
    (run_dir / "spec_hash256.txt").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec_hash256.txt").write_text(str(manifest_payload["spec_hash256"]) + "\n", encoding="utf-8")
    (run_dir / "config_hash256.txt").write_text(str(manifest_payload["config_hash256"]) + "\n", encoding="utf-8")

    (run_dir / "training" / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").write_text(
        json.dumps(
            {
                "update_count": 1,
                "wall_clock_seconds": 1.0,
                "wall_clock_ms": 1000,
                "policy_version": 1,
                "loss": 0.8,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    final_eval_dir = run_dir / "eval" / "final_eval"
    canonical_matchups = [
        (0, 0, policies[0], policies[0]),
        (0, 1, policies[0], policies[1]),
        (1, 1, policies[1], policies[1]),
    ]
    matchup_rows: list[str] = [
        "focal_policy_id,opponent_policy_id,matchup_dir,paired_seed_count,observed_paired_seed_count,excluded_paired_seed_count,has_payoff_samples,stop_reason"
    ]
    for focal_index, opponent_index, focal_policy_id, opponent_policy_id in canonical_matchups:
        matchup_rel_dir = (
            f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
            f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
        )
        matchup_dir = final_eval_dir / matchup_rel_dir
        matchup_dir.mkdir(parents=True, exist_ok=True)
        (matchup_dir / "episodes.jsonl").write_text(
            json.dumps(
                {
                    "episode_seed": 11,
                    "config_hash256": manifest_payload["config_hash256"],
                    "spec_hash256": manifest_payload["spec_hash256"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_json(
            matchup_dir / "matchup_summary.json",
            {
                "paired_seeds": 2,
                "observed_paired_seeds": 2,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "precision",
            },
        )
        _write_json(
            matchup_dir / "diagnostics.json",
            {
                "seat_results": {
                    "seat0_wins": 1,
                    "seat1_wins": 1,
                    "draws": 0,
                    "truncations": 0,
                    "engine_errors": 0,
                    "decisive_games": 2,
                    "total_games": 2,
                }
            },
        )
        _write_json(
            matchup_dir / "posterior_samples.json",
            {
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
                "sample_count": 4,
                "samples": [0.88, 0.91, 0.93, 0.95],
            },
        )
        matchup_rows.append(
            ",".join(
                (
                    focal_policy_id,
                    opponent_policy_id,
                    matchup_rel_dir,
                    "2",
                    "2",
                    "0",
                    "True",
                    "precision",
                )
            )
        )

    _write_json(final_eval_dir / "metadata.json", {"selection": {"mode": "deterministic_v1"}})
    _write_json(final_eval_dir / "policy_set.json", {"policy_ids": list(policies)})
    _write_json(
        final_eval_dir / "posterior_samples.json",
        {
            "policy_ids": list(policies),
            "sample_count": 4,
            "values": [[[], []], [[0.88, 0.91, 0.93, 0.95], []]],
        },
    )
    (final_eval_dir / "matchups.csv").write_text("\n".join(matchup_rows) + "\n", encoding="utf-8")
    (final_eval_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (final_eval_dir / "matrices" / "mean.csv").write_text(
        "focal_policy_id,B0 RandomLegal,policy_000300\nB0 RandomLegal,0.5,0.0\npolicy_000300,0.9,0.5\n",
        encoding="utf-8",
    )
    _write_json(
        final_eval_dir / "summary.json",
        {
            "policy_ids": list(policies),
            "metadata": {"selection": {"mode": "deterministic_v1"}},
            "matrices": {
                "games": {"policy_ids": list(policies), "values": [[2, 2], [2, 2]]},
                "truncations": {"policy_ids": list(policies), "values": [[0, 0], [0, 0]]},
                "mean": {"policy_ids": list(policies), "values": [[0.5, 0.0], [0.9, 0.5]]},
                "ci_low": {"policy_ids": list(policies), "values": [[0.5, 0.0], [0.88, 0.5]]},
                "ci_high": {"policy_ids": list(policies), "values": [[0.5, 0.0], [0.95, 0.5]]},
                "has_payoff_samples": {"policy_ids": list(policies), "values": [[True, True], [True, True]]},
                "paired_seed_count": {"policy_ids": list(policies), "values": [[1, 1], [2, 1]]},
                "stop_reason": {
                    "policy_ids": list(policies),
                    "values": [["precision", "precision"], ["precision", "precision"]],
                },
            },
            "posterior_samples": {
                "policy_ids": list(policies),
                "sample_count": 4,
                "values": [[[], []], [[0.88, 0.91, 0.93, 0.95], []]],
            },
            "matchups": [
                {
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                    "focal_policy_index": focal_index,
                    "opponent_policy_index": opponent_index,
                    "matchup_dir": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
                    ),
                    "episodes_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/episodes.jsonl"
                    ),
                    "summary_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/matchup_summary.json"
                    ),
                    "diagnostics_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                    ),
                    "posterior_samples_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/posterior_samples.json"
                    ),
                }
                for focal_index, opponent_index, focal_policy_id, opponent_policy_id in canonical_matchups
            ],
        },
    )

    diagnostics_dir = run_dir / "eval" / "diagnostics"
    _write_json(
        diagnostics_dir / "seat_bias.json",
        {
            "global": {"seat0_win_rate": 0.5, "ci_low": 0.4, "ci_high": 0.6, "decisive_games": 6},
            "matchups": [
                {
                    "policy_a": "B0 RandomLegal",
                    "policy_b": "policy_000300",
                    "seat0_win_rate": 0.5,
                    "seat1_win_rate": 0.5,
                    "decisive_games": 2,
                }
            ],
        },
    )
    (diagnostics_dir / "truncation_heatmap_data.csv").parent.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "truncation_heatmap_data.csv").write_text(
        ",B0 RandomLegal,policy_000300\nB0 RandomLegal,0.0,0.0\npolicy_000300,0.0,0.0\n",
        encoding="utf-8",
    )
    _write_json(diagnostics_dir / "replay_verification.json", {"status": "ok", "checked_replays": 3})
    _write_json(
        final_eval_dir / "artifact_hashes.json",
        {
            "kind": "final_eval_artifact_hashes_v1",
            "artifacts": {"eval/final_eval/summary.json": "ab" * 32},
        },
    )

    sensitivity_dir = run_dir / "eval" / "metagame"
    _write_json(
        sensitivity_dir / "summary.json",
        {
            "policy_ids": list(policies),
            "cases": {case_id: {"summary_json": f"{case_id}/summary.json"} for case_id in ("S0", "S1", "S2")},
        },
    )
    for case_id in ("S0", "S1", "S2"):
        _write_json(sensitivity_dir / case_id / "summary.json", {"case_id": case_id})
        (sensitivity_dir / case_id / "payoff").mkdir(parents=True, exist_ok=True)
        (sensitivity_dir / case_id / "payoff" / "matchups.csv").write_text(
            "focal_policy_id,opponent_policy_id,p_mean\nB0 RandomLegal,policy_000300,0.1\n",
            encoding="utf-8",
        )
        (sensitivity_dir / case_id / "nash").mkdir(parents=True, exist_ok=True)
        (sensitivity_dir / case_id / "nash" / "mixture_mean.csv").write_text(
            "policy_id,mean_mixture\nB0 RandomLegal,0.5\npolicy_000300,0.5\n",
            encoding="utf-8",
        )
        (sensitivity_dir / case_id / "alpharank").mkdir(parents=True, exist_ok=True)
        (sensitivity_dir / case_id / "alpharank" / "stationary_mean.csv").write_text(
            "policy_id,mean_stationary_mass\nB0 RandomLegal,0.5\npolicy_000300,0.5\n",
            encoding="utf-8",
        )

    figures_dir = run_dir / "figures" / "paper"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "fig_matchup_heatmap.pdf").write_text("pdf placeholder\n", encoding="utf-8")
    (figures_dir / "fig_matchup_heatmap.png").write_text("png placeholder\n", encoding="utf-8")
    return run_dir


def test_paper_readiness_facade_keeps_contract_helpers_aliased() -> None:
    assert paper_readiness.RequiredArtifactSpec is paper_readiness_contracts.RequiredArtifactSpec
    assert paper_readiness._build_run_directory_audit is paper_readiness_contracts.build_run_directory_audit
    assert paper_readiness._build_manifest_contract is paper_readiness_contracts.build_manifest_contract
    assert paper_readiness._build_final_eval_artifact_contract is (
        paper_readiness_contracts.build_final_eval_artifact_contract
    )


def test_paper_readiness_guardrails_keep_final_eval_summary_helpers_aliased() -> None:
    assert paper_readiness_guardrails.policy_ids is paper_readiness_final_eval_summary.policy_ids
    assert paper_readiness_guardrails.matchups is paper_readiness_final_eval_summary.matchups
    assert paper_readiness_guardrails.matchup_policy_index is paper_readiness_final_eval_summary.matchup_policy_index
    assert paper_readiness_guardrails.matrix_cell is paper_readiness_final_eval_summary.matrix_cell
    assert paper_readiness_guardrails.posterior_samples is paper_readiness_final_eval_summary.posterior_samples


def test_paper_readiness_check_entrypoint_keeps_cli_reporting_helpers_aliased() -> None:
    assert paper_readiness_check_entrypoint._closed_interval is paper_readiness_check_cli.closed_interval
    assert paper_readiness_check_entrypoint._default_readiness_json is paper_readiness_check_cli.default_readiness_json
    assert paper_readiness_check_entrypoint._format_alarm is paper_readiness_check_reporting.format_alarm
    assert paper_readiness_check_entrypoint._format_alarm_detail is paper_readiness_check_reporting.format_alarm_detail


def test_paper_readiness_check_parser_defaults_output_and_thresholds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ready"
    parser = paper_readiness_check_cli.build_paper_readiness_check_parser()

    args = parser.parse_args(["--run-dir", str(run_dir), "--focal-policy-id", "  policy_000300  "])

    assert args.run_dir == run_dir
    assert args.final_eval_dir is None
    assert args.readiness_json is None
    assert paper_readiness_check_cli.default_readiness_json(
        run_dir=args.run_dir,
        final_eval_dir=args.final_eval_dir,
    ) == (run_dir / "paper_readiness_summary.json")
    assert args.focal_policy_id == "  policy_000300  "
    assert args.max_truncation_rate == paper_readiness.DEFAULT_TRUNCATION_MAX_RATE
    assert args.baseline_policy_id == RANDOM_LEGAL_POLICY_ID


def test_paper_readiness_check_runtime_threads_args_and_writes_default_json(tmp_path: Path) -> None:
    final_eval_dir = tmp_path / "final_eval"
    parser = paper_readiness_check_cli.build_paper_readiness_check_parser()
    args = parser.parse_args(
        [
            "--final-eval-dir",
            str(final_eval_dir),
            "--focal-policy-id",
            "   ",
            "--baseline-win-rate-threshold",
            "0.61",
        ]
    )
    observed: dict[str, Any] = {}
    payload = {"passed": True, "alarms": []}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        observed["build"] = kwargs
        return payload

    def fake_write(path: Path, written_payload: dict[str, Any]) -> None:
        observed["write"] = (path, written_payload)

    result = paper_readiness_check_runtime.run_paper_readiness_check(
        args,
        build_paper_readiness_summary_fn=fake_build,
        write_paper_readiness_json_fn=fake_write,
    )

    assert result.readiness_json == final_eval_dir / "paper_readiness_summary.json"
    assert result.payload is payload
    assert observed["write"] == (final_eval_dir / "paper_readiness_summary.json", payload)
    assert observed["build"]["run_dir"] is None
    assert observed["build"]["final_eval_dir"] == final_eval_dir
    assert observed["build"]["focal_policy_id"] is None
    assert observed["build"]["baseline_win_rate_threshold"] == 0.61


def test_paper_readiness_check_reporting_formats_failure_sources() -> None:
    payload = {
        "passed": False,
        "alarms": ["baseline_win_rate_vs_b0", "manifest_contract", "final_eval_guardrails", "unknown_alarm"],
        "checks": {
            "baseline_win_rate_vs_b0": {
                "message": "pass --focal-policy-id to choose the focal policy explicitly",
            }
        },
        "manifest_contract": {"reason": "ValueError"},
        "final_eval_guardrails": {"message": "summary.json is missing"},
    }

    assert paper_readiness_check_reporting.format_failure_message(payload) == (
        "Paper readiness checks failed: "
        "baseline_win_rate_vs_b0 (pass --focal-policy-id to choose the focal policy explicitly), "
        "manifest_contract (ValueError), "
        "final_eval_guardrails (summary.json is missing), "
        "unknown_alarm"
    )


def test_paper_readiness_final_eval_summary_selects_canonical_unordered_matchups() -> None:
    policies = [RANDOM_LEGAL_POLICY_ID, "policy_000300", "policy_000400"]
    raw_matchups = [
        {
            "focal_policy_id": "policy_000300",
            "opponent_policy_id": RANDOM_LEGAL_POLICY_ID,
            "focal_policy_index": 1,
            "opponent_policy_index": 0,
            "diagnostics_path": "reciprocal.json",
        },
        {
            "focal_policy_id": RANDOM_LEGAL_POLICY_ID,
            "opponent_policy_id": "policy_000300",
            "focal_policy_index": 0,
            "opponent_policy_index": 1,
            "diagnostics_path": "canonical.json",
        },
        {
            "focal_policy_id": "policy_000400",
            "opponent_policy_id": "policy_000400",
            "focal_policy_index": 2,
            "opponent_policy_index": 2,
            "diagnostics_path": "self.json",
        },
    ]

    selected = paper_readiness_final_eval_summary.canonical_unordered_matchups(raw_matchups, policy_ids=policies)

    assert [matchup["diagnostics_path"] for matchup in selected] == ["canonical.json", "self.json"]


def test_paper_readiness_contract_catalog_keeps_thesis_artifact_surfaces() -> None:
    specs = {spec.artifact_id: spec for spec in paper_readiness_contracts.required_run_artifact_specs()}

    assert specs["training_metrics"].paths == (
        Path("training/logs/training_metrics.jsonl"),
        Path("eval/diagnostics/checkpoint_interpolation_summary.json"),
    )
    assert specs["sensitivity_summary"].paths == (
        Path("eval/metagame/summary.json"),
        Path("eval/final_eval/sensitivity/summary.json"),
    )
    assert specs["sensitivity_s0_nash_mixture"].paths == (
        Path("eval/metagame/S0/nash/mixture_mean.csv"),
        Path("eval/final_eval/sensitivity/S0/nash/mixture_mean.csv"),
    )
    assert specs["paper_figures_pdf"].glob == "figures/paper/*.pdf"
    assert specs["paper_figures_png"].glob == "figures/paper/*.png"


def test_build_paper_readiness_summary_passes_with_balanced_seats_and_strong_b0_winrate(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(
        tmp_path,
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        seat_results=[(1, 1), (1, 1), (1, 1), (1, 1)],
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    assert payload["passed"] is True
    assert payload["alarms"] == []
    assert payload["checks"]["truncation_rate"]["rate"] == 0.0
    assert payload["checks"]["seat_bias_alarm"]["alarm"] is False
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_source"] == "sole_eligible_non_baseline"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["prob_gt_threshold"] == 1.0


def test_build_paper_readiness_summary_flags_truncation_seat_bias_and_weak_b0_matchup(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(
        tmp_path,
        candidate_samples=[0.49, 0.5, 0.52, 0.54],
        seat_results=[
            (2, 0),
            {"seat0_wins": 2, "seat1_wins": 0, "truncations": 1},
            (2, 0),
            (2, 0),
        ],
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    assert payload["passed"] is False
    assert payload["alarms"] == [
        "truncation_rate",
        "seat_bias_alarm",
        "baseline_win_rate_vs_b0",
    ]
    assert payload["checks"]["truncation_rate"]["rate"] == 1 / 7
    assert payload["checks"]["seat_bias_alarm"]["alarm"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["passed"] is False
    assert payload["checks"]["baseline_win_rate_vs_b0"]["prob_gt_threshold"] == 0.0


def test_build_paper_readiness_summary_ignores_reciprocal_matchups_for_guardrail_aggregation(tmp_path: Path) -> None:
    baseline_dir = _write_final_eval_fixture(
        tmp_path / "baseline",
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        seat_results=[(1, 1), (1, 1), (1, 1), (1, 1)],
    )
    reciprocal_noise_dir = _write_final_eval_fixture(
        tmp_path / "reciprocal_noise",
        candidate_samples=[0.88, 0.91, 0.93, 0.95],
        seat_results=[
            (1, 1),
            (1, 1),
            {"seat0_wins": 20, "seat1_wins": 0, "truncations": 20},
            (1, 1),
        ],
        games_matrix=[[2, 2], [40, 2]],
        truncation_matrix=[[0, 0], [20, 0]],
    )

    baseline_payload = build_paper_readiness_summary(final_eval_dir=baseline_dir)
    reciprocal_noise_payload = build_paper_readiness_summary(final_eval_dir=reciprocal_noise_dir)

    assert reciprocal_noise_payload["passed"] == baseline_payload["passed"]
    assert reciprocal_noise_payload["alarms"] == baseline_payload["alarms"]
    assert reciprocal_noise_payload["checks"]["truncation_rate"] == baseline_payload["checks"]["truncation_rate"]
    assert (
        reciprocal_noise_payload["checks"]["seat_bias_alarm"]["observed"]
        == baseline_payload["checks"]["seat_bias_alarm"]["observed"]
    )
    assert [
        matchup["diagnostics_path"] for matchup in reciprocal_noise_payload["checks"]["seat_bias_alarm"]["per_matchup"]
    ] == [
        "matchups/00_b0_randomlegal__vs__/00_b0_randomlegal/diagnostics.json",
        "matchups/00_b0_randomlegal__vs__/01_policy_000300/diagnostics.json",
        "matchups/01_policy_000300__vs__/01_policy_000300/diagnostics.json",
    ]


def test_build_paper_readiness_summary_requires_explicit_focal_policy_when_multiple_candidates(tmp_path: Path) -> None:
    final_eval_dir = _write_multi_policy_final_eval_fixture(tmp_path)

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is False
    assert payload["alarms"] == ["baseline_win_rate_vs_b0"]
    assert check["focal_policy_id"] is None
    assert check["reason"] == "ambiguous_non_baseline_focal_policy"
    assert check["eligible_non_baseline_policy_ids"] == ["policy_000300", "policy_000400"]
    assert "pass --focal-policy-id" in check["message"]


def test_build_paper_readiness_summary_uses_metadata_named_focal_policy(tmp_path: Path) -> None:
    final_eval_dir = _write_multi_policy_final_eval_fixture(
        tmp_path,
        metadata={
            "selection": {"mode": "deterministic_v1"},
            "focal_policy": {"policy_id": "policy_000400"},
        },
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is True
    assert check["focal_policy_id"] == "policy_000400"
    assert check["focal_policy_source"] == "metadata"
    assert check["passed"] is True
    assert check["prob_gt_threshold"] == 1.0


def test_build_paper_readiness_summary_uses_recommended_focal_policy_metadata(tmp_path: Path) -> None:
    final_eval_dir = _write_multi_policy_final_eval_fixture(
        tmp_path,
        metadata={
            "selection": {"mode": "deterministic_v1"},
            "recommended_focal_policy_id": "policy_000400",
        },
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is True
    assert check["focal_policy_id"] == "policy_000400"
    assert check["focal_policy_source"] == "metadata"
    assert check["passed"] is True
    assert check["prob_gt_threshold"] == 1.0


def test_build_paper_readiness_summary_ignores_baseline_aliases_and_heuristic_variants(tmp_path: Path) -> None:
    final_eval_dir = tmp_path / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    policies = [
        RANDOM_LEGAL_POLICY_ID,
        "b1_noleague_baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
        "policy_000300",
    ]
    matchups: list[dict[str, Any]] = []
    values: list[list[float]] = []
    ci_low: list[list[float]] = []
    ci_high: list[list[float]] = []
    games: list[list[int]] = []
    truncations: list[list[int]] = []
    for focal_index, focal_policy_id in enumerate(policies):
        row_values: list[float] = []
        row_ci_low: list[float] = []
        row_ci_high: list[float] = []
        row_games: list[int] = []
        row_truncations: list[int] = []
        for opponent_index, opponent_policy_id in enumerate(policies):
            diagnostics_path = (
                final_eval_dir
                / "matchups"
                / f"{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                / f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
                / "diagnostics.json"
            )
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "seat_results": {
                            "seat0_wins": 6,
                            "seat1_wins": 6,
                            "draws": 0,
                            "truncations": 0,
                            "engine_errors": 0,
                            "decisive_games": 12,
                            "total_games": 12,
                        }
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            matchups.append(
                {
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                    "focal_policy_index": focal_index,
                    "opponent_policy_index": opponent_index,
                    "diagnostics_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                        f"/{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                    ),
                }
            )
            if focal_policy_id == "policy_000300" and opponent_policy_id == RANDOM_LEGAL_POLICY_ID:
                row_values.append(0.9)
                row_ci_low.append(0.82)
                row_ci_high.append(0.96)
            elif focal_policy_id == RANDOM_LEGAL_POLICY_ID and opponent_policy_id == "policy_000300":
                row_values.append(0.1)
                row_ci_low.append(0.04)
                row_ci_high.append(0.18)
            else:
                row_values.append(0.5)
                row_ci_low.append(0.5)
                row_ci_high.append(0.5)
            row_games.append(12)
            row_truncations.append(0)
        values.append(row_values)
        ci_low.append(row_ci_low)
        ci_high.append(row_ci_high)
        games.append(row_games)
        truncations.append(row_truncations)

    summary_payload = {
        "policy_ids": policies,
        "metadata": {"selection": {"mode": "deterministic_v1"}},
        "matrices": {
            "games": {"policy_ids": policies, "values": games},
            "truncations": {"policy_ids": policies, "values": truncations},
            "mean": {"policy_ids": policies, "values": values},
            "ci_low": {"policy_ids": policies, "values": ci_low},
            "ci_high": {"policy_ids": policies, "values": ci_high},
            "has_payoff_samples": {"policy_ids": policies, "values": [[True] * len(policies) for _ in policies]},
            "paired_seed_count": {"policy_ids": policies, "values": [[12] * len(policies) for _ in policies]},
            "stop_reason": {"policy_ids": policies, "values": [["precision"] * len(policies) for _ in policies]},
        },
        "posterior_samples": {
            "policy_ids": policies,
            "sample_count": 4,
            "values": [
                [
                    [0.5, 0.5, 0.5, 0.5]
                    if row == column
                    else ([0.1, 0.08, 0.12, 0.14] if row == 0 and column == len(policies) - 1 else [])
                    for column in range(len(policies))
                ]
                for row in range(len(policies))
            ],
        },
        "matchups": matchups,
    }
    posterior_samples = cast(dict[str, Any], summary_payload["posterior_samples"])
    posterior_values = cast(list[list[list[float]]], posterior_samples["values"])
    posterior_values[-1][0] = [0.9, 0.92, 0.88, 0.94]
    (final_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = build_paper_readiness_summary(final_eval_dir=final_eval_dir)

    check = payload["checks"]["baseline_win_rate_vs_b0"]
    assert payload["passed"] is True
    assert check["focal_policy_id"] == "policy_000300"
    assert check["focal_policy_source"] == "sole_eligible_non_baseline"


def test_build_paper_readiness_summary_audits_run_directory_artifacts(tmp_path: Path) -> None:
    run_dir = _write_run_dir_fixture(tmp_path)

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["scope"] == "run_dir"
    assert payload["passed"] is True
    assert payload["alarms"] == []
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True
    assert payload["final_eval_guardrails"]["passed"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"


def test_build_paper_readiness_summary_accepts_interpolation_provenance_instead_of_training_metrics(
    tmp_path: Path,
) -> None:
    run_dir = _write_run_dir_fixture(tmp_path)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").unlink()
    _write_json(
        run_dir / "eval" / "diagnostics" / "checkpoint_interpolation_summary.json",
        {
            "first_checkpoint": "runs/source_a/training/checkpoints/checkpoint_10.pt",
            "second_checkpoint": "runs/source_b/training/checkpoints/checkpoint_5.pt",
            "second_weight": 0.15,
        },
    )

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is True
    assert payload["run_directory_audit"]["artifacts"]["training_metrics"]["resolved_path"] == (
        "eval/diagnostics/checkpoint_interpolation_summary.json"
    )


def test_build_paper_readiness_summary_accepts_documented_unresolved_manifest_policy_selection(tmp_path: Path) -> None:
    run_dir = _write_run_dir_fixture(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_set_selection"] = []
    manifest["policy_set_selection_details"] = {
        "status": "unresolved",
        "version": "deterministic_v1",
        "final_policy_set_size": 10,
        "source_paths": {
            "snapshot_registry_json": None,
            "dev_eval_summaries_json": None,
        },
        "missing_inputs": ["snapshot_registry_json", "dev_eval_summaries_json"],
        "reason": "deterministic final policy set inputs were not provided",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is False
    assert "manifest_contract" in payload["alarms"]
    assert payload["manifest_contract"]["passed"] is False
    assert payload["manifest_contract"]["fields"]["policy_set_selection"]["passed"] is False


def test_build_paper_readiness_summary_reports_out_of_range_matchup_indices(tmp_path: Path) -> None:
    run_dir = _write_run_dir_fixture(tmp_path)
    summary_path = run_dir / "eval" / "final_eval" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["matchups"][0]["focal_policy_index"] = len(summary["policy_ids"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)
    contract = payload["final_eval_artifact_contract"]

    assert payload["passed"] is False
    assert payload["alarms"] == ["final_eval_artifact_contract", "final_eval_guardrails"]
    assert contract["passed"] is False
    assert contract["reason"] == "invalid_matchup_index"
    assert contract["reference_failures"] == [
        "matchups[0].focal_policy_index=2 is out of range for policy_ids with length 2"
    ]


def test_build_paper_readiness_summary_reports_negative_matchup_indices(tmp_path: Path) -> None:
    run_dir = _write_run_dir_fixture(tmp_path)
    summary_path = run_dir / "eval" / "final_eval" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["matchups"][0]["opponent_policy_index"] = -1
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)
    contract = payload["final_eval_artifact_contract"]

    assert payload["passed"] is False
    assert payload["alarms"] == ["final_eval_artifact_contract", "final_eval_guardrails"]
    assert contract["passed"] is False
    assert contract["reason"] == "invalid_matchup_index"
    assert contract["reference_failures"] == [
        "matchups[0].opponent_policy_index=-1 is out of range for policy_ids with length 2"
    ]


def test_build_paper_readiness_summary_flags_run_directory_gaps(tmp_path: Path) -> None:
    run_dir = _write_run_dir_fixture(tmp_path)
    (run_dir / "eval" / "diagnostics" / "replay_verification.json").unlink()
    _write_json(
        run_dir / "eval" / "final_eval" / "policy_set.json",
        {"policy_ids": ["policy_000300", RANDOM_LEGAL_POLICY_ID]},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["policy_set_selection"] = []
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is False
    assert payload["alarms"] == [
        "run_directory_audit",
        "manifest_contract",
        "final_eval_artifact_contract",
    ]
    assert "diagnostics_replay_verification" in payload["run_directory_audit"]["missing_artifacts"]
    assert payload["manifest_contract"]["fields"]["policy_set_selection"]["passed"] is False
    assert payload["final_eval_artifact_contract"]["policy_set"]["passed"] is False
