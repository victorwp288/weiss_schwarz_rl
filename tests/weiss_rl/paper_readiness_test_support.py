from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.eval.policies.set import RANDOM_LEGAL_POLICY_ID

SeatResultInput = tuple[int, int] | dict[str, int]


def normalize_seat_result(value: SeatResultInput) -> dict[str, int]:
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


def write_final_eval_fixture(
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
    normalized_seat_results = [normalize_seat_result(value) for value in seat_results]

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


def write_multi_policy_final_eval_fixture(tmp_path: Path, *, metadata: dict[str, Any] | None = None) -> Path:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run_dir_fixture(tmp_path: Path) -> Path:
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
    write_json(run_dir / "manifest.json", manifest_payload)
    write_json(run_dir / "spec_bundle.json", cast(dict[str, Any], manifest_payload["spec_bundle"]))
    write_json(run_dir / "config_canonical.json", cast(dict[str, Any], manifest_payload["config_canonical"]))
    write_json(
        run_dir / "environment.json",
        {
            "kind": "environment_manifest_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "run_id256": manifest_payload["run_id256"],
            "run_id64": manifest_payload["run_id64"],
        },
    )
    write_json(
        run_dir / "run_summary.json",
        {
            "kind": "run_summary_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "runtime_mode": "train_ordered",
            "policy_set_selection_mode": "deterministic_v1",
        },
    )
    write_json(
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
        write_json(
            matchup_dir / "matchup_summary.json",
            {
                "paired_seeds": 2,
                "observed_paired_seeds": 2,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "precision",
            },
        )
        write_json(
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
        write_json(
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

    write_json(final_eval_dir / "metadata.json", {"selection": {"mode": "deterministic_v1"}})
    write_json(final_eval_dir / "policy_set.json", {"policy_ids": list(policies)})
    write_json(
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
    write_json(
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
    write_json(
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
    write_json(diagnostics_dir / "replay_verification.json", {"status": "ok", "checked_replays": 3})
    write_json(
        final_eval_dir / "artifact_hashes.json",
        {
            "kind": "final_eval_artifact_hashes_v1",
            "artifacts": {"eval/final_eval/summary.json": "ab" * 32},
        },
    )

    sensitivity_dir = run_dir / "eval" / "metagame"
    write_json(
        sensitivity_dir / "summary.json",
        {
            "policy_ids": list(policies),
            "cases": {case_id: {"summary_json": f"{case_id}/summary.json"} for case_id in ("S0", "S1", "S2")},
        },
    )
    for case_id in ("S0", "S1", "S2"):
        write_json(sensitivity_dir / case_id / "summary.json", {"case_id": case_id})
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
