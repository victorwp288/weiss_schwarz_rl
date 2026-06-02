"""Writer for the minimal thesis-grade run tree used by readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout

__all__ = ["write_paper_readiness_run_fixture"]

_POLICY_IDS = ("B0 RandomLegal", "policy_000300")
_SENSITIVITY_CASE_IDS = ("S0", "S1", "S2")


def write_paper_readiness_run_fixture(run_dir: Path) -> Path:
    """Write a canonical run directory that satisfies the paper-readiness contract."""

    resolved_run_dir = Path(run_dir).resolve()
    layout = ArtifactLayout.from_run_dir(resolved_run_dir)
    layout.ensure_directories()

    manifest = {
        "run_id256": "ab" * 32,
        "run_id64": "0123456789abcdef",
        "start_nonce": 7,
        "git_commit": "deadbeef" * 5,
        "git_dirty": False,
        "spec_hash256": "cd" * 32,
        "config_hash256": "ef" * 32,
        "simulator": {"version": "0.7.0", "compatibility_hash": "feedfacecafebeef"},
        "spec_bundle": {"version": 1, "cards": []},
        "config_canonical": {"stack": {"name": "synthetic"}},
        "seed_files": {"final_eval": {"path": "configs/seeds/report_eval_seeds.txt", "sha256": "12" * 32}},
        "hardware": {"platform": "test", "cpu": "synthetic"},
        "evaluation_pinning": {"eval_sampling_algorithm": "pinned_cdf_pcg_v1"},
        "policy_set_selection": list(_POLICY_IDS),
        "policy_set_selection_details": {"mode": "deterministic_v1"},
    }
    _write_json(layout.manifest_path, manifest)
    _write_json(layout.spec_bundle_path, cast(dict[str, Any], manifest["spec_bundle"]))
    _write_json(layout.config_json_path, cast(dict[str, Any], manifest["config_canonical"]))
    _write_json(
        layout.environment_path,
        {
            "kind": "environment_manifest_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "run_id256": manifest["run_id256"],
            "run_id64": manifest["run_id64"],
        },
    )
    _write_json(
        layout.run_summary_path,
        {
            "kind": "run_summary_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "runtime_mode": "train_ordered",
            "policy_set_selection_mode": "deterministic_v1",
        },
    )
    _write_json(
        layout.determinism_report_path,
        {
            "kind": "determinism_report_v1",
            "artifact_schema_version": "run_artifacts_v2",
            "policy_selection_mode": "deterministic_v1",
            "replay_verification": {
                "path": layout.relative(layout.replay_verification_json()),
                "status": "pending",
            },
        },
    )
    _write_text(layout.spec_hash_path, f"{manifest['spec_hash256']}\n")
    _write_text(layout.config_hash_path, f"{manifest['config_hash256']}\n")
    _write_text(
        layout.training_logs_dir / "training_metrics.jsonl",
        json.dumps({"loss": 0.8, "policy_version": 1, "update_count": 1}, sort_keys=True) + "\n",
    )

    _write_final_eval_tree(layout)
    _write_diagnostics(layout)
    _write_metagame_tree(layout)
    _write_paper_figures(layout)
    return resolved_run_dir


def _write_final_eval_tree(layout: ArtifactLayout) -> None:
    matchups: list[dict[str, object]] = []
    matchup_rows = [
        (
            "focal_policy_id,opponent_policy_id,matchup_dir,paired_seed_count,"
            "observed_paired_seed_count,excluded_paired_seed_count,has_payoff_samples,stop_reason"
        )
    ]
    canonical_matchups = (
        (0, 0, _POLICY_IDS[0], _POLICY_IDS[0]),
        (0, 1, _POLICY_IDS[0], _POLICY_IDS[1]),
        (1, 1, _POLICY_IDS[1], _POLICY_IDS[1]),
    )

    for focal_index, opponent_index, focal_policy_id, opponent_policy_id in canonical_matchups:
        prefix = (
            f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
            f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
        )
        matchup_dir = layout.final_eval_dir / prefix
        matchup_dir.mkdir(parents=True, exist_ok=True)
        _write_text(matchup_dir / "episodes.jsonl", json.dumps({"episode_seed": 11}, sort_keys=True) + "\n")
        _write_json(
            matchup_dir / "matchup_summary.json",
            {
                "has_payoff_samples": True,
                "observed_paired_seeds": 2,
                "paired_seeds": 2,
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
                }
            },
        )
        _write_json(
            matchup_dir / "posterior_samples.json",
            {"sample_count": 4, "samples": [0.88, 0.91, 0.93, 0.95]},
        )
        matchups.append(
            {
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
                "focal_policy_index": focal_index,
                "opponent_policy_index": opponent_index,
                "matchup_dir": prefix,
                "episodes_path": prefix + "/episodes.jsonl",
                "summary_path": prefix + "/matchup_summary.json",
                "diagnostics_path": prefix + "/diagnostics.json",
                "posterior_samples_path": prefix + "/posterior_samples.json",
            }
        )
        matchup_rows.append(",".join((focal_policy_id, opponent_policy_id, prefix, "2", "2", "0", "True", "precision")))

    _write_json(layout.final_eval_policy_set_json(), {"policy_ids": list(_POLICY_IDS)})
    _write_json(
        layout.final_eval_posterior_samples_json(),
        {
            "policy_ids": list(_POLICY_IDS),
            "sample_count": 4,
            "values": [[[], []], [[0.88, 0.91, 0.93, 0.95], []]],
        },
    )
    _write_text(layout.final_eval_matchups_csv(), "\n".join(matchup_rows) + "\n")
    _write_text(
        layout.final_eval_matrix_csv("mean"),
        "focal_policy_id,B0 RandomLegal,policy_000300\nB0 RandomLegal,0.5,0.0\npolicy_000300,0.9,0.5\n",
    )
    _write_json(
        layout.final_eval_summary_json(),
        {
            "policy_ids": list(_POLICY_IDS),
            "metadata": {"selection": {"mode": "deterministic_v1"}},
            "matrices": {
                "games": {"policy_ids": list(_POLICY_IDS), "values": [[2, 2], [2, 2]]},
                "truncations": {"policy_ids": list(_POLICY_IDS), "values": [[0, 0], [0, 0]]},
                "mean": {"policy_ids": list(_POLICY_IDS), "values": [[0.5, 0.0], [0.9, 0.5]]},
                "ci_low": {"policy_ids": list(_POLICY_IDS), "values": [[0.5, 0.0], [0.88, 0.5]]},
                "ci_high": {"policy_ids": list(_POLICY_IDS), "values": [[0.5, 0.0], [0.95, 0.5]]},
                "has_payoff_samples": {"policy_ids": list(_POLICY_IDS), "values": [[True, True], [True, True]]},
                "paired_seed_count": {"policy_ids": list(_POLICY_IDS), "values": [[1, 1], [2, 1]]},
                "stop_reason": {
                    "policy_ids": list(_POLICY_IDS),
                    "values": [["precision", "precision"], ["precision", "precision"]],
                },
            },
            "posterior_samples": {
                "policy_ids": list(_POLICY_IDS),
                "sample_count": 4,
                "values": [[[], []], [[0.88, 0.91, 0.93, 0.95], []]],
            },
            "matchups": matchups,
        },
    )
    _write_json(
        layout.final_eval_aggregate_hashes_json(),
        {
            "kind": "final_eval_artifact_hashes_v1",
            "artifacts": {"eval/final_eval/summary.json": "ab" * 32},
        },
    )


def _write_diagnostics(layout: ArtifactLayout) -> None:
    _write_json(
        layout.seat_bias_json(),
        {
            "global": {"seat0_win_rate": 0.5, "ci_low": 0.4, "ci_high": 0.6, "decisive_games": 6},
            "matchups": [
                {
                    "policy_a": _POLICY_IDS[0],
                    "policy_b": _POLICY_IDS[1],
                    "seat0_win_rate": 0.5,
                    "seat1_win_rate": 0.5,
                    "decisive_games": 2,
                }
            ],
        },
    )
    _write_text(
        layout.truncation_heatmap_csv(),
        ",B0 RandomLegal,policy_000300\nB0 RandomLegal,0.0,0.0\npolicy_000300,0.0,0.0\n",
    )
    _write_json(layout.replay_verification_json(), {"status": "ok"})


def _write_metagame_tree(layout: ArtifactLayout) -> None:
    _write_json(
        layout.metagame_dir / "summary.json",
        {"policy_ids": list(_POLICY_IDS), "cases": {case_id: {} for case_id in _SENSITIVITY_CASE_IDS}},
    )
    for case_id in _SENSITIVITY_CASE_IDS:
        case_dir = layout.metagame_dir / case_id
        (case_dir / "payoff").mkdir(parents=True, exist_ok=True)
        (case_dir / "nash").mkdir(parents=True, exist_ok=True)
        (case_dir / "alpharank").mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "summary.json", {"case_id": case_id})
        _write_text(
            case_dir / "payoff" / "matchups.csv",
            "focal_policy_id,opponent_policy_id,p_mean\nB0 RandomLegal,policy_000300,0.1\n",
        )
        _write_text(
            case_dir / "nash" / "mixture_mean.csv",
            "policy_id,mean_mixture\nB0 RandomLegal,0.5\npolicy_000300,0.5\n",
        )
        _write_text(
            case_dir / "alpharank" / "stationary_mean.csv",
            "policy_id,mean_stationary_mass\nB0 RandomLegal,0.5\npolicy_000300,0.5\n",
        )


def _write_paper_figures(layout: ArtifactLayout) -> None:
    _write_text(layout.figures_paper_dir / "fig_matchup_heatmap.pdf", "pdf\n")
    _write_text(layout.figures_paper_dir / "fig_matchup_heatmap.png", "png\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
