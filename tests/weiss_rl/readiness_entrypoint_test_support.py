from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .entrypoints_test_support import REPO_ROOT, _write_paper_readiness_run_dir_fixture


def write_two_policy_final_eval_fixture(tmp_path: Path) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = ["B0 RandomLegal", "policy_000300"]
    _write_matchup_diagnostics(
        final_eval_dir=final_eval_dir,
        matchup_paths=[
            "matchups/00_b0_randomlegal__vs__00_b0_randomlegal/diagnostics.json",
            "matchups/00_b0_randomlegal__vs__01_policy_000300/diagnostics.json",
            "matchups/01_policy_000300__vs__00_b0_randomlegal/diagnostics.json",
            "matchups/01_policy_000300__vs__01_policy_000300/diagnostics.json",
        ],
    )
    _write_summary(
        final_eval_dir=final_eval_dir,
        policies=policies,
        matrix_values={
            "games": [[2, 2], [2, 2]],
            "truncations": [[0, 0], [0, 0]],
            "mean": [[0.5, 0.0], [0.9, 0.5]],
            "ci_low": [[0.5, 0.0], [0.88, 0.5]],
            "ci_high": [[0.5, 0.0], [0.95, 0.5]],
            "has_payoff_samples": [[True, True], [True, True]],
            "paired_seed_count": [[1, 1], [2, 1]],
            "stop_reason": [["precision", "precision"], ["precision", "precision"]],
        },
        posterior_values=[[[], []], [[0.88, 0.9, 0.92, 0.95], []]],
        matchups=[
            _matchup("B0 RandomLegal", "B0 RandomLegal", "00_b0_randomlegal", "00_b0_randomlegal"),
            _matchup("B0 RandomLegal", "policy_000300", "00_b0_randomlegal", "01_policy_000300"),
            _matchup("policy_000300", "B0 RandomLegal", "01_policy_000300", "00_b0_randomlegal"),
            _matchup("policy_000300", "policy_000300", "01_policy_000300", "01_policy_000300"),
        ],
    )
    return final_eval_dir


def write_ambiguous_final_eval_fixture(tmp_path: Path) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = ["B0 RandomLegal", "policy_000300", "policy_000400"]
    matchups: list[dict[str, object]] = []

    for focal_index, focal_policy_id in enumerate(policies):
        for opponent_index, opponent_policy_id in enumerate(policies):
            focal_slug = _policy_slug(focal_policy_id)
            opponent_slug = _policy_slug(opponent_policy_id)
            diagnostics_path = (
                final_eval_dir
                / "matchups"
                / f"{focal_index:02d}_{focal_slug}__vs__"
                / f"{opponent_index:02d}_{opponent_slug}"
                / "diagnostics.json"
            )
            _write_diagnostics_json(diagnostics_path)
            matchups.append(
                {
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                    "focal_policy_index": focal_index,
                    "opponent_policy_index": opponent_index,
                    "diagnostics_path": (
                        f"matchups/{focal_index:02d}_{focal_slug}__vs__/"
                        f"{opponent_index:02d}_{opponent_slug}/diagnostics.json"
                    ),
                }
            )

    _write_summary(
        final_eval_dir=final_eval_dir,
        policies=policies,
        matrix_values={
            "games": [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "truncations": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "mean": [[0.5, 0.0, 0.0], [0.9, 0.5, 0.49], [0.94, 0.51, 0.5]],
            "ci_low": [[0.5, 0.0, 0.0], [0.88, 0.5, 0.45], [0.9, 0.5, 0.5]],
            "ci_high": [[0.5, 0.0, 0.0], [0.95, 0.5, 0.53], [0.97, 0.54, 0.5]],
            "has_payoff_samples": [[True, True, True], [True, True, True], [True, True, True]],
            "paired_seed_count": [[1, 1, 1], [2, 1, 1], [2, 1, 1]],
            "stop_reason": [
                ["precision", "precision", "precision"],
                ["precision", "precision", "precision"],
                ["precision", "precision", "precision"],
            ],
        },
        posterior_values=[
            [[], [], []],
            [[0.88, 0.9, 0.92, 0.95], [], [0.45, 0.48, 0.5, 0.53]],
            [[0.9, 0.93, 0.95, 0.97], [0.5, 0.51, 0.52, 0.54], []],
        ],
        matchups=matchups,
    )
    return final_eval_dir


def write_readiness_run_dir_fixture(tmp_path: Path) -> Path:
    return _write_paper_readiness_run_dir_fixture(tmp_path)


def run_readiness_entrypoint(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    return subprocess.run(
        [sys.executable, "-m", "weiss_rl.eval.readiness.check_entrypoint", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def _write_summary(
    *,
    final_eval_dir: Path,
    policies: list[str],
    matrix_values: dict[str, object],
    posterior_values: object,
    matchups: list[dict[str, object]],
) -> None:
    matrices = {name: {"policy_ids": policies, "values": values} for name, values in matrix_values.items()}
    payload = {
        "policy_ids": policies,
        "metadata": {"selection": {"mode": "deterministic_v1"}},
        "matrices": matrices,
        "posterior_samples": {
            "policy_ids": policies,
            "sample_count": 4,
            "values": posterior_values,
        },
        "matchups": matchups,
    }
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    (final_eval_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_matchup_diagnostics(*, final_eval_dir: Path, matchup_paths: list[str]) -> None:
    for diagnostics_path in matchup_paths:
        _write_diagnostics_json(final_eval_dir / diagnostics_path)


def _write_diagnostics_json(diagnostics_path: Path) -> None:
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


def _matchup(focal: str, opponent: str, focal_slug: str, opponent_slug: str) -> dict[str, str]:
    return {
        "focal_policy_id": focal,
        "opponent_policy_id": opponent,
        "diagnostics_path": f"matchups/{focal_slug}__vs__{opponent_slug}/diagnostics.json",
    }


def _policy_slug(policy_id: str) -> str:
    return policy_id.lower().replace(" ", "_")
