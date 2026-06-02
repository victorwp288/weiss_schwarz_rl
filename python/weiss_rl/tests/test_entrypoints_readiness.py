from __future__ import annotations

from .test_entrypoints import (
    REPO_ROOT,
    Path,
    _write_paper_readiness_run_dir_fixture,
    json,
    os,
    subprocess,
    sys,
)


def test_paper_readiness_entrypoint_writes_summary_json(tmp_path: Path) -> None:
    final_eval_dir = tmp_path / "final_eval"
    summary_path = final_eval_dir / "summary.json"
    readiness_json = final_eval_dir / "paper_readiness_summary.json"
    diagnostics_paths = [
        final_eval_dir / "matchups" / "00_b0_randomlegal__vs__00_b0_randomlegal" / "diagnostics.json",
        final_eval_dir / "matchups" / "00_b0_randomlegal__vs__01_policy_000300" / "diagnostics.json",
        final_eval_dir / "matchups" / "01_policy_000300__vs__00_b0_randomlegal" / "diagnostics.json",
        final_eval_dir / "matchups" / "01_policy_000300__vs__01_policy_000300" / "diagnostics.json",
    ]
    for diagnostics_path in diagnostics_paths:
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

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "policy_ids": ["B0 RandomLegal", "policy_000300"],
                "metadata": {"selection": {"mode": "deterministic_v1"}},
                "matrices": {
                    "games": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[2, 2], [2, 2]]},
                    "truncations": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0, 0], [0, 0]]},
                    "mean": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0.5, 0.0], [0.9, 0.5]]},
                    "ci_low": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0.5, 0.0], [0.88, 0.5]]},
                    "ci_high": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0.5, 0.0], [0.95, 0.5]]},
                    "has_payoff_samples": {
                        "policy_ids": ["B0 RandomLegal", "policy_000300"],
                        "values": [[True, True], [True, True]],
                    },
                    "paired_seed_count": {
                        "policy_ids": ["B0 RandomLegal", "policy_000300"],
                        "values": [[1, 1], [2, 1]],
                    },
                    "stop_reason": {
                        "policy_ids": ["B0 RandomLegal", "policy_000300"],
                        "values": [["precision", "precision"], ["precision", "precision"]],
                    },
                },
                "posterior_samples": {
                    "policy_ids": ["B0 RandomLegal", "policy_000300"],
                    "sample_count": 4,
                    "values": [[[], []], [[0.88, 0.9, 0.92, 0.95], []]],
                },
                "matchups": [
                    {
                        "focal_policy_id": "B0 RandomLegal",
                        "opponent_policy_id": "B0 RandomLegal",
                        "diagnostics_path": "matchups/00_b0_randomlegal__vs__00_b0_randomlegal/diagnostics.json",
                    },
                    {
                        "focal_policy_id": "B0 RandomLegal",
                        "opponent_policy_id": "policy_000300",
                        "diagnostics_path": "matchups/00_b0_randomlegal__vs__01_policy_000300/diagnostics.json",
                    },
                    {
                        "focal_policy_id": "policy_000300",
                        "opponent_policy_id": "B0 RandomLegal",
                        "diagnostics_path": "matchups/01_policy_000300__vs__00_b0_randomlegal/diagnostics.json",
                    },
                    {
                        "focal_policy_id": "policy_000300",
                        "opponent_policy_id": "policy_000300",
                        "diagnostics_path": "matchups/01_policy_000300__vs__01_policy_000300/diagnostics.json",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.eval.readiness.check_entrypoint",
            "--final-eval-dir",
            str(final_eval_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"


def test_paper_readiness_entrypoint_requires_explicit_focal_policy_for_ambiguous_multi_policy_artifacts(
    tmp_path: Path,
) -> None:
    final_eval_dir = tmp_path / "final_eval"
    summary_path = final_eval_dir / "summary.json"
    readiness_json = final_eval_dir / "paper_readiness_summary.json"
    policies = ["B0 RandomLegal", "policy_000300", "policy_000400"]
    matchups: list[dict[str, object]] = []

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

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "policy_ids": policies,
                "metadata": {"selection": {"mode": "deterministic_v1"}},
                "matrices": {
                    "games": {"policy_ids": policies, "values": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]},
                    "truncations": {"policy_ids": policies, "values": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
                    "mean": {
                        "policy_ids": policies,
                        "values": [[0.5, 0.0, 0.0], [0.9, 0.5, 0.49], [0.94, 0.51, 0.5]],
                    },
                    "ci_low": {
                        "policy_ids": policies,
                        "values": [[0.5, 0.0, 0.0], [0.88, 0.5, 0.45], [0.9, 0.5, 0.5]],
                    },
                    "ci_high": {
                        "policy_ids": policies,
                        "values": [[0.5, 0.0, 0.0], [0.95, 0.5, 0.53], [0.97, 0.54, 0.5]],
                    },
                    "has_payoff_samples": {
                        "policy_ids": policies,
                        "values": [[True, True, True], [True, True, True], [True, True, True]],
                    },
                    "paired_seed_count": {
                        "policy_ids": policies,
                        "values": [[1, 1, 1], [2, 1, 1], [2, 1, 1]],
                    },
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
                        [[0.88, 0.9, 0.92, 0.95], [], [0.45, 0.48, 0.5, 0.53]],
                        [[0.9, 0.93, 0.95, 0.97], [0.5, 0.51, 0.52, 0.54], []],
                    ],
                },
                "matchups": matchups,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.eval.readiness.check_entrypoint",
            "--final-eval-dir",
            str(final_eval_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "pass --focal-policy-id" in result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["checks"]["baseline_win_rate_vs_b0"]["reason"] == "ambiguous_non_baseline_focal_policy"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["eligible_non_baseline_policy_ids"] == [
        "policy_000300",
        "policy_000400",
    ]


def test_paper_readiness_entrypoint_accepts_run_dir_and_writes_run_summary(tmp_path: Path) -> None:
    run_dir = _write_paper_readiness_run_dir_fixture(tmp_path)
    readiness_json = run_dir / "paper_readiness_summary.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.eval.readiness.check_entrypoint",
            "--run-dir",
            str(run_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["scope"] == "run_dir"
    assert payload["passed"] is True
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True
