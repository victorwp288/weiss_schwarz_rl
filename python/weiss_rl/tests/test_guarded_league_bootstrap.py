from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from weiss_rl.experiments.guarded_league_bootstrap import (
    GuardedLeagueBootstrapConfig,
    LeagueSegmentRuntime,
    build_targeted_confirm_command,
    evaluate_guard,
    latest_policy_snapshot,
    recent_policy_snapshots,
    run_guarded_league_bootstrap,
    runtime_overrides_with_defaults,
)
from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_registry(run_dir: Path, *, snapshots: list[tuple[str, int]]) -> None:
    _write_json(
        run_dir / "training" / "snapshots" / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {
                    "policy_id": policy_id,
                    "update": update,
                    "path": f"training/snapshots/{policy_id}/weights.pt",
                    "weights_sha256": f"sha-{policy_id}",
                }
                for policy_id, update in snapshots
            ],
        },
    )
    for _policy_id, update in snapshots:
        checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_selection(path: Path, *, run_dir: Path, policy_id: str, update: int, score: float = 0.62) -> None:
    _write_json(
        path,
        {
            "selected": {
                "run_dir": run_dir.as_posix(),
                "run_name": run_dir.name,
                "snapshot_policy_id": policy_id,
                "update_count": update,
                "eligible": True,
                "selection_score": score,
                "selection_paired_seeds": 4,
                "selection_anchor_scores": {
                    "B1 NoLeague baseline": 0.55,
                    "B2 HeuristicPublic": 0.59,
                    "B3 HeuristicPublicAggro": 0.56,
                    "B4 HeuristicPublicControl": 0.57,
                },
            },
            "run_summaries": [{"run_name": run_dir.name, "latest_minus_best": -0.03}],
        },
    )


def test_runtime_overrides_keep_pinned_seed_defaults_with_custom_overrides() -> None:
    overrides = runtime_overrides_with_defaults(
        [
            "training.structured_aux.policy_anchor_coef=0.2",
            "league.pool.seed_snapshot_import_filter=pinned_or_source_champions",
        ]
    )

    assert "league.pool.seed_snapshot_champion_import=pinned" in overrides
    assert "league.pool.seed_snapshot_import_filter=pinned" not in overrides
    assert "league.pool.seed_snapshot_import_filter=pinned_or_source_champions" in overrides
    assert overrides[-1] == "league.pool.seed_snapshot_import_filter=pinned_or_source_champions"


def test_runtime_overrides_can_use_stack_seed_snapshot_policy_without_pinned_defaults() -> None:
    overrides = runtime_overrides_with_defaults(
        ["training.structured_aux.policy_anchor_coef=0.2"],
        apply_seed_snapshot_defaults=False,
    )

    assert overrides == ("training.structured_aux.policy_anchor_coef=0.2",)
    assert "league.pool.seed_snapshot_champion_import=pinned" not in overrides
    assert "league.pool.seed_snapshot_import_filter=pinned" not in overrides


def test_latest_policy_snapshot_ignores_imported_seed_aliases(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "segment"
    _write_registry(
        run_dir,
        snapshots=[
            ("seed_imported_selected", 0),
            ("policy_000001", 5),
            ("policy_000002", 10),
        ],
    )

    latest = latest_policy_snapshot(run_dir)

    assert latest.policy_id == "policy_000002"
    assert latest.update == 10
    assert latest.checkpoint_path == run_dir / "training" / "checkpoints" / "checkpoint_10.pt"


def test_recent_policy_snapshots_returns_chronological_train_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "segment"
    _write_registry(
        run_dir,
        snapshots=[
            ("policy_000001", 1),
            ("seed_imported_policy_000009", 99),
            ("policy_000002", 2),
            ("policy_000003", 3),
        ],
    )

    recent = recent_policy_snapshots(run_dir, count=2)

    assert [candidate.policy_id for candidate in recent] == ["policy_000002", "policy_000003"]
    assert [candidate.update for candidate in recent] == [2, 3]


def test_evaluate_guard_rejects_anchor_below_reference_drop() -> None:
    guard = evaluate_guard(
        scores={
            "B2 HeuristicPublic": 0.59,
            "B3 HeuristicPublicAggro": 0.49,
            "B4 HeuristicPublicControl": 0.55,
        },
        required_anchors=("B2 HeuristicPublic", "B3 HeuristicPublicAggro", "B4 HeuristicPublicControl"),
        min_required_anchor_score=0.5,
        reference_anchor_scores={"B3 HeuristicPublicAggro": 0.57},
        max_reference_drop=0.04,
    )

    assert guard["passed"] is False
    assert guard["failures"] == [
        {
            "anchor": "B3 HeuristicPublicAggro",
            "reason": "below_min_required_anchor_score",
            "score": 0.49,
            "threshold": 0.5,
        },
        {
            "anchor": "B3 HeuristicPublicAggro",
            "reason": "below_reference_drop_limit",
            "score": 0.49,
            "reference": 0.57,
            "delta": -0.07999999999999996,
            "threshold": -0.04,
        },
    ]


def test_guarded_league_bootstrap_dry_run_writes_train_and_confirm_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_demo",
            segments=3,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            first_init_schedule_offset_updates=0,
            dry_run=True,
        )
    )

    assert summary["status"] == "planned"
    assert len(summary["segments"]) == 1
    segment = summary["segments"][0]
    assert segment["source_checkpoint"].endswith("runs/seed/training/checkpoints/checkpoint_25.pt")
    assert _option(segment["train_command"]["argv"], "--run-label") == "guarded_demo_seg01"
    assert _option(segment["train_command"]["argv"], "--b1-baseline-run-dir").endswith("runs/seed")
    assert _option(segment["train_command"]["argv"], "--init-schedule-offset-updates") == "0"
    assert "preselect_command" in segment
    assert "targeted_confirm_command_template" in segment
    assert "final_selector_command" in segment
    assert "publish_selector_command" in segment
    written = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert written["status"] == "planned"


def test_guarded_league_bootstrap_can_use_separate_b1_baseline_run_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    b1_run_dir = tmp_path / "runs" / "locked_b1"

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            b1_baseline_run_dir=b1_run_dir,
            run_prefix="guarded_separate_b1",
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            dry_run=True,
        )
    )

    segment = summary["segments"][0]
    assert summary["b1_baseline_run_dir"].endswith("runs/locked_b1")
    assert _option(segment["train_command"]["argv"], "--seed-snapshot-run-dir").endswith("runs/seed")
    assert _option(segment["train_command"]["argv"], "--b1-baseline-run-dir").endswith("runs/locked_b1")


def test_targeted_confirm_command_uses_separate_b1_baseline_run_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    segment_run_dir = tmp_path / "runs" / "segment"
    b1_run_dir = tmp_path / "runs" / "locked_b1"

    command = build_targeted_confirm_command(
        config=GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=tmp_path / "runs" / "seed",
            b1_baseline_run_dir=b1_run_dir,
            confirm_paired_seeds=64,
        ),
        run_dir=segment_run_dir,
        focal_policy_id="policy_000003",
        output_subdir="confirm_policy_000003",
    )

    assert _option(command, "--run-dir").endswith("runs/segment")
    assert _option(command, "--b1-baseline-run-dir").endswith("runs/locked_b1")


def test_targeted_confirm_command_includes_learned_guard_opponents(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    segment_run_dir = tmp_path / "runs" / "segment"

    command = build_targeted_confirm_command(
        config=GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=tmp_path / "runs" / "seed",
            confirm_opponents=("B1 NoLeague baseline", "B2 HeuristicPublic", "seed_champion"),
            learned_guard_opponents=("seed_champion", "seed_hard_negative"),
        ),
        run_dir=segment_run_dir,
        focal_policy_id="policy_000003",
        output_subdir="confirm_policy_000003",
    )

    opponents = [command[index + 1] for index, value in enumerate(command) if value == "--opponent"]
    assert opponents == [
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "seed_champion",
        "seed_hard_negative",
    ]


def test_guarded_league_bootstrap_stops_when_b3_guard_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((command, _.get("env")))
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "python/scripts/select_b1_candidate.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_json(
                output_json,
                {
                    "selected": {
                        "run_dir": run_dir.as_posix(),
                        "run_name": run_dir.name,
                        "snapshot_policy_id": "policy_000001",
                        "update_count": 5,
                        "eligible": True,
                        "selection_score": 0.50,
                        "selection_anchor_scores": {
                            "B1 NoLeague baseline": 0.55,
                            "B2 HeuristicPublic": 0.58,
                            "B3 HeuristicPublicAggro": 0.50,
                            "B4 HeuristicPublicControl": 0.56,
                        },
                    }
                },
            )
        elif "python/scripts/targeted_confirm_eval.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.55},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.58},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.45},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.56},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_fail",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=4,
            reference_anchor_scores={"B3 HeuristicPublicAggro": 0.55},
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "stopped_guard_failed"
    assert len(summary["segments"]) == 1
    guard = summary["segments"][0]["guard"]
    assert guard["passed"] is False
    assert guard["failures"][0]["anchor"] == "B3 HeuristicPublicAggro"


def test_guarded_league_bootstrap_stops_when_learned_multiobjective_guard_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    reference_summary = tmp_path / "diagnostics" / "selected_reference_confirm4_summary.json"
    _write_json(
        reference_summary,
        {
            "rows": [
                *[
                    {
                        "opponent_policy_id": opponent,
                        "wins": 5,
                        "games": 8,
                        "mean": 5 / 8,
                    }
                    for opponent in FIXED_THESIS_OPPONENTS
                ],
                {"opponent_policy_id": "seed_champion", "wins": 5, "games": 8, "mean": 5 / 8},
                {"opponent_policy_id": "seed_hard_negative", "wins": 5, "games": 8, "mean": 5 / 8},
            ]
        },
    )
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((command, _.get("env")))
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "python/scripts/select_b1_candidate.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_json(
                output_json,
                {
                    "selected": {
                        "run_dir": run_dir.as_posix(),
                        "run_name": run_dir.name,
                        "snapshot_policy_id": "policy_000001",
                        "update_count": 5,
                        "eligible": True,
                        "selection_score": 0.64,
                        "selection_anchor_scores": {
                            "B1 NoLeague baseline": 0.64,
                            "B2 HeuristicPublic": 0.64,
                            "B3 HeuristicPublicAggro": 0.64,
                            "B4 HeuristicPublicControl": 0.64,
                        },
                    }
                },
            )
        elif "python/scripts/targeted_confirm_eval.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "paired_seeds": paired_seeds,
                    "rows": [
                        *[
                            {
                                "opponent_policy_id": opponent,
                                "wins": 6,
                                "games": 8,
                                "mean": 6 / 8,
                            }
                            for opponent in FIXED_THESIS_OPPONENTS
                        ],
                        {"opponent_policy_id": "seed_champion", "wins": 4, "games": 8, "mean": 4 / 8},
                        {"opponent_policy_id": "seed_hard_negative", "wins": 4, "games": 8, "mean": 4 / 8},
                    ],
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_learned_gate_fail",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=4,
            confirm_opponents=(*FIXED_THESIS_OPPONENTS, "seed_champion", "seed_hard_negative"),
            publish_min_confirm_paired_seeds=256,
            continue_unpublished_confirmed=True,
            multiobjective_reference_summary_jsons=(reference_summary,),
            min_learned_guard_reference_delta=0.0,
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "stopped_multiobjective_guard_failed"
    assert len(summary["segments"]) == 1
    segment = summary["segments"][0]
    assert segment["status"] == "rejected"
    assert "publish_skipped" not in segment
    assert summary["learned_guard_opponents"] == ["seed_champion", "seed_hard_negative"]
    assert summary["learned_guard_opponents_inferred"] is True
    assert segment["multiobjective_guard"]["passed"] is False
    assert segment["multiobjective_guard"]["groups"]["fixed_baselines"]["mean"] == pytest.approx(0.75)
    assert segment["multiobjective_guard"]["groups"]["learned_opponents"]["reference_delta"] == pytest.approx(-0.125)
    train_commands = [command for command, _env in calls if "python/scripts/train.py" in command]
    train_envs = [env for command, env in calls if "python/scripts/train.py" in command]
    assert len(train_commands) == 1
    assert train_envs[0] is not None
    assert train_envs[0]["PYTHONHASHSEED"] == "0"


def test_guarded_league_bootstrap_writes_multiobjective_guard_when_legacy_guard_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    reference_summary = tmp_path / "diagnostics" / "selected_reference_confirm4_summary.json"
    _write_json(
        reference_summary,
        {
            "rows": [
                *[
                    {
                        "opponent_policy_id": opponent,
                        "wins": 5,
                        "games": 8,
                        "mean": 5 / 8,
                    }
                    for opponent in FIXED_THESIS_OPPONENTS
                ],
                {"opponent_policy_id": "seed_champion", "wins": 5, "games": 8, "mean": 5 / 8},
            ]
        },
    )

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "python/scripts/select_b1_candidate.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_json(
                output_json,
                {
                    "selected": {
                        "run_dir": run_dir.as_posix(),
                        "run_name": run_dir.name,
                        "snapshot_policy_id": "policy_000001",
                        "update_count": 5,
                        "eligible": True,
                        "selection_score": 0.62,
                        "selection_anchor_scores": {
                            "B1 NoLeague baseline": 0.62,
                            "B2 HeuristicPublic": 0.62,
                            "B3 HeuristicPublicAggro": 0.62,
                            "B4 HeuristicPublicControl": 0.62,
                        },
                    }
                },
            )
        elif "python/scripts/targeted_confirm_eval.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        *[
                            {
                                "opponent_policy_id": opponent,
                                "wins": 6,
                                "games": 8,
                                "mean": 6 / 8,
                            }
                            for opponent in FIXED_THESIS_OPPONENTS
                        ],
                        {"opponent_policy_id": "seed_champion", "wins": 6, "games": 8, "mean": 6 / 8},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_legacy_fail_keeps_multiobjective",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=4,
            reference_anchor_scores={"B4 HeuristicPublicControl": 0.70},
            multiobjective_reference_summary_jsons=(reference_summary,),
            learned_guard_opponents=("seed_champion",),
            min_learned_guard_reference_delta=0.0,
        ),
        runner=fake_runner,
    )

    segment = summary["segments"][0]
    assert summary["status"] == "stopped_guard_failed"
    assert segment["status"] == "rejected"
    assert segment["guard"]["passed"] is False
    assert segment["multiobjective_guard"]["passed"] is True
    assert segment["multiobjective_guard"]["groups"]["learned_opponents"]["reference_delta"] == pytest.approx(0.125)
    assert Path(tmp_path / segment["multiobjective_guard_json"]).is_file()


def test_guarded_league_bootstrap_advances_checkpoint_after_guard_passes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5), ("policy_000002", 10)])
        elif "python/scripts/select_b1_candidate.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_selection(output_json, run_dir=run_dir, policy_id="policy_000001", update=5)
            if "--publish-selected-alias" in command:
                _write_registry(
                    run_dir,
                    snapshots=[
                        ("policy_000001", 5),
                        ("policy_000002", 10),
                        ("main_league_selected", 5),
                    ],
                )
        elif "python/scripts/targeted_confirm_eval.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.55},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.59},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.56},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.57},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_pass",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            first_init_schedule_offset_updates=0,
            confirm_paired_seeds=4,
            publish_min_confirm_paired_seeds=4,
            reference_anchor_scores={"B3 HeuristicPublicAggro": 0.55},
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "completed"
    assert [segment["status"] for segment in summary["segments"]] == ["accepted", "accepted"]
    assert summary["segments"][0]["latest_policy_id"] == "policy_000002"
    assert summary["segments"][0]["selected"]["snapshot_policy_id"] == "policy_000001"
    assert summary["segments"][1]["source_checkpoint"].endswith(
        "runs/guarded_pass_seg01/training/checkpoints/checkpoint_5.pt"
    )
    train_commands = [command for command in calls if "python/scripts/train.py" in command]
    confirm_commands = [command for command in calls if "python/scripts/targeted_confirm_eval.py" in command]
    selector_commands = [command for command in calls if "python/scripts/select_b1_candidate.py" in command]
    assert len(train_commands) == 2
    assert len(confirm_commands) == 2
    assert len(selector_commands) == 6
    assert _option(train_commands[0], "--init-schedule-offset-updates") == "0"
    assert "--init-schedule-offset-updates" not in train_commands[1]
    assert _option(train_commands[1], "--init-from-checkpoint").endswith(
        "runs/guarded_pass_seg01/training/checkpoints/checkpoint_5.pt"
    )
    assert _option(confirm_commands[0], "--focal-policy-id") == "policy_000001"


def test_guarded_league_bootstrap_confirms_recent_candidates_before_selecting(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(
                run_dir,
                snapshots=[
                    ("policy_000001", 1),
                    ("policy_000002", 2),
                    ("policy_000003", 3),
                ],
            )
        elif "python/scripts/select_b1_candidate.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            if output_json.name.endswith("_candidate_preconfirm.json"):
                _write_json(output_json, {"ranked_candidates": []})
            else:
                _write_selection(output_json, run_dir=run_dir, policy_id="policy_000002", update=2, score=0.64)
        elif "python/scripts/targeted_confirm_eval.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.62},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.63},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.61},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_recent_candidates",
            segments=1,
            runtime=LeagueSegmentRuntime(segment_updates=3, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=64,
            confirm_recent_candidate_count=3,
            continue_unpublished_confirmed=True,
        ),
        runner=fake_runner,
    )

    segment = summary["segments"][0]
    confirm_commands = [command for command in calls if "python/scripts/targeted_confirm_eval.py" in command]
    assert [_option(command, "--focal-policy-id") for command in confirm_commands] == [
        "policy_000001",
        "policy_000002",
        "policy_000003",
    ]
    assert segment["confirm_focal_policy_ids"] == ["policy_000001", "policy_000002", "policy_000003"]
    assert segment["selected"]["snapshot_policy_id"] == "policy_000002"
    assert segment["selected_checkpoint"].endswith(
        "runs/guarded_recent_candidates_seg01/training/checkpoints/checkpoint_2.pt"
    )
    assert summary["status"] == "completed_unpublished_confirmation_insufficient"


def test_guarded_league_bootstrap_does_not_publish_when_confirmation_below_publish_floor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "python/scripts/select_b1_candidate.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_selection(output_json, run_dir=run_dir, policy_id="policy_000001", update=5)
        elif "python/scripts/targeted_confirm_eval.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.62},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.63},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.61},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_unpublished",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=64,
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "stopped_publish_confirmation_insufficient"
    assert summary["segments"][0]["status"] == "accepted_unpublished"
    assert summary["segments"][0]["publish_skipped"] == {
        "reason": "confirmation_seed_count_below_publish_minimum",
        "confirm_paired_seeds": 64,
        "publish_min_confirm_paired_seeds": 256,
        "continued_without_publish": False,
    }
    selector_commands = [command for command in calls if "python/scripts/select_b1_candidate.py" in command]
    assert len(selector_commands) == 2
    assert all("--publish-selected-alias" not in command for command in selector_commands)


def test_guarded_league_bootstrap_can_continue_unpublished_confirmed_segments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "python/scripts/select_b1_candidate.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_selection(output_json, run_dir=run_dir, policy_id="policy_000001", update=5)
        elif "python/scripts/targeted_confirm_eval.py" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.62},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.63},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.61},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_unpublished_continue",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=64,
            continue_unpublished_confirmed=True,
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "completed_unpublished_confirmation_insufficient"
    assert [segment["status"] for segment in summary["segments"]] == ["accepted_unpublished", "accepted_unpublished"]
    assert all(segment["publish_skipped"]["continued_without_publish"] is True for segment in summary["segments"])
    assert summary["segments"][1]["source_checkpoint"].endswith(
        "runs/guarded_unpublished_continue_seg01/training/checkpoints/checkpoint_5.pt"
    )
    train_commands = [command for command in calls if "python/scripts/train.py" in command]
    confirm_commands = [command for command in calls if "python/scripts/targeted_confirm_eval.py" in command]
    selector_commands = [command for command in calls if "python/scripts/select_b1_candidate.py" in command]
    assert len(train_commands) == 2
    assert len(confirm_commands) == 2
    assert len(selector_commands) == 4
    assert all("--publish-selected-alias" not in command for command in selector_commands)


def test_guarded_league_bootstrap_script_rejects_seed_opponents_with_pinned_import_defaults(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "python/scripts/guarded_league_bootstrap.py",
            "--init-from-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--seed-snapshot-run-dir",
            str(tmp_path / "seed_run"),
            "--required-anchor",
            "seed_example_policy",
            "--confirm-opponent",
            "seed_example_policy",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--use-stack-seed-snapshot-policy" in result.stderr
