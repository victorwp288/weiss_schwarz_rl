from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from weiss_rl.experiments.bootstrap_commands import (
    build_b2_disagreement_audit_entrypoint_command,
    build_candidate_selector_command,
    build_targeted_confirm_entrypoint_command,
    build_train_entrypoint_command,
    build_training_entrypoint_command,
    fixed_hash_seed_env,
    repo_relative,
    run_command,
)


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_bootstrap_train_command_uses_package_entrypoint_and_repo_relative_paths(tmp_path: Path) -> None:
    command = build_train_entrypoint_command(
        repo_root=tmp_path,
        stack_config=tmp_path / "configs" / "stack.yaml",
        run_label="seg01",
        num_envs=2,
        unroll_length=4,
        max_updates=5,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cpu",
        checkpoint_interval_updates=1,
        seed_snapshot_run_dir=tmp_path / "runs" / "seed",
        b1_baseline_run_dir=tmp_path / "runs" / "b1",
        init_checkpoint_path=tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_5.pt",
        collection_backend="process",
        init_schedule_offset_updates=0,
        overrides=("training.profile_timers=false",),
    )

    assert command[1:3] == ["-m", "weiss_rl.training.train_entrypoint"]
    assert _option(command, "--stack-config") == "configs/stack.yaml"
    assert _option(command, "--run-label") == "seg01"
    assert _option(command, "--seed-snapshot-run-dir") == "runs/seed"
    assert _option(command, "--b1-baseline-run-dir") == "runs/b1"
    assert _option(command, "--init-from-checkpoint") == "runs/seed/training/checkpoints/checkpoint_5.pt"
    assert _option(command, "--init-schedule-offset-updates") == "0"
    assert "system.collection_backend=process" in command
    assert "training.profile_timers=true" in command
    assert "training.profile_timers=false" in command


def test_bootstrap_selector_and_confirm_commands_share_package_entrypoint_surface(tmp_path: Path) -> None:
    selector = build_candidate_selector_command(
        repo_root=tmp_path,
        stack_config=tmp_path / "configs" / "stack.yaml",
        run_dir=tmp_path / "runs" / "seg01",
        output_json=tmp_path / "diagnostics" / "selection.json",
        min_required_anchor_score=0.5,
        confirm_paired_seeds=64,
        required_anchors=("B2 HeuristicPublic",),
        confirm_opponents=("B1 NoLeague baseline", "B2 HeuristicPublic"),
        publish_alias=True,
        selected_alias_policy_id="main_league_selected",
    )
    confirm = build_targeted_confirm_entrypoint_command(
        repo_root=tmp_path,
        stack_config=tmp_path / "configs" / "stack.yaml",
        run_dir=tmp_path / "runs" / "seg01",
        b1_baseline_run_dir=tmp_path / "runs" / "locked_b1",
        focal_policy_id="policy_000005",
        paired_seeds=64,
        bootstrap_samples=2000,
        output_subdir="confirm_policy_000005",
        opponents=("B1 NoLeague baseline", "B2 HeuristicPublic"),
    )

    assert selector[1:3] == ["-m", "weiss_rl.experiments.select_b1_candidate_entrypoint"]
    assert _option(selector, "--output-json") == "diagnostics/selection.json"
    assert _option(selector, "--selected-alias-policy-id") == "main_league_selected"
    assert selector.count("--confirm-opponent") == 2

    assert confirm[1:3] == ["-m", "weiss_rl.eval.targeted_confirm_entrypoint"]
    assert _option(confirm, "--snapshot-registry-json") == "runs/seg01/training/snapshots/registry.json"
    assert _option(confirm, "--b1-baseline-run-dir") == "runs/locked_b1"
    assert _option(confirm, "--output-subdir") == "confirm_policy_000005"
    assert confirm.count("--opponent") == 2


def test_targeted_confirm_builder_can_preserve_uv_operator_command_prefix(tmp_path: Path) -> None:
    command = build_targeted_confirm_entrypoint_command(
        repo_root=None,
        stack_config=tmp_path / "configs" / "stack.yaml",
        run_dir=tmp_path / "runs" / "b1_candidate",
        b1_baseline_run_dir=tmp_path / "runs" / "b1_candidate",
        focal_policy_id="policy_000002",
        paired_seeds=64,
        bootstrap_samples=2000,
        output_subdir="b1_candidate_confirm64_policy_000002",
        opponents=("B2 HeuristicPublic",),
        python_command=("uv", "run", "--extra", "dev", "--extra", "sim", "python"),
    )

    assert command[:7] == ["uv", "run", "--extra", "dev", "--extra", "sim", "python"]
    assert command[7:9] == ["-m", "weiss_rl.eval.targeted_confirm_entrypoint"]
    assert _option(command, "--run-dir") == (tmp_path / "runs" / "b1_candidate").as_posix()
    assert _option(command, "--snapshot-registry-json").endswith("training/snapshots/registry.json")
    assert _option(command, "--output-subdir") == "b1_candidate_confirm64_policy_000002"


def test_general_train_and_b2_audit_builders_are_package_module_commands(tmp_path: Path) -> None:
    train = build_training_entrypoint_command(
        repo_root=tmp_path,
        stack_config=tmp_path / "configs" / "stack.yaml",
        run_label="campaign_b1_seed7_u120",
        seed=7,
        num_envs=96,
        unroll_length=64,
        max_updates=120,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        b1_baseline_run_dir=tmp_path / "runs" / "b1",
        overrides=('experiment.role="baseline_noleague"', "league.enabled=false"),
        python_executable="python.exe",
    )
    audit = build_b2_disagreement_audit_entrypoint_command(
        repo_root=tmp_path,
        stack_config=tmp_path / "configs" / "stack.yaml",
        run_dir=tmp_path / "runs" / "campaign_canary_seed7_u120",
        output_run_dir=tmp_path / "runs" / "campaign_canary_seed7_u120_audit_b2_u120",
        episodes_jsonl=tmp_path / "runs" / "campaign_canary_seed7_u120" / "eval" / "dev_eval" / "episodes.jsonl",
        policy_id="train_u120_pX",
        python_executable="python.exe",
    )

    assert train[:3] == ["python.exe", "-m", "weiss_rl.training.train_entrypoint"]
    assert _option(train, "--stack-config") == "configs/stack.yaml"
    assert _option(train, "--seed") == "7"
    assert _option(train, "--b1-baseline-run-dir") == "runs/b1"
    assert 'experiment.role="baseline_noleague"' in train
    assert "league.enabled=false" in train

    assert audit[:3] == ["python.exe", "-m", "weiss_rl.diagnostics.b2_disagreement_audit"]
    assert _option(audit, "--run-dir") == "runs/campaign_canary_seed7_u120"
    assert _option(audit, "--output-run-dir") == "runs/campaign_canary_seed7_u120_audit_b2_u120"
    assert _option(audit, "--policy-id") == "train_u120_pX"


def test_bootstrap_command_runner_and_path_helpers_preserve_controller_contracts(tmp_path: Path) -> None:
    observed: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        observed.append((command, kwargs.get("env")))
        return subprocess.CompletedProcess(command, 0)

    env = fixed_hash_seed_env()
    run_command(["python", "-m", "demo"], cwd=tmp_path, runner=fake_runner, env=env)

    assert observed == [(["python", "-m", "demo"], env)]
    assert repo_relative(tmp_path / "runs" / "demo", repo_root=tmp_path) == Path("runs/demo")
