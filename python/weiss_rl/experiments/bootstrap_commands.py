from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def repo_relative(path: Path, *, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    try:
        return resolved.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return resolved.resolve()


def command_path(path: Path, *, repo_root: Path | None) -> str:
    if repo_root is None:
        return path.as_posix()
    return repo_relative(path, repo_root=repo_root).as_posix()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_snapshot_checkpoint_path(*, run_dir: Path, policy_id: str) -> Path:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")
    payload = read_json_object(registry_path)
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError(f"snapshot registry must contain a snapshots list: {registry_path}")
    normalized = str(policy_id).strip()
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping) or str(snapshot.get("policy_id", "")).strip() != normalized:
            continue
        update = snapshot.get("update", snapshot.get("update_count"))
        if isinstance(update, bool) or not isinstance(update, int):
            raise ValueError(f"snapshot {normalized!r} is missing an integer update in {registry_path}")
        checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{int(update)}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint for snapshot {normalized!r} was not found: {checkpoint_path}")
        return checkpoint_path
    raise ValueError(f"snapshot policy id not found in {registry_path}: {normalized}")


def command_record(command: Sequence[str]) -> dict[str, Any]:
    return {"argv": list(command), "display": " ".join(f'"{part}"' if " " in part else part for part in command)}


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    runner: CommandRunner,
    env: Mapping[str, str] | None = None,
) -> None:
    completed = runner(list(command), cwd=cwd, check=False, env=None if env is None else dict(env))
    if int(completed.returncode) != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def fixed_hash_seed_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    return env


def build_train_entrypoint_command(
    *,
    repo_root: Path,
    stack_config: Path,
    run_label: str,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
    runtime_mode: str,
    simulator_profile: str,
    device: str,
    checkpoint_interval_updates: int,
    seed_snapshot_run_dir: Path,
    init_checkpoint_path: Path,
    collection_backend: str,
    b1_baseline_run_dir: Path | None = None,
    init_schedule_offset_updates: int | None = None,
    profile_timers: bool = True,
    overrides: Sequence[str] = (),
) -> list[str]:
    extra_args: list[str] = []
    extra_args.extend(["--seed-snapshot-run-dir", repo_relative(seed_snapshot_run_dir, repo_root=repo_root).as_posix()])
    extra_args.extend(["--init-from-checkpoint", repo_relative(init_checkpoint_path, repo_root=repo_root).as_posix()])
    if b1_baseline_run_dir is not None:
        extra_args.extend(["--b1-baseline-run-dir", repo_relative(b1_baseline_run_dir, repo_root=repo_root).as_posix()])
    if init_schedule_offset_updates is not None:
        extra_args.extend(["--init-schedule-offset-updates", str(int(init_schedule_offset_updates))])
    if profile_timers:
        extra_args.extend(["--override", "training.profile_timers=true"])
    for override in overrides:
        extra_args.extend(["--override", str(override)])
    return build_training_entrypoint_command(
        repo_root=repo_root,
        stack_config=stack_config,
        run_label=run_label,
        seed=None,
        num_envs=num_envs,
        unroll_length=unroll_length,
        max_updates=max_updates,
        runtime_mode=runtime_mode,
        simulator_profile=simulator_profile,
        device=device,
        checkpoint_interval_updates=checkpoint_interval_updates,
        collection_backend=collection_backend,
        extra_args=extra_args,
    )


def build_training_entrypoint_command(
    *,
    repo_root: Path,
    stack_config: Path,
    run_label: str,
    seed: int | None = None,
    num_envs: int | None = None,
    unroll_length: int | None = None,
    max_updates: int | None = None,
    runtime_mode: str | None = None,
    simulator_profile: str | None = None,
    device: str | None = None,
    checkpoint_interval_updates: int | None = None,
    collection_backend: str | None = None,
    b1_baseline_run_dir: Path | None = None,
    overrides: Sequence[str] = (),
    extra_args: Sequence[str] = (),
    python_executable: str | None = None,
) -> list[str]:
    command = [
        python_executable or sys.executable,
        "-m",
        "weiss_rl.training.train_entrypoint",
        "--stack-config",
        repo_relative(stack_config, repo_root=repo_root).as_posix(),
        "--run-label",
        run_label,
    ]
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    if num_envs is not None:
        command.extend(["--num-envs", str(int(num_envs))])
    if unroll_length is not None:
        command.extend(["--unroll-length", str(int(unroll_length))])
    if max_updates is not None:
        command.extend(["--max-updates", str(int(max_updates))])
    if runtime_mode is not None:
        command.extend(["--runtime-mode", str(runtime_mode)])
    if simulator_profile is not None:
        command.extend(["--profile", str(simulator_profile)])
    if device is not None:
        command.extend(["--device", str(device)])
    if checkpoint_interval_updates is not None:
        command.extend(["--checkpoint-interval-updates", str(int(checkpoint_interval_updates))])
    if collection_backend:
        command.extend(["--override", f"system.collection_backend={collection_backend}"])
    if b1_baseline_run_dir is not None:
        command.extend(["--b1-baseline-run-dir", repo_relative(b1_baseline_run_dir, repo_root=repo_root).as_posix()])
    for override in overrides:
        command.extend(["--override", str(override)])
    command.extend(str(arg) for arg in extra_args)
    return command


def build_b2_disagreement_audit_entrypoint_command(
    *,
    repo_root: Path,
    stack_config: Path,
    run_dir: Path,
    output_run_dir: Path,
    episodes_jsonl: Path,
    policy_id: str,
    python_executable: str | None = None,
) -> list[str]:
    return [
        python_executable or sys.executable,
        "-m",
        "weiss_rl.diagnostics.b2_disagreement_audit",
        "--stack-config",
        repo_relative(stack_config, repo_root=repo_root).as_posix(),
        "--run-dir",
        repo_relative(run_dir, repo_root=repo_root).as_posix(),
        "--output-run-dir",
        repo_relative(output_run_dir, repo_root=repo_root).as_posix(),
        "--episodes-jsonl",
        repo_relative(episodes_jsonl, repo_root=repo_root).as_posix(),
        "--policy-id",
        str(policy_id),
    ]


def build_targeted_confirm_entrypoint_command(
    *,
    repo_root: Path | None,
    stack_config: Path,
    run_dir: Path,
    b1_baseline_run_dir: Path,
    focal_policy_id: str,
    paired_seeds: int,
    bootstrap_samples: int | None,
    output_subdir: str,
    opponents: Sequence[str],
    python_command: Sequence[str] | None = None,
) -> list[str]:
    command = [
        *(list(python_command) if python_command is not None else [sys.executable]),
        "-m",
        "weiss_rl.eval.targeted_confirm.entrypoint",
        "--stack-config",
        command_path(stack_config, repo_root=repo_root),
        "--run-dir",
        command_path(run_dir, repo_root=repo_root),
        "--snapshot-registry-json",
        command_path(run_dir / "training" / "snapshots" / "registry.json", repo_root=repo_root),
        "--b1-baseline-run-dir",
        command_path(b1_baseline_run_dir, repo_root=repo_root),
        "--focal-policy-id",
        str(focal_policy_id),
        "--paired-seeds",
        str(int(paired_seeds)),
        "--workers",
        "1",
    ]
    if bootstrap_samples is not None:
        command.extend(["--bootstrap-samples", str(int(bootstrap_samples))])
    command.extend(["--output-subdir", str(output_subdir)])
    for opponent in opponents:
        command.extend(["--opponent", str(opponent)])
    return command
