from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_selection import (
    DEFAULT_CONFIRM_OPPONENTS,
    DEFAULT_REQUIRED_ANCHORS,
    build_b1_candidate_selection,
)

DEFAULT_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml")
DEFAULT_ALIAS_POLICY_ID = "guided_bootstrap_floor_segmented_selected"
DEFAULT_RUN_LABEL = "b1_guided_floor_segmented"


@dataclass(frozen=True, slots=True)
class SegmentRuntime:
    num_envs: int = 288
    unroll_length: int = 64
    segment_updates: int = 25
    runtime_mode: str = "train_async_fast"
    simulator_profile: str = "fast"
    device: str = "cuda"
    checkpoint_interval_updates: int = 5
    collection_backend: str = "process"
    profile_timers: bool = True


@dataclass(frozen=True, slots=True)
class SegmentedBootstrapConfig:
    repo_root: Path
    initial_run_dir: Path
    initial_policy_id: str
    run_prefix: str
    stack_config: Path = DEFAULT_STACK_CONFIG
    seed_run_dir: Path | None = None
    alias_policy_id: str = DEFAULT_ALIAS_POLICY_ID
    segments: int = 4
    runtime: SegmentRuntime = SegmentRuntime()
    confirm_paired_seeds: int = 64
    bootstrap_samples: int = 2000
    required_anchors: tuple[str, ...] = DEFAULT_REQUIRED_ANCHORS
    confirm_opponents: tuple[str, ...] = DEFAULT_CONFIRM_OPPONENTS
    min_required_anchor_score: float = 0.5
    max_selected_drop: float = 0.02
    stop_on_latest_falloff: bool = False
    max_latest_drop: float = 0.05
    dry_run: bool = False


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def repo_relative(path: Path, *, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    try:
        return resolved.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return resolved.resolve()


def resolve_snapshot_checkpoint_path(*, run_dir: Path, policy_id: str) -> Path:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    snapshots = payload.get("snapshots") if isinstance(payload, Mapping) else None
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


def build_train_segment_command(
    *,
    config: SegmentedBootstrapConfig,
    segment_run_label: str,
    init_checkpoint_path: Path,
    seed_run_dir: Path,
) -> list[str]:
    runtime = config.runtime
    command = [
        sys.executable,
        "python/scripts/train.py",
        "--stack-config",
        repo_relative(config.stack_config, repo_root=config.repo_root).as_posix(),
        "--run-label",
        segment_run_label,
        "--num-envs",
        str(int(runtime.num_envs)),
        "--unroll-length",
        str(int(runtime.unroll_length)),
        "--max-updates",
        str(int(runtime.segment_updates)),
        "--runtime-mode",
        runtime.runtime_mode,
        "--profile",
        runtime.simulator_profile,
        "--device",
        runtime.device,
        "--override",
        f"system.collection_backend={runtime.collection_backend}",
        "--checkpoint-interval-updates",
        str(int(runtime.checkpoint_interval_updates)),
        "--seed-snapshot-run-dir",
        repo_relative(seed_run_dir, repo_root=config.repo_root).as_posix(),
        "--init-from-checkpoint",
        repo_relative(init_checkpoint_path, repo_root=config.repo_root).as_posix(),
    ]
    if bool(runtime.profile_timers):
        command.extend(["--override", "training.profile_timers=true"])
    return command


def build_selector_command(
    *,
    config: SegmentedBootstrapConfig,
    run_dir: Path,
    output_json: Path,
    publish_alias: bool,
) -> list[str]:
    command = [
        sys.executable,
        "python/scripts/select_b1_candidate.py",
        "--run-dir",
        repo_relative(run_dir, repo_root=config.repo_root).as_posix(),
        "--stack-config",
        repo_relative(config.stack_config, repo_root=config.repo_root).as_posix(),
        "--min-required-anchor-score",
        str(float(config.min_required_anchor_score)),
        "--confirm-paired-seeds",
        str(int(config.confirm_paired_seeds)),
        "--output-json",
        repo_relative(output_json, repo_root=config.repo_root).as_posix(),
    ]
    for anchor in config.required_anchors:
        command.extend(["--required-anchor", anchor])
    for opponent in config.confirm_opponents:
        command.extend(["--confirm-opponent", opponent])
    if publish_alias:
        command.extend(["--publish-selected-alias", "--selected-alias-policy-id", config.alias_policy_id])
    return command


def build_targeted_confirm_command(
    *,
    config: SegmentedBootstrapConfig,
    run_dir: Path,
    focal_policy_id: str,
    output_subdir: str,
) -> list[str]:
    command = [
        sys.executable,
        "python/scripts/targeted_confirm_eval.py",
        "--stack-config",
        repo_relative(config.stack_config, repo_root=config.repo_root).as_posix(),
        "--run-dir",
        repo_relative(run_dir, repo_root=config.repo_root).as_posix(),
        "--snapshot-registry-json",
        repo_relative(run_dir / "training" / "snapshots" / "registry.json", repo_root=config.repo_root).as_posix(),
        "--b1-baseline-run-dir",
        repo_relative(run_dir, repo_root=config.repo_root).as_posix(),
        "--focal-policy-id",
        focal_policy_id,
        "--paired-seeds",
        str(int(config.confirm_paired_seeds)),
        "--workers",
        "1",
        "--bootstrap-samples",
        str(int(config.bootstrap_samples)),
        "--output-subdir",
        output_subdir,
    ]
    for opponent in config.confirm_opponents:
        command.extend(["--opponent", opponent])
    return command


def _command_record(command: Sequence[str]) -> dict[str, Any]:
    return {"argv": list(command), "display": " ".join(f'"{part}"' if " " in part else part for part in command)}


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    runner: CommandRunner,
    env: Mapping[str, str] | None = None,
) -> None:
    completed = runner(list(command), cwd=cwd, check=False, env=None if env is None else dict(env))
    if int(completed.returncode) != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def _fixed_hash_seed_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    return env


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _selected_candidate(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError(f"candidate selector did not produce a selected candidate: {path}")
    return selected


def _selection_score(candidate: Mapping[str, Any]) -> float:
    raw_score = candidate.get("selection_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return 0.0
    return float(raw_score)


def _latest_minus_best(path: Path) -> float | None:
    payload = _read_json_object(path)
    run_summaries = payload.get("run_summaries")
    if not isinstance(run_summaries, list) or not run_summaries:
        return None
    raw_value = run_summaries[0].get("latest_minus_best") if isinstance(run_summaries[0], Mapping) else None
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    return float(raw_value)


def _initial_selection_score(config: SegmentedBootstrapConfig) -> float | None:
    summary = build_b1_candidate_selection(
        [config.initial_run_dir],
        stack_config=config.stack_config,
        required_anchors=config.required_anchors,
        confirm_opponents=config.confirm_opponents,
        min_required_anchor_score=config.min_required_anchor_score,
        confirm_paired_seeds=config.confirm_paired_seeds,
    )
    selected = summary.get("selected")
    if not isinstance(selected, Mapping):
        return None
    return _selection_score(selected)


def run_segmented_b1_guided_bootstrap(
    config: SegmentedBootstrapConfig,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    repo_root = config.repo_root.resolve()
    current_run_dir = config.initial_run_dir.resolve()
    current_seed_run_dir = (config.seed_run_dir or config.initial_run_dir).resolve()
    current_policy_id = str(config.initial_policy_id).strip()
    diagnostics_dir = repo_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / f"{config.run_prefix}_segmented_bootstrap_summary.json"
    previous_score = _initial_selection_score(config)
    summary: dict[str, Any] = {
        "kind": "segmented_b1_guided_bootstrap_v1",
        "created_unix": time.time(),
        "repo_root": repo_root.as_posix(),
        "stack_config": repo_relative(config.stack_config, repo_root=repo_root).as_posix(),
        "initial_run_dir": repo_relative(config.initial_run_dir, repo_root=repo_root).as_posix(),
        "initial_policy_id": config.initial_policy_id,
        "alias_policy_id": config.alias_policy_id,
        "segments_requested": int(config.segments),
        "confirm_paired_seeds": int(config.confirm_paired_seeds),
        "min_required_anchor_score": float(config.min_required_anchor_score),
        "max_selected_drop": float(config.max_selected_drop),
        "max_latest_drop": float(config.max_latest_drop),
        "stop_on_latest_falloff": bool(config.stop_on_latest_falloff),
        "initial_selection_score": previous_score,
        "segments": [],
        "status": "planned" if config.dry_run else "running",
    }

    for segment_index in range(1, int(config.segments) + 1):
        segment_run_label = f"{config.run_prefix}_seg{segment_index:02d}"
        segment_run_dir = repo_root / "runs" / segment_run_label
        init_checkpoint_path = resolve_snapshot_checkpoint_path(
            run_dir=current_run_dir,
            policy_id=current_policy_id,
        )
        train_command = build_train_segment_command(
            config=config,
            segment_run_label=segment_run_label,
            init_checkpoint_path=init_checkpoint_path,
            seed_run_dir=current_seed_run_dir,
        )
        preselect_json = diagnostics_dir / f"{segment_run_label}_candidate_preconfirm.json"
        final_json = diagnostics_dir / f"{segment_run_label}_candidate_selection.json"
        segment_record: dict[str, Any] = {
            "segment": segment_index,
            "run_label": segment_run_label,
            "run_dir": repo_relative(segment_run_dir, repo_root=repo_root).as_posix(),
            "source_run_dir": repo_relative(current_run_dir, repo_root=repo_root).as_posix(),
            "source_policy_id": current_policy_id,
            "source_checkpoint": repo_relative(init_checkpoint_path, repo_root=repo_root).as_posix(),
            "seed_run_dir": repo_relative(current_seed_run_dir, repo_root=repo_root).as_posix(),
            "train_command": _command_record(train_command),
            "preselect_json": repo_relative(preselect_json, repo_root=repo_root).as_posix(),
            "final_selection_json": repo_relative(final_json, repo_root=repo_root).as_posix(),
        }
        summary["segments"].append(segment_record)

        if config.dry_run:
            segment_record["status"] = "planned"
            segment_record["preselect_command"] = _command_record(
                build_selector_command(
                    config=config, run_dir=segment_run_dir, output_json=preselect_json, publish_alias=False
                )
            )
            segment_record["targeted_confirm_command_template"] = _command_record(
                build_targeted_confirm_command(
                    config=config,
                    run_dir=segment_run_dir,
                    focal_policy_id="<selected-policy-id>",
                    output_subdir=f"segmented_confirm{int(config.confirm_paired_seeds)}_<selected-policy-id>",
                )
            )
            segment_record["final_selector_command"] = _command_record(
                build_selector_command(
                    config=config, run_dir=segment_run_dir, output_json=final_json, publish_alias=True
                )
            )
            break

        _run_command(train_command, cwd=repo_root, runner=runner, env=_fixed_hash_seed_env())
        preselect_command = build_selector_command(
            config=config,
            run_dir=segment_run_dir,
            output_json=preselect_json,
            publish_alias=False,
        )
        segment_record["preselect_command"] = _command_record(preselect_command)
        _run_command(preselect_command, cwd=repo_root, runner=runner)

        preselected = _selected_candidate(preselect_json)
        focal_policy_id = str(preselected.get("snapshot_policy_id", "")).strip()
        if not focal_policy_id:
            raise RuntimeError(f"selected candidate has no snapshot_policy_id: {preselect_json}")
        confirm_subdir = f"segmented_confirm{int(config.confirm_paired_seeds)}_{focal_policy_id}"
        confirm_command = build_targeted_confirm_command(
            config=config,
            run_dir=segment_run_dir,
            focal_policy_id=focal_policy_id,
            output_subdir=confirm_subdir,
        )
        segment_record["targeted_confirm_command"] = _command_record(confirm_command)
        confirm_env = os.environ.copy()
        confirm_env["PYTHONHASHSEED"] = "0"
        _run_command(confirm_command, cwd=repo_root, runner=runner, env=confirm_env)

        final_selector_command = build_selector_command(
            config=config,
            run_dir=segment_run_dir,
            output_json=final_json,
            publish_alias=True,
        )
        segment_record["final_selector_command"] = _command_record(final_selector_command)
        _run_command(final_selector_command, cwd=repo_root, runner=runner)
        final_selected = _selected_candidate(final_json)
        latest_drop = _latest_minus_best(final_json)
        score = _selection_score(final_selected)
        selected_drop = None if previous_score is None else score - previous_score
        segment_record["status"] = "completed"
        segment_record["selected"] = final_selected
        segment_record["selection_score"] = score
        segment_record["selected_minus_previous"] = selected_drop
        segment_record["latest_minus_best"] = latest_drop

        if not bool(final_selected.get("eligible")):
            summary["status"] = "stopped_ineligible"
            summary["stop_reason"] = "selected candidate did not meet required anchor threshold"
            break
        if selected_drop is not None and selected_drop < -float(config.max_selected_drop):
            summary["status"] = "stopped_selected_drop"
            summary["stop_reason"] = (
                f"selected score dropped by {selected_drop:.4f}, below -{float(config.max_selected_drop):.4f}"
            )
            break
        if (
            bool(config.stop_on_latest_falloff)
            and latest_drop is not None
            and latest_drop < -float(config.max_latest_drop)
        ):
            summary["status"] = "stopped_latest_falloff"
            summary["stop_reason"] = (
                f"latest fell behind best by {latest_drop:.4f}, below -{float(config.max_latest_drop):.4f}"
            )
            break

        previous_score = score
        current_run_dir = segment_run_dir
        current_seed_run_dir = segment_run_dir
        current_policy_id = config.alias_policy_id
        summary["status"] = "completed"

    summary["summary_path"] = summary_path.as_posix()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
