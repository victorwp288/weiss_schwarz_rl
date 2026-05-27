"""Helpers for updating training run report payloads."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any


def training_controls_payload(training_config: Any | None) -> dict[str, str | bool | float] | None:
    """Return the run-report training controls payload for a configured training run."""

    if training_config is None:
        return None
    return {
        "profile_timers": bool(training_config.profile_timers),
        "torch_profiler": bool(training_config.torch_profiler),
        "structured_metrics_mode": str(training_config.structured_metrics_mode),
        "teacher_aux_mode": str(training_config.teacher_aux_mode),
        "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
        "actor_policy_backend": str(training_config.actor_policy_backend),
        "actor_heuristic_fraction": float(training_config.actor_heuristic_fraction),
        "actor_heuristic_final_fraction": float(training_config.actor_heuristic_final_fraction),
        "actor_sampling_temperature": float(getattr(training_config, "actor_sampling_temperature", 1.0)),
        "train_on_heuristic_actor_rows": bool(training_config.train_on_heuristic_actor_rows),
    }


def profiling_enabled_message(training_config: Any) -> str | None:
    """Return the user-facing structured profiling message when profiling is active."""

    profile_timers = bool(training_config.profile_timers)
    torch_profiler = bool(training_config.torch_profiler)
    if not profile_timers and not torch_profiler:
        return None
    return (
        "Structured profiling enabled: "
        f"profile_timers={profile_timers} "
        f"torch_profiler={torch_profiler} "
        f"structured_metrics_mode={training_config.structured_metrics_mode} "
        f"teacher_aux_mode={training_config.teacher_aux_mode} "
        f"fixed_opponent_backend={training_config.fixed_opponent_backend}"
    )


def augment_run_summary_payload(
    payload: MutableMapping[str, Any],
    *,
    public_demo_enabled: bool,
    runtime_mode: str,
    policy_set_selection_details: Mapping[str, Any],
    training_config: Any | None,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None,
    init_from_checkpoint_path: Path | None,
    resume_run_dir: Path | None,
    resume_checkpoint_path: Path | None,
) -> MutableMapping[str, Any]:
    """Apply train-entrypoint fields to the run summary payload."""

    payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(runtime_mode)
    payload["policy_set_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    training_controls = training_controls_payload(training_config)
    if training_controls is not None:
        payload["training_controls"] = training_controls
    if b1_baseline_run_dir is not None:
        payload["b1_baseline_run_dir"] = b1_baseline_run_dir.resolve().as_posix()
    if seed_snapshot_run_dir is not None:
        payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.resolve().as_posix()
    if init_from_checkpoint_path is not None:
        payload["init_from_checkpoint_path"] = init_from_checkpoint_path.resolve().as_posix()
    if resume_checkpoint_path is not None:
        payload["resume"] = {
            "enabled": True,
            "resume_run_dir": None if resume_run_dir is None else resume_run_dir.as_posix(),
            "resume_checkpoint_path": resume_checkpoint_path.as_posix(),
        }
    return payload


def augment_determinism_payload(
    payload: MutableMapping[str, Any],
    *,
    public_demo_enabled: bool,
    runtime_mode: str,
    policy_set_selection_details: Mapping[str, Any],
    training_config: Any | None,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None,
    init_from_checkpoint_path: Path | None,
    resume_checkpoint_path: Path | None,
) -> MutableMapping[str, Any]:
    """Apply train-entrypoint fields to the determinism report payload."""

    payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(runtime_mode)
    payload["policy_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    training_controls = training_controls_payload(training_config)
    if training_controls is not None:
        payload["training_controls"] = training_controls
    if b1_baseline_run_dir is not None:
        payload["b1_baseline_run_dir"] = b1_baseline_run_dir.resolve().as_posix()
    if seed_snapshot_run_dir is not None:
        payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.resolve().as_posix()
    if init_from_checkpoint_path is not None:
        payload["init_from_checkpoint_path"] = init_from_checkpoint_path.resolve().as_posix()
    if resume_checkpoint_path is not None:
        payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
    return payload


def augment_environment_payload(
    payload: MutableMapping[str, Any],
    *,
    root: Path,
    argv: Sequence[str],
    hardware: Mapping[str, Any],
    init_from_checkpoint_path: Path | None,
    resume_checkpoint_path: Path | None,
) -> MutableMapping[str, Any]:
    """Apply train-entrypoint fields to the environment manifest payload."""

    payload["cwd"] = root.as_posix()
    payload["argv"] = list(argv)
    payload["hardware"] = hardware
    if init_from_checkpoint_path is not None:
        payload["init_from_checkpoint_path"] = init_from_checkpoint_path.as_posix()
    if resume_checkpoint_path is not None:
        payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
    return payload
