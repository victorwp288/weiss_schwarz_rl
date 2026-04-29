"""Manifest payload helpers for training runs."""

from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from typing import Any

import torch

from weiss_rl.runtime import QueueRuntimeMode, build_runtime_config, resolve_actor_device_layout


def hardware_summary(
    learner_device: torch.device | str = "cpu",
    *,
    actor_device: torch.device | str = "cpu",
    actor_device_layout: Sequence[str] | None = None,
) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "learner_device": str(learner_device),
        "actor_device": str(actor_device),
    }
    if actor_device_layout:
        payload["actor_device_layout"] = ",".join(str(device_name) for device_name in actor_device_layout)
        payload["actor_device_unique_count"] = len(
            dict.fromkeys(str(device_name) for device_name in actor_device_layout)
        )
    return payload


def manifest_actor_device_layout(
    *,
    stack: Any,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
    learner_device: torch.device,
    resolved_topology: Any | None = None,
    rank_local_actor_devices: bool = False,
) -> tuple[str, ...] | None:
    if stack.config.system is None or stack.config.training is None:
        return None
    actor_device_request = str(getattr(stack.config.system, "actor_device", "")).strip().lower()
    if (
        rank_local_actor_devices
        and resolved_topology is not None
        and actor_device_request in {"auto", "cuda:auto", "cuda:all", "all"}
        and int(getattr(resolved_topology, "learner_gpu_count", 0)) > 1
    ):
        actor_count = int(getattr(resolved_topology, "actor_count", 1))
        learner_gpu_count = int(getattr(resolved_topology, "learner_gpu_count", 1))
        return tuple(f"cuda:{actor_index % learner_gpu_count}" for actor_index in range(actor_count))
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
        resolved_actor_count=None if resolved_topology is None else int(resolved_topology.actor_count),
        resolved_envs_per_actor=None if resolved_topology is None else int(resolved_topology.envs_per_actor),
        resolved_batch_unrolls_per_update=(
            None if resolved_topology is None else int(resolved_topology.batch_unrolls_per_update)
        ),
        resolved_queue_capacity_unrolls=(
            None if resolved_topology is None else int(resolved_topology.queue_capacity_unrolls)
        ),
    )
    return tuple(
        str(device_name)
        for device_name in resolve_actor_device_layout(
            stack,
            actor_count=int(runtime_config.actor_count),
            learner_device=learner_device,
            prefer_process_collectors=True,
            rank_local_cuda_auto=bool(rank_local_actor_devices),
        )
    )


def evaluation_pinning(stack: Any) -> dict[str, str | bool]:
    if stack.config.evaluation is None:
        return {}
    evaluation = stack.config.evaluation
    return {
        "eval_device": evaluation.eval_device,
        "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
        "eval_inference_mode": evaluation.eval_inference_mode,
        "seat_swap": evaluation.seat_swap,
        "legal_fingerprint_version": evaluation.legal_fingerprint_checks.version,
        "legal_fingerprint_mismatch_policy": evaluation.legal_fingerprint_checks.mismatch_policy,
    }


def training_controls_payload(
    training_config: Any,
    *,
    max_wall_clock_minutes: float | None = None,
    include_wall_clock_budget: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile_timers": bool(training_config.profile_timers),
        "torch_profiler": bool(training_config.torch_profiler),
        "structured_metrics_mode": str(training_config.structured_metrics_mode),
        "teacher_aux_mode": str(training_config.teacher_aux_mode),
        "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
        "heuristic_native_rollout_enabled": bool(training_config.heuristic_native_rollout_enabled),
        "heuristic_native_rollout_profile": str(training_config.heuristic_native_rollout_profile),
        "heuristic_native_rollout_profiles": list(training_config.heuristic_native_rollout_profiles),
        "heuristic_native_rollout_profile_mode": str(training_config.heuristic_native_rollout_profile_mode),
    }
    if include_wall_clock_budget:
        payload["max_wall_clock_minutes"] = None if max_wall_clock_minutes is None else float(max_wall_clock_minutes)
    return payload
