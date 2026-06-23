"""QueueRuntime constructor-time configuration and backend setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.league.registry import REGISTRY_FILENAME
from weiss_rl.runtime.components.action_catalog_setup import resolve_runtime_action_catalog_setup
from weiss_rl.runtime.components.collection_backend import select_runtime_collection_backend
from weiss_rl.runtime.components.config import QueueRuntimeConfig
from weiss_rl.runtime.components.devices import (
    configured_learner_device_name,
    is_cuda_auto_request,
    resolve_actor_device_layout,
)
from weiss_rl.runtime.components.heuristic_policy_setup import build_runtime_heuristic_policy_setup
from weiss_rl.runtime.components.opponent_startup import initialize_runtime_opponent_state
from weiss_rl.runtime.components.shared_memory.config import DEFAULT_ACTION_META_WIDTH
from weiss_rl.runtime.components.teacher_settings import resolve_runtime_teacher_settings
from weiss_rl.runtime.components.training_settings import resolve_runtime_training_settings


@dataclass(frozen=True)
class RuntimeConfigurationContext:
    model_kind: str
    training_config: Any
    structured_warmstart_cfg: Any
    actor_torch_threads: int | None


def initialize_runtime_configuration(
    runtime: Any,
    *,
    stack: Any,
    config: QueueRuntimeConfig,
    observation_dim: int,
    action_dim: int,
    observation_spec: dict[str, Any] | None,
    spec_bundle: dict[str, Any] | None,
    run_dir: Path | None,
    learner_device: torch.device | str | None,
) -> RuntimeConfigurationContext:
    _validate_runtime_config(config)

    runtime.stack = stack
    runtime.config = config
    runtime.observation_dim = int(observation_dim)
    runtime.action_dim = int(action_dim)
    runtime._observation_spec = None if observation_spec is None else dict(observation_spec)
    runtime._spec_bundle = None if spec_bundle is None else dict(spec_bundle)
    action_meta_spec = {} if runtime._spec_bundle is None else dict(runtime._spec_bundle.get("action_meta_v1", {}))
    runtime._action_meta_width = int(action_meta_spec.get("width", DEFAULT_ACTION_META_WIDTH))

    system_config = stack.config.system
    training_config = stack.config.training
    experiment_config = stack.config.experiment

    runtime._learner_device = torch.device(configured_learner_device_name(stack, learner_device=learner_device))
    runtime._requested_actor_device = (
        "cpu" if system_config is None else str(getattr(system_config, "actor_device", "cpu")).strip()
    )
    runtime._process_actor_device_names = resolve_actor_device_layout(
        stack,
        actor_count=int(config.actor_count),
        learner_device=runtime._learner_device,
        prefer_process_collectors=True,
    )
    runtime._device = resolve_runtime_actor_device(stack, learner_device=runtime._learner_device)
    runtime._run_dir = None if run_dir is None else Path(run_dir)
    runtime._artifact_layout = None if runtime._run_dir is None else ArtifactLayout.from_run_dir(runtime._run_dir)
    runtime._experiment_role = "" if experiment_config is None else str(experiment_config.role).strip()
    runtime._actor_amp_enabled = bool(
        training_config is not None and bool(training_config.mixed_precision) and runtime._device.type == "cuda"
    )
    runtime._compile_actor_inference = bool(
        training_config is not None
        and bool(getattr(training_config, "compile_actor_inference", False))
        and runtime._device.type == "cpu"
    )
    runtime._league_config = stack.config.league
    runtime._league_enabled = bool(
        runtime._artifact_layout is not None
        and runtime._league_config is not None
        and runtime._league_config.enabled
        and runtime._experiment_role != "baseline_noleague"
    )
    runtime._registry_path = (
        None
        if runtime._artifact_layout is None
        else runtime._artifact_layout.training_snapshots_dir / REGISTRY_FILENAME
    )
    runtime._opponent_models = {}
    runtime._opponent_model_locks = {}

    structured_warmstart_cfg = (
        None if training_config is None else getattr(training_config, "structured_warmstart", None)
    )
    runtime_teacher = resolve_runtime_teacher_settings(training_config=training_config)
    _assign_runtime_teacher_settings(runtime, runtime_teacher)

    runtime_training = resolve_runtime_training_settings(
        training_config=training_config,
        actor_count=int(runtime.config.actor_count),
    )
    _assign_runtime_training_settings(runtime, runtime_training)

    runtime_action_catalog = resolve_runtime_action_catalog_setup(spec_bundle=runtime._spec_bundle)
    runtime._action_catalog = runtime_action_catalog.action_catalog
    runtime._action_family_index = runtime_action_catalog.action_family_index
    runtime._action_attack_type_index = runtime_action_catalog.action_attack_type_index
    runtime._last_action_arg0_obs_index = runtime_action_catalog.last_action_arg0_obs_index

    runtime_heuristic_policies = build_runtime_heuristic_policy_setup(
        spec_bundle=runtime._spec_bundle,
        action_catalog=runtime._action_catalog,
        teacher_settings=runtime_teacher,
        actor_policy_backend=runtime._actor_policy_backend,
        league_config=runtime._league_config,
        diverse_opponent_actor_count=runtime._diverse_opponent_actor_count,
        actor_count=int(runtime.config.actor_count),
    )
    runtime._teacher_policy = runtime_heuristic_policies.teacher_policy
    runtime._teacher_policy_by_profile = runtime_heuristic_policies.teacher_policy_by_profile
    runtime._teacher_action_catalog = runtime_heuristic_policies.teacher_action_catalog
    runtime._teacher_family_index = runtime_heuristic_policies.teacher_family_index
    runtime._teacher_attack_type_index = runtime_heuristic_policies.teacher_attack_type_index
    runtime._opponent_heuristic_policies = runtime_heuristic_policies.opponent_heuristic_policies

    initialize_runtime_opponent_state(runtime, league_config=runtime._league_config)
    _assign_reserved_fixed_lane_settings(runtime, config=config)
    model_kind = "" if stack.config.model is None else str(stack.config.model.encoder_kind).strip().lower()
    runtime._structured_fixed_opponents_expected = _structured_fixed_opponents_expected(
        model_kind=model_kind,
        structured_warmstart_cfg=structured_warmstart_cfg,
        heuristic_reserved_envs=runtime._heuristic_public_reserved_envs_per_actor,
        noleague_reserved_envs=runtime._noleague_baseline_reserved_envs_per_actor,
    )
    _assign_collection_backend(runtime, config=config, system_config=system_config, model_kind=model_kind)
    _assign_process_collector_state(runtime)
    _assign_fixed_opponent_backend(runtime, training_config=training_config)
    runtime._profile_timers = bool(getattr(training_config, "profile_timers", False))
    runtime._debug_validate_sampled_packed_actions = (
        os.environ.get("WEISS_DEBUG_VALIDATE_SAMPLED_PACKED_ACTIONS", "").strip() == "1"
    )
    runtime._batch_timer_metrics = {}

    return RuntimeConfigurationContext(
        model_kind=model_kind,
        training_config=training_config,
        structured_warmstart_cfg=structured_warmstart_cfg,
        actor_torch_threads=None if system_config is None else int(system_config.actor_torch_threads),
    )


def resolve_runtime_actor_device(
    stack: Any,
    *,
    learner_device: torch.device | str | None = None,
) -> torch.device:
    system = stack.config.system
    requested = "cpu" if system is None else str(system.actor_device).strip()
    prefer_process_collectors = "," in requested or is_cuda_auto_request(requested)
    resolved = resolve_actor_device_layout(
        stack,
        actor_count=1,
        learner_device=learner_device,
        prefer_process_collectors=prefer_process_collectors,
    )[0]
    return torch.device(resolved)


def _validate_runtime_config(config: QueueRuntimeConfig) -> None:
    if config.actor_count < 1:
        raise ValueError("actor_count must be >= 1")
    if config.envs_per_actor < 1:
        raise ValueError("envs_per_actor must be >= 1")
    if config.batch_unrolls_per_update < 1:
        raise ValueError("batch_unrolls_per_update must be >= 1")
    if config.queue_capacity_unrolls < config.batch_unrolls_per_update:
        raise ValueError("queue_capacity_unrolls must be >= batch_unrolls_per_update")
    if config.mode == "train_ordered" and config.batch_unrolls_per_update < config.actor_count:
        raise ValueError("train_ordered requires batch_unrolls_per_update >= actor_count")


def _assign_runtime_teacher_settings(runtime: Any, runtime_teacher: Any) -> None:
    runtime._teacher_guidance_enabled = runtime_teacher.teacher_guidance_enabled
    runtime._teacher_aux_mode = runtime_teacher.teacher_aux_mode
    runtime._teacher_label_profiles = runtime_teacher.teacher_label_profiles
    runtime._teacher_label_profile_mode = runtime_teacher.teacher_label_profile_mode
    runtime._teacher_label_profiles_end_updates = runtime_teacher.teacher_label_profiles_end_updates
    runtime._teacher_guidance_warmstart_updates = runtime_teacher.teacher_guidance_warmstart_updates


def _assign_runtime_training_settings(runtime: Any, runtime_training: Any) -> None:
    runtime._actor_policy_backend = runtime_training.actor_policy_backend
    runtime._actor_heuristic_fraction = runtime_training.actor_heuristic_fraction
    runtime._actor_heuristic_start_updates = runtime_training.actor_heuristic_start_updates
    runtime._actor_heuristic_end_updates = runtime_training.actor_heuristic_end_updates
    runtime._actor_heuristic_final_fraction = runtime_training.actor_heuristic_final_fraction
    runtime._train_on_heuristic_actor_rows = runtime_training.train_on_heuristic_actor_rows
    runtime._diverse_opponent_actor_count = runtime_training.diverse_opponent_actor_count
    runtime._diverse_model_actor_count = runtime_training.diverse_model_actor_count
    runtime._diverse_opponent_batch_fraction = runtime_training.diverse_opponent_batch_fraction
    runtime._diverse_opponent_batch_wait_ms = runtime_training.diverse_opponent_batch_wait_ms
    runtime._heuristic_actor_hidden_state_tracking = runtime_training.heuristic_actor_hidden_state_tracking
    runtime._trajectory_retention_enabled = runtime_training.trajectory_retention_enabled
    runtime._trajectory_retention_policy_ids = runtime_training.trajectory_retention_policy_ids
    runtime._trajectory_retention_sources = runtime_training.trajectory_retention_sources
    runtime._actor_behavior_values_required = runtime_training.actor_behavior_values_required


def _assign_reserved_fixed_lane_settings(runtime: Any, *, config: QueueRuntimeConfig) -> None:
    runtime._heuristic_public_reserved_envs_per_actor = 0
    runtime._noleague_baseline_reserved_envs_per_actor = 0
    if runtime._league_config is not None:
        sampling_cfg = getattr(runtime._league_config, "sampling", runtime._league_config)
        runtime._heuristic_public_reserved_envs_per_actor = int(
            getattr(sampling_cfg, "heuristic_public_reserved_envs_per_actor", 0)
        )
        runtime._noleague_baseline_reserved_envs_per_actor = int(
            getattr(sampling_cfg, "noleague_baseline_reserved_envs_per_actor", 0)
        )
    reserved = runtime._heuristic_public_reserved_envs_per_actor + runtime._noleague_baseline_reserved_envs_per_actor
    if int(reserved) > int(config.envs_per_actor):
        raise ValueError("league.sampling reserved env counts per actor cannot exceed training.envs_per_actor")


def _structured_fixed_opponents_expected(
    *,
    model_kind: str,
    structured_warmstart_cfg: Any,
    heuristic_reserved_envs: int,
    noleague_reserved_envs: int,
) -> bool:
    return bool(
        model_kind == "structured_v2"
        and (
            bool(getattr(structured_warmstart_cfg, "enabled", False))
            or int(heuristic_reserved_envs) > 0
            or int(noleague_reserved_envs) > 0
        )
    )


def _assign_collection_backend(
    runtime: Any,
    *,
    config: QueueRuntimeConfig,
    system_config: Any,
    model_kind: str,
) -> None:
    requested_collection_backend = (
        "auto" if system_config is None else str(getattr(system_config, "collection_backend", "auto")).strip().lower()
    )
    backend_selection = select_runtime_collection_backend(
        config=config,
        requested_collection_backend=requested_collection_backend,
        requested_actor_device=runtime._requested_actor_device,
        process_actor_device_names=runtime._process_actor_device_names,
        actor_device=runtime._device,
        actor_policy_backend=runtime._actor_policy_backend,
        model_kind=model_kind,
        league_enabled=bool(runtime._league_enabled),
    )
    runtime._collection_backend = backend_selection.collection_backend
    runtime._use_process_collectors = bool(backend_selection.use_process_collectors)
    runtime._use_central_batched_collection = bool(backend_selection.use_central_batched_collection)
    runtime._use_shared_collector_transport = bool(runtime._use_process_collectors)
    runtime._use_simulator_fused_logits_step = bool(
        config.mode == "train_async_fast" and str(config.profile).strip().lower() == "fast" and model_kind == "mlp"
    )


def _assign_process_collector_state(runtime: Any) -> None:
    runtime._process_context = None
    runtime._collector_processes = []
    runtime._collector_control_queues = []
    runtime._collector_free_queues = []
    runtime._collector_result_queue = None
    runtime._collector_shared_slots = {}


def _assign_fixed_opponent_backend(runtime: Any, *, training_config: Any) -> None:
    fixed_opponent_backend = str(getattr(training_config, "fixed_opponent_backend", "python_scalar")).strip().lower()
    if fixed_opponent_backend not in {"python_scalar", "python_batched", "simulator_native"}:
        raise ValueError(
            "training.fixed_opponent_backend must be one of: python_scalar, python_batched, simulator_native"
        )
    runtime._fixed_opponent_backend = fixed_opponent_backend


__all__ = [
    "RuntimeConfigurationContext",
    "initialize_runtime_configuration",
    "resolve_runtime_actor_device",
]
