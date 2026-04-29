"""Collector process and actor-state helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.eval.policy_set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.runtime_shared import (
    _open_shared_collector_slot,
    _shared_unroll_metadata,
    _write_unroll_to_shared_slot,
)
from weiss_rl.runtime_types import QueueRuntimeConfig
from weiss_rl.termination_reason import classify_episode_end_reason

_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"


def _configure_runtime_actor_torch_threads(actor_torch_threads: int) -> None:
    threads = int(actor_torch_threads)
    if threads < 1:
        raise ValueError("actor_torch_threads must be >= 1")
    torch.set_num_threads(threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _serialize_state_dict_for_ipc(state_dict: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            serialized[str(key)] = np.array(value.detach().cpu().numpy(), copy=True)
        else:
            serialized[str(key)] = copy.deepcopy(value)
    return serialized


def _deserialize_state_dict_from_ipc(state_dict: dict[str, Any]) -> dict[str, Any]:
    restored: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            restored[str(key)] = torch.from_numpy(np.array(value, copy=True))
        else:
            restored[str(key)] = copy.deepcopy(value)
    return restored


def _model_guidance_payload(model: PolicyValueModel | None) -> dict[str, float]:
    if model is None:
        return {}
    get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
    if not callable(get_bias_scale):
        return {}
    return {
        "public_heuristic_logit_bias_scale": float(get_bias_scale(scoring_mode="learner")),
        "public_heuristic_actor_logit_bias_scale": float(get_bias_scale(scoring_mode="actor")),
    }


def _restore_model_guidance_from_payload(model: PolicyValueModel | None, payload: Mapping[str, object]) -> None:
    if model is None:
        return
    set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
    if not callable(set_bias_scale):
        return
    learner_scale = payload.get("public_heuristic_logit_bias_scale")
    actor_scale = payload.get("public_heuristic_actor_logit_bias_scale")
    if learner_scale is None and actor_scale is None:
        return
    resolved_learner_scale = None if learner_scale is None else float(learner_scale)
    resolved_actor_scale = None if actor_scale is None else float(actor_scale)
    if resolved_learner_scale is None:
        get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
        if not callable(get_bias_scale):
            return
        resolved_learner_scale = float(get_bias_scale(scoring_mode="learner"))
    set_bias_scale(resolved_learner_scale, actor_value=resolved_actor_scale)


def _is_cuda_auto_request(requested: str) -> bool:
    normalized = str(requested).strip().lower()
    return normalized in {"auto", "cuda:auto", "cuda:all", "all"}


def _available_cuda_device_names() -> tuple[str, ...]:
    if not torch.cuda.is_available():
        return ()
    return tuple(f"cuda:{index}" for index in range(int(torch.cuda.device_count())))


def _normalize_device_name(requested: str) -> str:
    value = str(requested).strip()
    if not value:
        return "cpu"
    if _is_cuda_auto_request(value):
        available = _available_cuda_device_names()
        return "cpu" if not available else available[0]
    if value.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else int(device.index)}"
    return str(device)


def _configured_learner_device_name(
    stack: StackConfig,
    *,
    learner_device: torch.device | str | None = None,
) -> str:
    if learner_device is not None:
        return _normalize_device_name(str(learner_device))
    system = stack.config.system
    requested = "cpu" if system is None else str(getattr(system, "learner_device", "cpu")).strip()
    return _normalize_device_name(requested)


def resolve_actor_device_layout(
    stack: StackConfig,
    *,
    actor_count: int,
    learner_device: torch.device | str | None = None,
    prefer_process_collectors: bool = False,
    rank_local_cuda_auto: bool = False,
) -> tuple[str, ...]:
    count = max(1, int(actor_count))
    system = stack.config.system
    requested = "cpu" if system is None else str(getattr(system, "actor_device", "cpu")).strip()
    if not requested:
        requested = "cpu"
    requested_parts = tuple(part.strip() for part in requested.split(",") if part.strip())
    if len(requested_parts) > 1:
        normalized = tuple(_normalize_device_name(part) for part in requested_parts)
        return tuple(normalized[index % len(normalized)] for index in range(count))
    if _is_cuda_auto_request(requested):
        available = _available_cuda_device_names()
        if not available:
            return ("cpu",) * count
        learner_name = _configured_learner_device_name(stack, learner_device=learner_device)
        if rank_local_cuda_auto and torch.device(learner_name).type == "cuda":
            return (learner_name,) * count
        actor_pool = tuple(device_name for device_name in available if device_name != learner_name)
        if not actor_pool:
            actor_pool = available
        if not prefer_process_collectors:
            return (actor_pool[0],) * count
        return tuple(actor_pool[index % len(actor_pool)] for index in range(count))
    normalized = _normalize_device_name(requested)
    return (normalized,) * count


def _resolve_runtime_actor_device(
    stack: StackConfig,
    *,
    learner_device: torch.device | str | None = None,
) -> torch.device:
    system = stack.config.system
    requested = "cpu" if system is None else str(system.actor_device).strip()
    prefer_process_collectors = "," in requested or _is_cuda_auto_request(requested)
    resolved = resolve_actor_device_layout(
        stack,
        actor_count=1,
        learner_device=learner_device,
        prefer_process_collectors=prefer_process_collectors,
    )[0]
    return torch.device(resolved)


def _maybe_compile_runtime_actor_model(model: PolicyValueModel, *, enabled: bool) -> Any | None:
    if not enabled:
        return None
    if bool(getattr(model, "supports_legal_candidate_scoring", False)):
        enable_trunk_compile = getattr(model, "enable_trunk_compile", None)
        if not callable(enable_trunk_compile):
            return None
        try:
            enable_trunk_compile(mode="reduce-overhead")
        except Exception:
            return None
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return None


@dataclass(slots=True)
class _ActorState:
    actor_id: int
    env: DecisionBoundaryEnv
    model: PolicyValueModel
    compiled_model: Any | None
    rng: np.random.Generator
    seat_hidden: torch.Tensor
    current_batch: DecisionBoundaryBatch
    layout_name: str
    focal_seat_by_env: np.ndarray
    opponent_policy_id_by_env: np.ndarray
    opponent_hidden: torch.Tensor
    diverse_opponent_lane: bool
    force_model_policy_lane: bool
    fixed_opponent_policy_id_by_env: np.ndarray | None = None
    snapshot_version: int = 0
    next_unroll_seq: int = 0


class PerformanceLogger:
    """Write runtime performance records as JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _collector_counter_template() -> dict[str, int]:
    return {
        "engine_fault_done_rows": 0,
        "no_progress_timeout_rows": 0,
        "natural_timeout_rows": 0,
        "decision_limit_timeout_rows": 0,
        "tick_limit_timeout_rows": 0,
        "timeout_unknown_rows": 0,
        "total_actions": 0,
        "pass_actions": 0,
        "main_move_actions": 0,
        "pass_with_nonpass_available": 0,
        "max_consecutive_main_moves": 0,
        "focal_row_count": 0,
        "opponent_row_count": 0,
        "tactical_row_count": 0,
        "teacher_tactical_row_count": 0,
        "fixed_opponent_tactical_row_count": 0,
        "packed_candidate_count": 0,
        "pfsp_sampled_envs": 0,
        "pfsp_mirror_envs": 0,
        "pfsp_heuristic_public_envs": 0,
        "pfsp_heuristic_public_variant_envs": 0,
        "pfsp_noleague_baseline_envs": 0,
        "pfsp_champion_envs": 0,
        "pfsp_recent_envs": 0,
        "pfsp_hard_negative_envs": 0,
        "pfsp_residual_opponent_envs": 0,
        "pfsp_warmup_snapshot_envs": 0,
        "b1_opponent_env_steps": 0,
        "b1_opponent_train_rows": 0,
        "native_rollout_profile_base_unrolls": 0,
        "native_rollout_profile_aggressive_unrolls": 0,
        "native_rollout_profile_control_unrolls": 0,
        "actor_model_rows": 0,
        "actor_heuristic_rows": 0,
        "policy_train_model_rows": 0,
        "policy_train_heuristic_rows": 0,
        "policy_excluded_heuristic_rows": 0,
        "copied_bytes_estimate": 0,
        "collect_actor_unroll_ms": 0,
        "actor_policy_forward_ms": 0,
        "actor_env_step_ms": 0,
        "actor_action_summary_ms": 0,
        "actor_done_reset_ms": 0,
        "actor_bootstrap_ms": 0,
        "teacher_label_ms": 0,
        "fixed_opponent_routing_ms": 0,
        "simulator_select_actions_from_logits_count": 0,
        "simulator_select_actions_from_logits_ns": 0,
        "simulator_sample_actions_from_logits_count": 0,
        "simulator_sample_actions_from_logits_ns": 0,
        "simulator_step_select_from_logits_into_i16_legal_ids_count": 0,
        "simulator_step_select_from_logits_into_i16_legal_ids_ns": 0,
        "simulator_step_sample_from_logits_into_i16_legal_ids_count": 0,
        "simulator_step_sample_from_logits_into_i16_legal_ids_ns": 0,
        "simulator_step_sample_from_logits_with_logp_into_i16_legal_ids_count": 0,
        "simulator_step_sample_from_logits_with_logp_into_i16_legal_ids_ns": 0,
        "simulator_legal_ids_materialize_count": 0,
        "simulator_legal_ids_materialize_ns": 0,
        "simulator_legal_action_meta_materialize_count": 0,
        "simulator_legal_action_meta_materialize_ns": 0,
        "simulator_python_reset": 0,
        "simulator_python_step": 0,
        "simulator_python_step_sample_from_logits": 0,
        "simulator_python_step_sample_from_logits_with_logp": 0,
        "simulator_python_reset_done": 0,
    }


def _timeout_limits_for_env(env: DecisionBoundaryEnv) -> dict[str, int | None]:
    return {
        "max_decisions": _optional_int(getattr(env, "max_decisions", None)),
        "max_ticks": _optional_int(getattr(env, "max_ticks", None)),
        "max_no_progress_decisions": _optional_int(getattr(env, "max_no_progress_decisions", None)),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _merge_simulator_timing_counters(counters: dict[str, int], env: DecisionBoundaryEnv) -> None:
    drain_timing_counters = getattr(env, "drain_timing_counters", None)
    if not callable(drain_timing_counters):
        return
    for key, value in drain_timing_counters().items():
        counters[f"simulator_{str(key)}"] = counters.get(f"simulator_{str(key)}", 0) + int(value)


def _accumulate_timeout_counters(
    *,
    counters: dict[str, int],
    batch: DecisionBoundaryBatch,
    done: np.ndarray,
    timeout_limits: dict[str, int | None],
) -> None:
    done_mask = np.asarray(done, dtype=np.bool_)
    if not np.any(done_mask):
        return
    decision_count = np.asarray(
        getattr(batch, "decision_count", np.zeros(done_mask.shape, dtype=np.int32)), dtype=np.int64
    )
    tick_count = np.asarray(getattr(batch, "tick_count", np.zeros(done_mask.shape, dtype=np.int32)), dtype=np.int64)
    no_progress_count = np.asarray(
        getattr(batch, "no_progress_count", np.zeros(done_mask.shape, dtype=np.int32)),
        dtype=np.int64,
    )
    terminated = np.asarray(batch.terminated, dtype=np.bool_)
    truncated = np.asarray(batch.truncated, dtype=np.bool_)
    engine_status = np.asarray(batch.engine_status, dtype=np.int64)
    for env_index in np.flatnonzero(done_mask):
        reason = classify_episode_end_reason(
            terminated=bool(terminated[int(env_index)]),
            truncated=bool(truncated[int(env_index)]),
            engine_status=int(engine_status[int(env_index)]),
            decision_count=int(decision_count[int(env_index)]),
            tick_count=int(tick_count[int(env_index)]),
            no_progress_count=int(no_progress_count[int(env_index)]),
            max_decisions=timeout_limits["max_decisions"],
            max_ticks=timeout_limits["max_ticks"],
            max_no_progress_decisions=timeout_limits["max_no_progress_decisions"],
        )
        if reason == "engine_fault":
            counters["engine_fault_done_rows"] += 1
        elif reason == "no_progress_timeout":
            counters["no_progress_timeout_rows"] += 1
        elif reason == "decision_limit_timeout":
            counters["natural_timeout_rows"] += 1
            counters["decision_limit_timeout_rows"] += 1
        elif reason == "tick_limit_timeout":
            counters["natural_timeout_rows"] += 1
            counters["tick_limit_timeout_rows"] += 1
        elif reason == "timeout_unknown":
            counters["natural_timeout_rows"] += 1
            counters["timeout_unknown_rows"] += 1


def _packed_legal_views_from_step_out(step_out: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    legal_offsets = np.asarray(step_out.legal_offsets, dtype=np.uint32)
    used = 0 if legal_offsets.size == 0 else int(legal_offsets[-1])
    legal_ids = np.asarray(step_out.legal_ids, dtype=np.uint32)[:used]
    raw_meta = getattr(step_out, "legal_action_meta", None)
    legal_action_meta = None if raw_meta is None else np.asarray(raw_meta, dtype=np.uint16)[:used]
    return legal_ids, legal_offsets, legal_action_meta


def _actor_inference_model(actor: _ActorState) -> Any:
    return actor.compiled_model if actor.compiled_model is not None else actor.model


def _process_debug_log(*, run_dir: Path | None, actor_id: int, message: str) -> None:
    if str(os.environ.get("WEISS_RL_PROCESS_DEBUG", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    if run_dir is None:
        return
    log_path = Path(run_dir) / "training" / "logs" / f"collector_debug_actor{int(actor_id):02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.time():.6f} {message}\n")


def _handle_collector_commands(
    *,
    runtime: Any,
    actor: _ActorState,
    control_queue: Any,
    default_fixed_slots: np.ndarray | None,
    default_forced_policy_ids: tuple[str, ...],
    default_teacher_active: bool,
    default_has_noleague_baseline: bool,
) -> bool:
    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return False
        kind = str(command.get("kind", ""))
        _process_debug_log(
            run_dir=getattr(runtime, "_run_dir", None),
            actor_id=getattr(actor, "actor_id", -1),
            message=f"command kind={kind}",
        )
        if kind == "stop":
            return True
        if kind == "reload":
            actor.model.load_state_dict(_deserialize_state_dict_from_ipc(command["model_state_dict"]))
            _restore_model_guidance_from_payload(actor.model, command)
            actor.model.eval()
            update = int(command.get("update", actor.snapshot_version))
            actor.snapshot_version = update
            runtime._current_learner_update = update
            if "effective_update" in command:
                runtime._effective_learner_update = int(command["effective_update"])
            if bool(command.get("refresh_opponent_pool", False)):
                _process_debug_log(
                    run_dir=getattr(runtime, "_run_dir", None),
                    actor_id=getattr(actor, "actor_id", -1),
                    message="command reload refresh_opponent_pool start",
                )
                runtime.refresh_opponent_pool()
                _process_debug_log(
                    run_dir=getattr(runtime, "_run_dir", None),
                    actor_id=getattr(actor, "actor_id", -1),
                    message="command reload refresh_opponent_pool done",
                )
            continue
        if kind == "set_update":
            runtime._current_learner_update = int(command.get("update", getattr(runtime, "_current_learner_update", 0)))
            if "effective_update" in command:
                runtime._effective_learner_update = int(command["effective_update"])
            if bool(command.get("refresh_opponent_pool", False)):
                _process_debug_log(
                    run_dir=getattr(runtime, "_run_dir", None),
                    actor_id=getattr(actor, "actor_id", -1),
                    message="command set_update refresh_opponent_pool start",
                )
                runtime.refresh_opponent_pool()
                _process_debug_log(
                    run_dir=getattr(runtime, "_run_dir", None),
                    actor_id=getattr(actor, "actor_id", -1),
                    message="command set_update refresh_opponent_pool done",
                )
            continue
        if kind == "refresh_opponent_pool":
            if "update" in command:
                runtime._current_learner_update = int(command["update"])
            if "effective_update" in command:
                runtime._effective_learner_update = int(command["effective_update"])
            _process_debug_log(
                run_dir=getattr(runtime, "_run_dir", None),
                actor_id=getattr(actor, "actor_id", -1),
                message="command refresh_opponent_pool start",
            )
            runtime.refresh_opponent_pool()
            _process_debug_log(
                run_dir=getattr(runtime, "_run_dir", None),
                actor_id=getattr(actor, "actor_id", -1),
                message="command refresh_opponent_pool done",
            )
            continue
        if kind == "set_league_eval_warmup_gate":
            runtime._league_eval_warmup_gate_open = bool(command.get("open", True))
            _process_debug_log(
                run_dir=getattr(runtime, "_run_dir", None),
                actor_id=getattr(actor, "actor_id", -1),
                message=f"command set_league_eval_warmup_gate open={runtime._league_eval_warmup_gate_open}",
            )
            runtime.refresh_opponent_pool()
            continue
        if kind == "set_fixed_opponents":
            restore_defaults = bool(command.get("restore_defaults", False))
            activate_teacher = (
                default_teacher_active if restore_defaults else bool(command.get("activate_teacher_heuristic", False))
            )
            if activate_teacher and runtime._teacher_policy is not None:
                runtime._opponent_heuristic_policies[HEURISTIC_PUBLIC_POLICY_ID] = runtime._teacher_policy
            elif not default_teacher_active:
                runtime._opponent_heuristic_policies.pop(HEURISTIC_PUBLIC_POLICY_ID, None)

            if restore_defaults:
                runtime._forced_fixed_opponent_policy_ids = tuple(default_forced_policy_ids)
            else:
                runtime._forced_fixed_opponent_policy_ids = tuple(
                    str(policy_id) for policy_id in command.get("forced_policy_ids", ())
                )

            baseline_state_dict = None if restore_defaults else command.get("noleague_baseline_state_dict")
            if baseline_state_dict is not None:
                baseline_model = build_policy_value_model(
                    observation_dim=int(runtime.observation_dim),
                    config=runtime.stack.config.model,
                    action_dim=int(runtime.action_dim),
                    observation_spec=runtime._observation_spec,
                    spec_bundle=runtime._spec_bundle,
                ).to(runtime._device)
                baseline_model.load_state_dict(_deserialize_state_dict_from_ipc(baseline_state_dict))
                baseline_model.eval()
                runtime._opponent_models[_NOLEAGUE_BASELINE_POLICY_ID] = baseline_model
                runtime._opponent_model_locks[_NOLEAGUE_BASELINE_POLICY_ID] = threading.Lock()
            elif restore_defaults and not default_has_noleague_baseline:
                runtime._opponent_models.pop(_NOLEAGUE_BASELINE_POLICY_ID, None)
                runtime._opponent_model_locks.pop(_NOLEAGUE_BASELINE_POLICY_ID, None)

            if restore_defaults:
                actor.fixed_opponent_policy_id_by_env = (
                    None if default_fixed_slots is None else np.asarray(default_fixed_slots, dtype=object).copy()
                )
            else:
                fixed_slots = command.get("fixed_opponent_policy_id_by_env")
                actor.fixed_opponent_policy_id_by_env = (
                    None if fixed_slots is None else np.asarray(fixed_slots, dtype=object)
                )
            runtime._reset_actor_state_for_fixed_opponents(actor)


def _collector_process_main(
    *,
    stack: StackConfig,
    config: QueueRuntimeConfig,
    model_state_dict: dict[str, Any],
    model_guidance_payload: dict[str, float],
    observation_dim: int,
    action_dim: int,
    observation_spec: dict[str, Any] | None,
    spec_bundle: dict[str, Any] | None,
    run_dir: str | None,
    actor_id: int,
    actor_device_name: str | None,
    learner_device_name: str | None,
    control_queue: Any,
    free_queue: Any | None,
    result_queue: Any,
    shared_slot_configs: list[dict[str, Any]] | None,
    initial_learner_update: int = 0,
) -> None:
    from weiss_rl.runtime import QueueRuntime

    system_config = stack.config.system
    stack_for_child = stack
    if system_config is not None:
        child_system = system_config
        if actor_device_name is not None:
            child_system = replace(child_system, actor_device=str(actor_device_name))
        if learner_device_name is not None:
            child_system = replace(child_system, learner_device=str(learner_device_name))
        if child_system is not system_config:
            stack_for_child = replace(
                stack,
                config=replace(
                    stack.config,
                    system=child_system,
                ),
            )
            system_config = stack_for_child.config.system
    if (
        system_config is not None
        and str(getattr(system_config, "collection_backend", "auto")).strip().lower() == "process"
    ):
        stack_for_child = replace(
            stack_for_child,
            config=replace(
                stack_for_child.config,
                system=replace(system_config, collection_backend="auto"),
            ),
        )
        system_config = stack_for_child.config.system
    if system_config is not None:
        _configure_runtime_actor_torch_threads(int(getattr(system_config, "actor_torch_threads", 1)))
    model_config = stack_for_child.config.model
    if model_config is None:
        raise RuntimeError("stack config is missing model config")
    model = build_policy_value_model(
        observation_dim=int(observation_dim),
        config=model_config,
        action_dim=int(action_dim),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(torch.device("cpu"))
    model.load_state_dict(_deserialize_state_dict_from_ipc(model_state_dict))
    _restore_model_guidance_from_payload(model, model_guidance_payload)
    model.eval()
    local_config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=1,
        envs_per_actor=int(config.envs_per_actor),
        unroll_length=int(config.unroll_length),
        batch_unrolls_per_update=1,
        queue_capacity_unrolls=1,
        profile=str(config.profile),
        base_seed=int(config.base_seed),
        pass_action_id=int(config.pass_action_id),
        actor_reload_interval_updates=int(config.actor_reload_interval_updates),
    )
    shared_slots = (
        None
        if shared_slot_configs is None
        else tuple(_open_shared_collector_slot(shared_slot_config) for shared_slot_config in shared_slot_configs)
    )
    runtime = QueueRuntime(
        stack=stack_for_child,
        config=local_config,
        model=model,
        observation_dim=int(observation_dim),
        action_dim=int(action_dim),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
        run_dir=(None if run_dir is None else Path(run_dir)),
        performance_log_path=None,
        defer_initial_opponent_pool_refresh=True,
        learner_device=(None if learner_device_name is None else learner_device_name),
        initial_learner_update=int(initial_learner_update),
    )
    _process_debug_log(
        run_dir=(None if run_dir is None else Path(run_dir)),
        actor_id=int(actor_id),
        message="collector runtime initialized",
    )
    if int(actor_id) != 0:
        runtime._actors[0].env.close()
        runtime._actors[0] = runtime._build_actor_state(model=model, actor_id=int(actor_id))
    actor = runtime._actors[0]
    runtime.refresh_opponent_pool()
    _process_debug_log(
        run_dir=(None if run_dir is None else Path(run_dir)),
        actor_id=int(actor_id),
        message=(
            "collector initial refresh_opponent_pool done "
            f"opponent_models={sorted(getattr(runtime, '_opponent_models', {}).keys())}"
        ),
    )
    _process_debug_log(
        run_dir=(None if run_dir is None else Path(run_dir)), actor_id=int(actor_id), message="collector actor ready"
    )
    default_fixed_slots = (
        None
        if actor.fixed_opponent_policy_id_by_env is None
        else np.asarray(actor.fixed_opponent_policy_id_by_env, dtype=object).copy()
    )
    default_forced_policy_ids = tuple(getattr(runtime, "_forced_fixed_opponent_policy_ids", ()))
    default_teacher_active = HEURISTIC_PUBLIC_POLICY_ID in runtime._opponent_heuristic_policies
    default_has_noleague_baseline = _NOLEAGUE_BASELINE_POLICY_ID in runtime._opponent_models
    try:
        while True:
            if _handle_collector_commands(
                runtime=runtime,
                actor=actor,
                control_queue=control_queue,
                default_fixed_slots=default_fixed_slots,
                default_forced_policy_ids=default_forced_policy_ids,
                default_teacher_active=default_teacher_active,
                default_has_noleague_baseline=default_has_noleague_baseline,
            ):
                return
            _process_debug_log(
                run_dir=(None if run_dir is None else Path(run_dir)),
                actor_id=int(actor_id),
                message="collector collect start",
            )
            unroll = runtime._collect_actor_unroll(actor)
            _process_debug_log(
                run_dir=(None if run_dir is None else Path(run_dir)),
                actor_id=int(actor_id),
                message="collector collect done",
            )
            if shared_slots is None or free_queue is None:
                result_queue.put(unroll)
                _process_debug_log(
                    run_dir=(None if run_dir is None else Path(run_dir)),
                    actor_id=int(actor_id),
                    message="collector result queued direct",
                )
                continue
            while True:
                if _handle_collector_commands(
                    runtime=runtime,
                    actor=actor,
                    control_queue=control_queue,
                    default_fixed_slots=default_fixed_slots,
                    default_forced_policy_ids=default_forced_policy_ids,
                    default_teacher_active=default_teacher_active,
                    default_has_noleague_baseline=default_has_noleague_baseline,
                ):
                    return
                try:
                    token = free_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if token == "stop":
                    return
                slot_id = int(token)
                if slot_id < 0 or slot_id >= len(shared_slots):
                    raise RuntimeError(f"collector {actor_id} received invalid shared slot token {slot_id}")
                break
            _write_unroll_to_shared_slot(shared_slots[slot_id], unroll)
            result_queue.put(_shared_unroll_metadata(unroll, slot_id=slot_id))
            _process_debug_log(
                run_dir=(None if run_dir is None else Path(run_dir)),
                actor_id=int(actor_id),
                message=f"collector result queued shared slot={slot_id}",
            )
    finally:
        if shared_slots is not None:
            for shared_slot in shared_slots:
                shared_slot.close(unlink=False)
        runtime.close()
