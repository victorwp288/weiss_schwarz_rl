"""Queue-based single-node runtime for deterministic and throughput-aware training."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import queue
import threading
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.action_diagnostics import (
    make_action_sequence_state,
    reset_action_sequence_state,
    update_action_summary_from_ids,
    update_action_summary_from_mask,
)
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval.harness import game_result_from_step
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policy_set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.league.opponent_pool import OpponentPoolSampler, sample_opponent_snapshot_ids
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.masking import (
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
)
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.termination_reason import classify_episode_end_reason

QueueRuntimeMode = Literal["train_ordered", "train_async_fast"]
_MIRROR_OPPONENT_POLICY_ID = "latest_policy_mirror"
_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
_FIXED_OPPONENT_EXCLUSIONS = frozenset({"b1_noleague_baseline"})
_PFSP_TIMEOUT_FILTER_MIN_SAMPLES = 32
_PROMOTION_GATED_RECENT_RESERVOIR_MIN_SIZE = 2
_PFSP_DIVERSITY_FLOOR_SIZE = 2
_TACTICAL_TEACHER_DECISION_KINDS = frozenset({1, 2, 3, 4})
_DEFAULT_ACTION_META_WIDTH = 4


def _configure_runtime_actor_torch_threads(actor_torch_threads: int) -> None:
    threads = int(actor_torch_threads)
    if threads < 1:
        raise ValueError("actor_torch_threads must be >= 1")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


@dataclass(frozen=True, slots=True)
class QueueRuntimeConfig:
    mode: QueueRuntimeMode
    actor_count: int
    envs_per_actor: int
    unroll_length: int
    batch_unrolls_per_update: int
    queue_capacity_unrolls: int
    profile: str
    base_seed: int
    pass_action_id: int
    actor_reload_interval_updates: int

    @property
    def total_envs(self) -> int:
        return int(self.actor_count * self.envs_per_actor)


@dataclass(frozen=True, slots=True)
class RuntimeUnroll:
    actor_id: int
    unroll_seq: int
    behavior_policy_version: int
    unroll_hash: str
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    values: np.ndarray
    legal_actions: LegalActionBatch
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray
    bootstrap_value: np.ndarray
    initial_hidden_state: np.ndarray
    final_hidden_state: np.ndarray
    episode_seed: np.ndarray
    policy_train_mask: np.ndarray
    teacher_family: np.ndarray | None = None
    teacher_slot: np.ndarray | None = None
    teacher_attack_type: np.ndarray | None = None
    teacher_valid: np.ndarray | None = None
    behavior_logits: np.ndarray | None = None
    counters: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class RuntimeBatch:
    learner_batch: dict[str, Any]
    runtime_metrics: dict[str, float]


@dataclass(slots=True)
class _SharedCollectorSlot:
    actor_id: int
    layout_name: str
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    values: np.ndarray
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray
    bootstrap_value: np.ndarray
    initial_hidden_state: np.ndarray
    final_hidden_state: np.ndarray
    episode_seed: np.ndarray
    policy_train_mask: np.ndarray
    teacher_family: np.ndarray
    teacher_slot: np.ndarray
    teacher_attack_type: np.ndarray
    teacher_valid: np.ndarray
    legal_ids: np.ndarray | None
    legal_action_meta: np.ndarray | None
    legal_offsets: np.ndarray | None
    legal_mask: np.ndarray | None
    _segments: tuple[shared_memory.SharedMemory, ...]

    def close(self, *, unlink: bool) -> None:
        seen: set[str] = set()
        for segment in self._segments:
            if segment.name in seen:
                continue
            seen.add(segment.name)
            segment.close()
            if unlink:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass


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


def _sample_actions_from_packed_scores(
    packed_logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    rng: np.random.Generator,
    pass_action_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    packed_scores = np.asarray(packed_logits, dtype=np.float32)
    packed_ids = np.asarray(legal_ids, dtype=np.uint32)
    offsets = np.asarray(legal_offsets, dtype=np.uint32)
    row_count = max(int(offsets.shape[0]) - 1, 0)
    actions = np.empty((row_count,), dtype=np.int64)
    logp = np.empty((row_count,), dtype=np.float32)
    for row_index in range(row_count):
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        if start == end:
            actions[row_index] = int(pass_action_id)
            logp[row_index] = 0.0
            continue
        row_scores = packed_scores[start:end]
        row_ids = packed_ids[start:end]
        row_max = float(np.max(row_scores))
        shifted = (row_scores - row_max).astype(np.float32, copy=False)
        exps = np.exp(shifted, dtype=np.float32)
        denom = float(np.sum(exps, dtype=np.float32))
        if denom <= 0.0:
            raise ValueError(f"row {row_index} has zero denom in softmax over packed legal logits")
        row_logp = shifted - np.log(denom)
        probs = np.exp(row_logp, dtype=np.float32)
        choice = int(rng.choice(probs.shape[0], p=probs))
        actions[row_index] = int(row_ids[choice])
        logp[row_index] = np.float32(row_logp[choice])
    return actions, logp


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
    decision_count = np.asarray(getattr(batch, "decision_count", np.zeros(done_mask.shape, dtype=np.int32)), dtype=np.int64)
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
        if kind == "stop":
            return True
        if kind == "reload":
            actor.model.load_state_dict(command["model_state_dict"])
            actor.model.eval()
            actor.snapshot_version = int(command.get("update", actor.snapshot_version))
            continue
        if kind == "set_fixed_opponents":
            restore_defaults = bool(command.get("restore_defaults", False))
            activate_teacher = default_teacher_active if restore_defaults else bool(command.get("activate_teacher_heuristic", False))
            if activate_teacher and runtime._teacher_policy is not None:
                runtime._opponent_heuristic_policies[HEURISTIC_PUBLIC_POLICY_ID] = runtime._teacher_policy
            elif not default_teacher_active:
                runtime._opponent_heuristic_policies.pop(HEURISTIC_PUBLIC_POLICY_ID, None)

            if restore_defaults:
                runtime._forced_fixed_opponent_policy_ids = tuple(default_forced_policy_ids)
            else:
                runtime._forced_fixed_opponent_policy_ids = tuple(str(policy_id) for policy_id in command.get("forced_policy_ids", ()))

            baseline_state_dict = None if restore_defaults else command.get("noleague_baseline_state_dict")
            if baseline_state_dict is not None:
                baseline_model = build_policy_value_model(
                    observation_dim=int(runtime.observation_dim),
                    config=runtime.stack.config.model,
                    action_dim=int(runtime.action_dim),
                    observation_spec=runtime._observation_spec,
                    spec_bundle=runtime._spec_bundle,
                ).to(runtime._device)
                baseline_model.load_state_dict(baseline_state_dict)
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


def _obs_numpy_dtype_for_profile(profile: str) -> np.dtype[Any]:
    normalized = str(profile).strip().lower()
    if normalized == "debug":
        return np.dtype(np.int32)
    return np.dtype(np.int16)


def _resolve_runtime_actor_device(stack: StackConfig) -> torch.device:
    system = stack.config.system
    requested = "cpu" if system is None else str(system.actor_device).strip()
    if not requested:
        requested = "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


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


def _actor_inference_model(actor: _ActorState) -> Any:
    return actor.compiled_model if actor.compiled_model is not None else actor.model


def _shared_segment_spec(*, actor_id: int, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> dict[str, Any]:
    size = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    return {
        "name": f"weissrl_{actor_id}_{name}_{time.time_ns()}",
        "shape": tuple(int(dim) for dim in shape),
        "dtype": dtype.str,
        "size": size,
    }


def _create_shared_collector_slot_config(
    *,
    actor_id: int,
    profile: str,
    unroll_length: int,
    envs_per_actor: int,
    observation_dim: int,
    action_dim: int,
    hidden_size: int,
    layout_name: str,
    legal_action_meta_width: int = _DEFAULT_ACTION_META_WIDTH,
) -> dict[str, Any]:
    rows = int(unroll_length * envs_per_actor)
    obs_dtype = _obs_numpy_dtype_for_profile(profile)
    specs = {
        "obs": _shared_segment_spec(
            actor_id=actor_id,
            name="obs",
            shape=(int(unroll_length), int(envs_per_actor), int(observation_dim)),
            dtype=obs_dtype,
        ),
        "actions": _shared_segment_spec(actor_id=actor_id, name="actions", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.uint16)),
        "rewards": _shared_segment_spec(actor_id=actor_id, name="rewards", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.float32)),
        "terminated": _shared_segment_spec(actor_id=actor_id, name="terminated", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.bool_)),
        "truncated": _shared_segment_spec(actor_id=actor_id, name="truncated", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.bool_)),
        "to_play_seat": _shared_segment_spec(actor_id=actor_id, name="to_play_seat", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.int8)),
        "behavior_logp": _shared_segment_spec(actor_id=actor_id, name="behavior_logp", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.float32)),
        "values": _shared_segment_spec(actor_id=actor_id, name="values", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.float32)),
        "bootstrap_obs": _shared_segment_spec(actor_id=actor_id, name="bootstrap_obs", shape=(int(envs_per_actor), int(observation_dim)), dtype=np.dtype(np.float32)),
        "bootstrap_actor": _shared_segment_spec(actor_id=actor_id, name="bootstrap_actor", shape=(int(envs_per_actor),), dtype=np.dtype(np.int64)),
        "bootstrap_value": _shared_segment_spec(actor_id=actor_id, name="bootstrap_value", shape=(int(envs_per_actor),), dtype=np.dtype(np.float32)),
        "initial_hidden_state": _shared_segment_spec(actor_id=actor_id, name="initial_hidden_state", shape=(int(envs_per_actor), 2, int(hidden_size)), dtype=np.dtype(np.float32)),
        "final_hidden_state": _shared_segment_spec(actor_id=actor_id, name="final_hidden_state", shape=(int(envs_per_actor), 2, int(hidden_size)), dtype=np.dtype(np.float32)),
        "episode_seed": _shared_segment_spec(actor_id=actor_id, name="episode_seed", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.uint64)),
        "policy_train_mask": _shared_segment_spec(actor_id=actor_id, name="policy_train_mask", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.bool_)),
        "teacher_family": _shared_segment_spec(actor_id=actor_id, name="teacher_family", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.int32)),
        "teacher_slot": _shared_segment_spec(actor_id=actor_id, name="teacher_slot", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.int32)),
        "teacher_attack_type": _shared_segment_spec(actor_id=actor_id, name="teacher_attack_type", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.int32)),
        "teacher_valid": _shared_segment_spec(actor_id=actor_id, name="teacher_valid", shape=(int(unroll_length), int(envs_per_actor)), dtype=np.dtype(np.bool_)),
    }
    if str(layout_name) == "i16_legal_ids":
        specs["legal_ids"] = _shared_segment_spec(actor_id=actor_id, name="legal_ids", shape=(rows * int(action_dim),), dtype=np.dtype(np.uint32))
        specs["legal_action_meta"] = _shared_segment_spec(
            actor_id=actor_id,
            name="legal_action_meta",
            shape=(rows * int(action_dim), int(legal_action_meta_width)),
            dtype=np.dtype(np.uint16),
        )
        specs["legal_offsets"] = _shared_segment_spec(actor_id=actor_id, name="legal_offsets", shape=(rows + 1,), dtype=np.dtype(np.uint32))
    else:
        specs["legal_mask"] = _shared_segment_spec(
            actor_id=actor_id,
            name="legal_mask",
            shape=(int(unroll_length), int(envs_per_actor), int(action_dim)),
            dtype=np.dtype(np.bool_),
        )
    return {
        "actor_id": int(actor_id),
        "layout_name": str(layout_name),
        "specs": specs,
    }


def _open_shared_collector_slot(config: dict[str, Any], *, create: bool = False) -> _SharedCollectorSlot:
    specs = dict(config["specs"])
    segments: list[shared_memory.SharedMemory] = []
    arrays: dict[str, np.ndarray] = {}
    for key, spec in specs.items():
        shape = tuple(int(dim) for dim in spec["shape"])
        dtype = np.dtype(spec["dtype"])
        segment = shared_memory.SharedMemory(name=spec["name"], create=create, size=int(spec["size"]))
        segments.append(segment)
        arrays[key] = np.ndarray(shape, dtype=dtype, buffer=segment.buf)
    return _SharedCollectorSlot(
        actor_id=int(config["actor_id"]),
        layout_name=str(config["layout_name"]),
        obs=arrays["obs"],
        actions=arrays["actions"],
        rewards=arrays["rewards"],
        terminated=arrays["terminated"],
        truncated=arrays["truncated"],
        to_play_seat=arrays["to_play_seat"],
        behavior_logp=arrays["behavior_logp"],
        values=arrays["values"],
        bootstrap_obs=arrays["bootstrap_obs"],
        bootstrap_actor=arrays["bootstrap_actor"],
        bootstrap_value=arrays["bootstrap_value"],
        initial_hidden_state=arrays["initial_hidden_state"],
        final_hidden_state=arrays["final_hidden_state"],
        episode_seed=arrays["episode_seed"],
        policy_train_mask=arrays["policy_train_mask"],
        teacher_family=arrays["teacher_family"],
        teacher_slot=arrays["teacher_slot"],
        teacher_attack_type=arrays["teacher_attack_type"],
        teacher_valid=arrays["teacher_valid"],
        legal_ids=arrays.get("legal_ids"),
        legal_action_meta=arrays.get("legal_action_meta"),
        legal_offsets=arrays.get("legal_offsets"),
        legal_mask=arrays.get("legal_mask"),
        _segments=tuple(segments),
    )


def _shared_unroll_metadata(unroll: RuntimeUnroll) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "kind": "shared_unroll_v1",
        "actor_id": int(unroll.actor_id),
        "unroll_seq": int(unroll.unroll_seq),
        "behavior_policy_version": int(unroll.behavior_policy_version),
        "unroll_hash": str(unroll.unroll_hash),
        "action_space": int(unroll.legal_actions.action_space)
        if unroll.legal_actions.action_space is not None
        else None,
    }
    if unroll.legal_actions.ids is not None and unroll.legal_actions.offsets is not None:
        metadata["legal_kind"] = "packed"
        metadata["legal_ids_size"] = int(unroll.legal_actions.ids.size)
        metadata["has_legal_action_meta"] = bool(unroll.legal_actions.meta is not None)
    else:
        metadata["legal_kind"] = "mask"
    if unroll.counters:
        metadata["counters"] = {str(key): int(value) for key, value in unroll.counters.items()}
    metadata["has_teacher_labels"] = bool(
        unroll.teacher_family is not None
        and unroll.teacher_slot is not None
        and unroll.teacher_attack_type is not None
        and unroll.teacher_valid is not None
    )
    return metadata


def _write_unroll_to_shared_slot(slot: _SharedCollectorSlot, unroll: RuntimeUnroll) -> None:
    slot.obs[...] = unroll.obs
    slot.actions[...] = unroll.actions
    slot.rewards[...] = unroll.rewards
    slot.terminated[...] = unroll.terminated
    slot.truncated[...] = unroll.truncated
    slot.to_play_seat[...] = unroll.to_play_seat
    slot.behavior_logp[...] = unroll.behavior_logp
    slot.values[...] = unroll.values
    slot.bootstrap_obs[...] = unroll.bootstrap_obs
    slot.bootstrap_actor[...] = unroll.bootstrap_actor
    slot.bootstrap_value[...] = unroll.bootstrap_value
    slot.initial_hidden_state[...] = unroll.initial_hidden_state
    slot.final_hidden_state[...] = unroll.final_hidden_state
    slot.episode_seed[...] = unroll.episode_seed
    slot.policy_train_mask[...] = unroll.policy_train_mask
    if (
        unroll.teacher_family is None
        or unroll.teacher_slot is None
        or unroll.teacher_attack_type is None
        or unroll.teacher_valid is None
    ):
        slot.teacher_family.fill(-1)
        slot.teacher_slot.fill(-1)
        slot.teacher_attack_type.fill(-1)
        slot.teacher_valid.fill(False)
    else:
        slot.teacher_family[...] = unroll.teacher_family
        slot.teacher_slot[...] = unroll.teacher_slot
        slot.teacher_attack_type[...] = unroll.teacher_attack_type
        slot.teacher_valid[...] = unroll.teacher_valid
    if slot.legal_ids is not None and slot.legal_offsets is not None:
        assert unroll.legal_actions.ids is not None and unroll.legal_actions.offsets is not None
        ids = np.asarray(unroll.legal_actions.ids, dtype=np.uint32)
        meta = None if unroll.legal_actions.meta is None else np.asarray(unroll.legal_actions.meta, dtype=np.uint16)
        offsets = np.asarray(unroll.legal_actions.offsets, dtype=np.uint32)
        slot.legal_ids[: ids.size] = ids
        if slot.legal_action_meta is not None:
            slot.legal_action_meta[...] = np.iinfo(slot.legal_action_meta.dtype).max
            if meta is not None and meta.size:
                slot.legal_action_meta[: meta.shape[0]] = meta
        slot.legal_offsets[:] = offsets
        return
    assert slot.legal_mask is not None
    slot.legal_mask[...] = unroll.legal_actions.to_mask(
        expected_shape=(int(unroll.obs.shape[0]), int(unroll.obs.shape[1])),
        action_space=int(slot.legal_mask.shape[-1]),
    )


def _read_unroll_from_shared_slot(slot: _SharedCollectorSlot, metadata: dict[str, Any]) -> RuntimeUnroll:
    action_space = metadata.get("action_space")
    if str(metadata.get("legal_kind", "")) == "packed":
        assert slot.legal_ids is not None and slot.legal_offsets is not None
        ids_size = int(metadata["legal_ids_size"])
        legal_actions = LegalActionBatch.from_packed(
            np.array(slot.legal_ids[:ids_size], copy=True),
            np.array(slot.legal_offsets, copy=True),
            meta=(
                None
                if slot.legal_action_meta is None or not bool(metadata.get("has_legal_action_meta", False))
                else np.array(slot.legal_action_meta[:ids_size], copy=True)
            ),
            action_space=None if action_space is None else int(action_space),
        )
    else:
        assert slot.legal_mask is not None
        legal_actions = LegalActionBatch.from_mask(
            np.array(slot.legal_mask, copy=True),
            action_space=None if action_space is None else int(action_space),
        )
    return RuntimeUnroll(
        actor_id=int(metadata["actor_id"]),
        unroll_seq=int(metadata["unroll_seq"]),
        behavior_policy_version=int(metadata["behavior_policy_version"]),
        unroll_hash=str(metadata["unroll_hash"]),
        obs=np.array(slot.obs, copy=True),
        actions=np.array(slot.actions, copy=True),
        rewards=np.array(slot.rewards, copy=True),
        terminated=np.array(slot.terminated, copy=True),
        truncated=np.array(slot.truncated, copy=True),
        to_play_seat=np.array(slot.to_play_seat, copy=True),
        behavior_logp=np.array(slot.behavior_logp, copy=True),
        values=np.array(slot.values, copy=True),
        legal_actions=legal_actions,
        bootstrap_obs=np.array(slot.bootstrap_obs, copy=True),
        bootstrap_actor=np.array(slot.bootstrap_actor, copy=True),
        bootstrap_value=np.array(slot.bootstrap_value, copy=True),
        initial_hidden_state=np.array(slot.initial_hidden_state, copy=True),
        final_hidden_state=np.array(slot.final_hidden_state, copy=True),
        episode_seed=np.array(slot.episode_seed, copy=True),
        policy_train_mask=np.array(slot.policy_train_mask, copy=True),
        teacher_family=(
            np.array(slot.teacher_family, copy=True)
            if bool(metadata.get("has_teacher_labels", False))
            else None
        ),
        teacher_slot=(
            np.array(slot.teacher_slot, copy=True)
            if bool(metadata.get("has_teacher_labels", False))
            else None
        ),
        teacher_attack_type=(
            np.array(slot.teacher_attack_type, copy=True)
            if bool(metadata.get("has_teacher_labels", False))
            else None
        ),
        teacher_valid=(
            np.array(slot.teacher_valid, copy=True)
            if bool(metadata.get("has_teacher_labels", False))
            else None
        ),
        behavior_logits=None,
        counters=(
            None
            if not isinstance(metadata.get("counters"), dict)
            else {str(key): int(value) for key, value in dict(metadata["counters"]).items()}
        ),
    )


def _collector_process_main(
    *,
    stack: StackConfig,
    config: QueueRuntimeConfig,
    model_state_dict: dict[str, Any],
    observation_dim: int,
    action_dim: int,
    observation_spec: dict[str, Any] | None,
    spec_bundle: dict[str, Any] | None,
    actor_id: int,
    control_queue: Any,
    free_queue: Any | None,
    result_queue: Any,
    shared_slot_config: dict[str, Any] | None,
) -> None:
    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("stack config is missing model config")
    model = build_policy_value_model(
        observation_dim=int(observation_dim),
        config=model_config,
        action_dim=int(action_dim),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(torch.device("cpu"))
    model.load_state_dict(model_state_dict)
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
    shared_slot = None if shared_slot_config is None else _open_shared_collector_slot(shared_slot_config)
    runtime = QueueRuntime(
        stack=stack,
        config=local_config,
        model=model,
        observation_dim=int(observation_dim),
        action_dim=int(action_dim),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
        run_dir=None,
        performance_log_path=None,
    )
    if int(actor_id) != 0:
        runtime._actors[0].env.close()
        runtime._actors[0] = runtime._build_actor_state(model=model, actor_id=int(actor_id))
    actor = runtime._actors[0]
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
            unroll = runtime._collect_actor_unroll(actor)
            if shared_slot is None or free_queue is None:
                result_queue.put(unroll)
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
                break
            _write_unroll_to_shared_slot(shared_slot, unroll)
            result_queue.put(_shared_unroll_metadata(unroll))
    finally:
        if shared_slot is not None:
            shared_slot.close(unlink=False)
        runtime.close()


class QueueRuntime:
    """Single-node actor queue runtime with deterministic ordered mode."""

    def __init__(
        self,
        *,
        stack: StackConfig,
        config: QueueRuntimeConfig,
        model: Any,
        observation_dim: int,
        action_dim: int,
        observation_spec: dict[str, Any] | None = None,
        spec_bundle: dict[str, Any] | None = None,
        run_dir: Path | None = None,
        performance_log_path: Path | None = None,
    ) -> None:
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

        self.stack = stack
        self.config = config
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self._observation_spec = None if observation_spec is None else dict(observation_spec)
        self._spec_bundle = None if spec_bundle is None else dict(spec_bundle)
        action_meta_spec = (
            {} if self._spec_bundle is None else dict(self._spec_bundle.get("action_meta_v1", {}))
        )
        self._action_meta_width = int(action_meta_spec.get("width", _DEFAULT_ACTION_META_WIDTH))
        self._device = _resolve_runtime_actor_device(stack)
        self._run_dir = None if run_dir is None else Path(run_dir)
        self._artifact_layout = None if self._run_dir is None else ArtifactLayout.from_run_dir(self._run_dir)
        training_config = stack.config.training
        experiment_config = stack.config.experiment
        self._experiment_role = "" if experiment_config is None else str(experiment_config.role).strip()
        self._actor_amp_enabled = bool(
            training_config is not None
            and bool(training_config.mixed_precision)
            and self._device.type == "cuda"
        )
        self._compile_actor_inference = bool(
            training_config is not None
            and bool(getattr(training_config, "compile_actor_inference", False))
            and self._device.type == "cpu"
        )
        self._league_config = stack.config.league
        self._league_enabled = bool(
            self._artifact_layout is not None
            and self._league_config is not None
            and self._league_config.enabled
            and self._experiment_role != "baseline_noleague"
        )
        self._registry_path = (
            None if self._artifact_layout is None else self._artifact_layout.training_snapshots_dir / REGISTRY_FILENAME
        )
        self._opponent_models: dict[str, PolicyValueModel] = {}
        self._opponent_model_locks: dict[str, threading.Lock] = {}
        self._opponent_heuristic_policies: dict[str, HeuristicPublicPolicy] = {}
        self._teacher_guidance_enabled = bool(
            training_config is not None and bool(getattr(training_config, "structured_aux_enabled", False))
        )
        self._teacher_policy: HeuristicPublicPolicy | None = None
        self._teacher_action_catalog: ActionCatalog | None = None
        self._teacher_family_index: dict[str, int] = {}
        self._teacher_attack_type_index: dict[str, int] = {}
        if self._teacher_guidance_enabled:
            if self._spec_bundle is None:
                raise RuntimeError("structured_aux.enabled requires the runtime spec bundle")
            try:
                self._teacher_policy = HeuristicPublicPolicy.from_spec_bundle(self._spec_bundle)
                self._teacher_action_catalog = ActionCatalog.from_spec_bundle(self._spec_bundle)
            except Exception as exc:
                raise RuntimeError(
                    "Structured teacher guidance requires a heuristic-compatible simulator contract"
                ) from exc
            self._teacher_family_index = {
                family.name: index for index, family in enumerate(self._teacher_action_catalog.families)
            }
            self._teacher_attack_type_index = {
                name: index for index, name in enumerate(self._teacher_action_catalog.attack_type_names)
            }
        self._opponent_sampler: OpponentPoolSampler | None = None
        self._opponent_candidate_ids: tuple[str, ...] = ()
        self._outcomes = OnlineOutcomeTracker(
            window_size=(50_000 if self._league_config is None else int(self._league_config.pfsp_window_episodes))
        )
        self._pfsp_epoch = int(self._outcomes.current_epoch)
        self._current_learner_update = 0
        self._effective_learner_update = 0
        self._published_snapshot_update_by_fingerprint: dict[str, int] = {}
        self._pfsp_pool_size = 0
        self._pfsp_quarantined_opponents = 0
        self._pfsp_champion_pool_size = 0
        self._pfsp_recent_pool_size = 0
        self._pfsp_hard_negative_pool_size = 0
        self._pfsp_last_sampled_envs = 0
        self._pfsp_last_mirror_envs = 0
        self._pfsp_last_heuristic_public_envs = 0
        self._pfsp_last_noleague_baseline_envs = 0
        self._pfsp_last_champion_envs = 0
        self._pfsp_last_recent_envs = 0
        self._pfsp_last_hard_negative_envs = 0
        self._disable_mirror_policy_fusion = False
        self._opponent_champion_ids: tuple[str, ...] = ()
        self._opponent_recent_ids: tuple[str, ...] = ()
        self._opponent_hard_negative_ids: tuple[str, ...] = ()
        heuristic_public_mix_fraction = 0.0
        if self._league_config is not None:
            sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
            heuristic_public_mix_fraction = float(
                getattr(sampling_cfg, "heuristic_public_mix_fraction", 0.0)
            )
        if heuristic_public_mix_fraction > 0.0:
            if self._spec_bundle is None:
                raise RuntimeError(
                    "league.sampling.heuristic_public_mix_fraction > 0 requires the runtime spec bundle"
                )
            try:
                self._opponent_heuristic_policies[HEURISTIC_PUBLIC_POLICY_ID] = HeuristicPublicPolicy.from_spec_bundle(
                    self._spec_bundle
                )
            except Exception as exc:
                raise RuntimeError(
                    "Training-time B2 HeuristicPublic requires a heuristic-compatible simulator contract"
                ) from exc
        self._heuristic_public_reserved_envs_per_actor = 0
        self._noleague_baseline_reserved_envs_per_actor = 0
        if self._league_config is not None:
            sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
            self._heuristic_public_reserved_envs_per_actor = int(
                getattr(sampling_cfg, "heuristic_public_reserved_envs_per_actor", 0)
            )
            self._noleague_baseline_reserved_envs_per_actor = int(
                getattr(sampling_cfg, "noleague_baseline_reserved_envs_per_actor", 0)
            )
        if (
            self._heuristic_public_reserved_envs_per_actor + self._noleague_baseline_reserved_envs_per_actor
            > int(config.envs_per_actor)
        ):
            raise ValueError("league.sampling reserved env counts per actor cannot exceed training.envs_per_actor")
        model_kind = "" if stack.config.model is None else str(stack.config.model.encoder_kind).strip().lower()
        structured_warmstart_cfg = getattr(stack.config.training, "structured_warmstart", None)
        structured_fixed_opponents_expected = bool(
            model_kind == "structured_v2"
            and (
                bool(getattr(structured_warmstart_cfg, "enabled", False))
                or self._heuristic_public_reserved_envs_per_actor > 0
                or self._noleague_baseline_reserved_envs_per_actor > 0
            )
        )
        self._structured_fixed_opponents_expected = structured_fixed_opponents_expected
        self._use_process_collectors = bool(
            config.mode == "train_async_fast"
            and int(config.actor_count) > 1
            and not self._league_enabled
            and self._device.type == "cpu"
            and model_kind != "typed_v1"
        )
        self._use_central_batched_collection = bool(
            config.mode == "train_async_fast"
            and (
                (self._device.type == "cpu" and model_kind in {"typed_v1", "structured_v2"})
                or (self._device.type == "cuda" and model_kind == "structured_v2")
            )
            and (model_kind != "structured_v2" or structured_fixed_opponents_expected)
        )
        if self._use_central_batched_collection:
            self._use_process_collectors = False
        self._use_shared_collector_transport = False
        self._use_simulator_fused_logits_step = bool(
            config.mode == "train_async_fast"
            and str(config.profile).strip().lower() == "fast"
            and model_kind == "mlp"
        )
        self._process_context: Any | None = None
        self._collector_processes: list[Any] = []
        self._collector_control_queues: list[Any] = []
        self._collector_free_queues: list[Any] = []
        self._collector_result_queue: Any | None = None
        self._collector_shared_slots: dict[int, _SharedCollectorSlot] = {}
        self._shared_actor_model = None
        self._shared_compiled_actor_model = None
        fixed_opponent_backend = str(getattr(stack.config.training, "fixed_opponent_backend", "python_scalar")).strip().lower()
        if fixed_opponent_backend not in {"python_scalar", "python_batched", "simulator_native"}:
            raise ValueError(
                "training.fixed_opponent_backend must be one of: "
                "python_scalar, python_batched, simulator_native"
            )
        self._fixed_opponent_backend = fixed_opponent_backend
        self._profile_timers = bool(getattr(stack.config.training, "profile_timers", False))
        self._batch_timer_metrics: dict[str, float] = {}
        if self._use_central_batched_collection:
            self._shared_actor_model = copy.deepcopy(model).to(self._device)
            self._shared_actor_model.eval()
            self._shared_compiled_actor_model = _maybe_compile_runtime_actor_model(
                self._shared_actor_model,
                enabled=self._compile_actor_inference,
            )
        self._bootstrap_models = (
            [copy.deepcopy(model).to(self._device) for _ in range(int(config.actor_count))]
            if self._use_process_collectors
            else None
        )
        self._actors = (
            []
            if self._use_process_collectors
            else [self._build_actor_state(model=model, actor_id=actor_id) for actor_id in range(int(config.actor_count))]
        )
        self._pending_unrolls: deque[RuntimeUnroll] = deque()
        self._next_actor_index = 0
        self._collector_executor = (
            None
            if self._use_process_collectors or self._use_central_batched_collection or len(self._actors) <= 1
            else ThreadPoolExecutor(
                max_workers=len(self._actors),
                thread_name_prefix="weiss-runtime-actor",
            )
        )
        if self._collector_executor is not None and stack.config.system is not None:
            _configure_runtime_actor_torch_threads(int(stack.config.system.actor_torch_threads))
        self._last_published_snapshot_version = 0
        self._performance_logger = None if performance_log_path is None else PerformanceLogger(performance_log_path)
        if self._performance_logger is not None:
            self._performance_logger.log(
                {
                    "kind": "runtime_startup_v1",
                    "actor_device": self._device.type,
                    "compile_actor_inference": bool(self._compile_actor_inference),
                    "fixed_opponent_backend": self._fixed_opponent_backend,
                    "league_enabled": bool(self._league_enabled),
                    "model_kind": model_kind,
                    "structured_fixed_opponents_expected": bool(self._structured_fixed_opponents_expected),
                    "structured_warmstart_enabled": bool(
                        training_config is not None
                        and bool(getattr(training_config, "structured_warmstart_enabled", False))
                    ),
                    "structured_warmstart_flag_enabled": bool(
                        structured_warmstart_cfg is not None
                        and bool(getattr(structured_warmstart_cfg, "enabled", False))
                    ),
                    "use_central_batched_collection": bool(self._use_central_batched_collection),
                    "use_process_collectors": bool(self._use_process_collectors),
                }
            )
        self._runtime_start = time.time()
        self._runtime_last_metrics_time = self._runtime_start
        self._runtime_cumulative_env_steps = 0
        self.refresh_opponent_pool()
        if self._use_process_collectors:
            self._start_process_collectors(model)

    def _reset_batch_timer_metrics(self) -> None:
        self._batch_timer_metrics = {}

    def _record_batch_timer_ms(self, name: str, elapsed_seconds: float) -> None:
        if not bool(getattr(self, "_profile_timers", False)):
            return
        if not hasattr(self, "_batch_timer_metrics"):
            self._batch_timer_metrics = {}
        key = f"timer_runtime_{name}_ms"
        self._batch_timer_metrics[key] = self._batch_timer_metrics.get(key, 0.0) + (float(elapsed_seconds) * 1000.0)

    def _overwrite_central_outputs_with_configured_opponents(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray | None],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        if str(getattr(self, "_fixed_opponent_backend", "python_batched")) == "python_scalar":
            for actor, batch, obs_step, actor_step, logits_out, values_out in zip(
                actors,
                batches,
                obs_steps,
                actor_steps,
                logits_outs,
                values_outs,
                strict=True,
            ):
                self._overwrite_central_outputs_with_batched_opponents(
                    actors=[actor],
                    batches=[batch],
                    obs_steps=[obs_step],
                    actor_steps=[actor_step],
                    logits_outs=[logits_out],
                    values_outs=[values_out],
                )
            return
        self._overwrite_central_outputs_with_batched_opponents(
            actors=actors,
            batches=batches,
            obs_steps=obs_steps,
            actor_steps=actor_steps,
            logits_outs=logits_outs,
            values_outs=values_outs,
        )

    def _set_process_collector_fixed_opponents(
        self,
        *,
        slots: np.ndarray | None,
        forced_policy_ids: Sequence[str],
        activate_teacher_heuristic: bool,
    ) -> None:
        if self._collector_result_queue is None:
            return
        baseline_model = self._opponent_models.get(_NOLEAGUE_BASELINE_POLICY_ID)
        baseline_state_dict = (
            None
            if baseline_model is None or _NOLEAGUE_BASELINE_POLICY_ID not in forced_policy_ids
            else {key: value.detach().cpu().clone() for key, value in baseline_model.state_dict().items()}
        )
        payload = {
            "kind": "set_fixed_opponents",
            "restore_defaults": False,
            "fixed_opponent_policy_id_by_env": None if slots is None else np.asarray(slots, dtype=object).tolist(),
            "forced_policy_ids": tuple(str(policy_id) for policy_id in forced_policy_ids),
            "activate_teacher_heuristic": bool(activate_teacher_heuristic),
            "noleague_baseline_state_dict": baseline_state_dict,
        }
        for control_queue in self._collector_control_queues:
            control_queue.put(payload)

    def _restore_process_collector_fixed_opponents(self) -> None:
        if self._collector_result_queue is None:
            return
        payload = {
            "kind": "set_fixed_opponents",
            "restore_defaults": True,
        }
        for control_queue in self._collector_control_queues:
            control_queue.put(payload)

    @contextmanager
    def structured_warmstart_source_mix(self) -> Any:
        inserted_teacher_heuristic = False
        if (
            self._teacher_policy is not None
            and HEURISTIC_PUBLIC_POLICY_ID not in self._opponent_heuristic_policies
        ):
            self._opponent_heuristic_policies[HEURISTIC_PUBLIC_POLICY_ID] = self._teacher_policy
            inserted_teacher_heuristic = True

        previous_forced_policy_ids = tuple(getattr(self, "_forced_fixed_opponent_policy_ids", ()))
        previous_fixed_slots = [
            (
                None
                if actor.fixed_opponent_policy_id_by_env is None
                else np.asarray(actor.fixed_opponent_policy_id_by_env, dtype=object).copy()
            )
            for actor in self._actors
        ]

        available_sources = ["self_play"]
        if _NOLEAGUE_BASELINE_POLICY_ID in self._opponent_models:
            available_sources.append(_NOLEAGUE_BASELINE_POLICY_ID)
        if HEURISTIC_PUBLIC_POLICY_ID in self._opponent_heuristic_policies:
            available_sources.append(HEURISTIC_PUBLIC_POLICY_ID)

        envs_per_actor = int(self.config.envs_per_actor)
        source_count = max(1, len(available_sources))
        counts_by_source: dict[str, int] = {}
        remaining = envs_per_actor
        for source_index, source_name in enumerate(available_sources):
            slots_left = max(1, source_count - source_index)
            count = int(np.ceil(float(remaining) / float(slots_left)))
            count = max(0, min(count, remaining))
            counts_by_source[source_name] = count
            remaining -= count

        slots = np.full((envs_per_actor,), "", dtype=object)
        cursor = 0
        for source_name in (_NOLEAGUE_BASELINE_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID):
            count = int(counts_by_source.get(source_name, 0))
            if count <= 0:
                continue
            slots[cursor : cursor + count] = source_name
            cursor += count
        forced_policy_ids = tuple(policy_id for policy_id in (_NOLEAGUE_BASELINE_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID) if counts_by_source.get(policy_id, 0) > 0)
        self._forced_fixed_opponent_policy_ids = forced_policy_ids
        try:
            if self._collector_result_queue is not None:
                self._set_process_collector_fixed_opponents(
                    slots=(None if cursor <= 0 else slots.copy()),
                    forced_policy_ids=forced_policy_ids,
                    activate_teacher_heuristic=counts_by_source.get(HEURISTIC_PUBLIC_POLICY_ID, 0) > 0,
                )
            else:
                for actor in self._actors:
                    actor.fixed_opponent_policy_id_by_env = (None if cursor <= 0 else slots.copy())
                    self._reset_actor_state_for_fixed_opponents(actor)
            yield {
                "structured_warmstart_source_count": float(source_count),
                "structured_warmstart_self_play_envs_per_actor": float(counts_by_source.get("self_play", 0)),
                "structured_warmstart_b1_envs_per_actor": float(counts_by_source.get(_NOLEAGUE_BASELINE_POLICY_ID, 0)),
                "structured_warmstart_b2_envs_per_actor": float(counts_by_source.get(HEURISTIC_PUBLIC_POLICY_ID, 0)),
            }
        finally:
            self._forced_fixed_opponent_policy_ids = previous_forced_policy_ids
            if self._collector_result_queue is not None:
                self._restore_process_collector_fixed_opponents()
            else:
                for actor, saved_slots in zip(self._actors, previous_fixed_slots, strict=True):
                    actor.fixed_opponent_policy_id_by_env = saved_slots
                    self._reset_actor_state_for_fixed_opponents(actor)
            if inserted_teacher_heuristic:
                self._opponent_heuristic_policies.pop(HEURISTIC_PUBLIC_POLICY_ID, None)

    @contextmanager
    def disable_mirror_policy_fusion(self) -> Any:
        previous = bool(getattr(self, "_disable_mirror_policy_fusion", False))
        self._disable_mirror_policy_fusion = True
        try:
            yield
        finally:
            self._disable_mirror_policy_fusion = previous

    def close(self) -> None:
        if self._collector_result_queue is not None:
            for control_queue in self._collector_control_queues:
                control_queue.put({"kind": "stop"})
            if self._use_shared_collector_transport:
                for free_queue in self._collector_free_queues:
                    free_queue.put("stop")
            for process in self._collector_processes:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
            self._collector_control_queues.clear()
            self._collector_free_queues.clear()
            self._collector_processes.clear()
            self._collector_result_queue.close()
            self._collector_result_queue = None
        if self._use_shared_collector_transport:
            for slot in self._collector_shared_slots.values():
                slot.close(unlink=True)
            self._collector_shared_slots.clear()
        if self._collector_executor is not None:
            self._collector_executor.shutdown(wait=True)
        for actor in self._actors:
            actor.env.close()

    def maybe_publish_snapshot(
        self,
        *,
        learner_model: PolicyValueModel,
        learner_update_count: int,
        force: bool = False,
    ) -> dict[str, float]:
        self._current_learner_update = int(learner_update_count)
        if learner_update_count <= 0:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}
        if not force and learner_update_count == self._last_published_snapshot_version:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}
        if not force and learner_update_count % int(self.config.actor_reload_interval_updates) != 0:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}

        publish_started = time.perf_counter()
        state_dict = {key: value.detach().cpu().clone() for key, value in learner_model.state_dict().items()}
        state_fingerprint = _hash_state_dict(state_dict)
        published_snapshot_update_by_fingerprint = getattr(self, "_published_snapshot_update_by_fingerprint", None)
        if published_snapshot_update_by_fingerprint is None:
            published_snapshot_update_by_fingerprint = {}
            self._published_snapshot_update_by_fingerprint = published_snapshot_update_by_fingerprint
        published_snapshot_update = int(
            published_snapshot_update_by_fingerprint.setdefault(state_fingerprint, int(learner_update_count))
        )
        self._effective_learner_update = published_snapshot_update
        publish_latency_ms = (time.perf_counter() - publish_started) * 1000.0

        apply_started = time.perf_counter()
        if self._collector_result_queue is not None:
            for control_queue in self._collector_control_queues:
                control_queue.put(
                    {
                        "kind": "reload",
                        "model_state_dict": state_dict,
                        "update": int(learner_update_count),
                    }
                )
        else:
            if self._shared_actor_model is not None:
                self._shared_actor_model.load_state_dict(state_dict)
                self._shared_actor_model.eval()
                for actor in self._actors:
                    actor.snapshot_version = int(learner_update_count)
            else:
                for actor in self._actors:
                    actor.model.load_state_dict(state_dict)
                    actor.model.eval()
                    actor.snapshot_version = int(learner_update_count)
        if self._bootstrap_models is not None:
            for bootstrap_model in self._bootstrap_models:
                bootstrap_model.load_state_dict(state_dict)
                bootstrap_model.eval()
        apply_latency_ms = (time.perf_counter() - apply_started) * 1000.0
        self._last_published_snapshot_version = int(learner_update_count)
        return {
            "snapshot_publish_latency_ms": publish_latency_ms,
            "snapshot_apply_latency_ms": apply_latency_ms,
        }

    def refresh_opponent_pool(self) -> None:
        if not self._league_enabled or self._registry_path is None or not self._registry_path.is_file():
            self._opponent_sampler = None
            self._opponent_candidate_ids = ()
            self._opponent_models = {}
            self._opponent_model_locks = {}
            self._pfsp_pool_size = 0
            self._pfsp_quarantined_opponents = 0
            self._pfsp_champion_pool_size = 0
            self._pfsp_recent_pool_size = 0
            self._pfsp_hard_negative_pool_size = 0
            self._opponent_champion_ids = ()
            self._opponent_recent_ids = ()
            self._opponent_hard_negative_ids = ()
            return
        assert self._league_config is not None
        sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
        pool_cfg = getattr(self._league_config, "pool", self._league_config)
        current_update = int(self._league_reference_update())
        registry = SnapshotRegistry.load(self._registry_path)
        stale_demoted: list[str] = []
        max_age_updates = int(getattr(pool_cfg, "champion_max_age_updates", 0))
        if max_age_updates > 0 and current_update > 0:
            stale_demoted = registry.demote_stale_champions(
                current_update=current_update,
                max_age_updates=max_age_updates,
            )
        admitted_champion_ids = tuple(
            registry.latest_champions(
                int(self._league_config.snapshot_pool_champion_size),
                current_update=current_update,
                max_age_updates=max_age_updates,
            )
        )
        recent_size = int(self._league_config.snapshot_pool_recent_size)
        # Promotion gating should keep rejected snapshots out of the steady-state live PFSP pool.
        # However, if no champion has been admitted yet, forcing recent_size=0 collapses training to
        # mirror-only self-play. Keep a small probationary reservoir before the first champion and a
        # small exploration reservoir afterward so the live pool cannot narrow to champions only.
        if bool(self._league_config.promotion_gate_enabled):
            recent_size = self._promotion_gated_recent_reservoir_size(
                base_recent_size=recent_size,
                champion_size=int(self._league_config.snapshot_pool_champion_size),
                admitted_champion_ids=admitted_champion_ids,
            )
        sampler = OpponentPoolSampler(
            registry=registry,
            recent_size=recent_size,
            champion_size=int(self._league_config.snapshot_pool_champion_size),
            power=float(self._league_config.pfsp_power),
            eps_uniform=float(self._league_config.pfsp_epsilon_uniform),
        )
        self._opponent_sampler = sampler
        champion_ids = tuple(
            policy_id
            for policy_id in admitted_champion_ids
            if policy_id not in _FIXED_OPPONENT_EXCLUSIONS
        )
        recent_ids = tuple(
            policy_id for policy_id in registry.latest_ids(recent_size) if policy_id not in _FIXED_OPPONENT_EXCLUSIONS
        )
        candidate_ids = tuple(dict.fromkeys([*champion_ids, *recent_ids]))
        filtered_candidate_ids = self._filter_timeout_heavy_opponents(candidate_ids)
        candidate_ids, quarantined_count = self._apply_opponent_pool_diversity_floor(
            candidate_ids=candidate_ids,
            filtered_candidate_ids=filtered_candidate_ids,
        )
        self._pfsp_quarantined_opponents = quarantined_count
        hard_negative_ids = self._select_hard_negative_ids(candidate_ids)
        hard_negative_set = set(hard_negative_ids)
        champion_ids = tuple(
            policy_id for policy_id in champion_ids if policy_id in candidate_ids and policy_id not in hard_negative_set
        )
        champion_set = set(champion_ids)
        recent_ids = tuple(
            policy_id
            for policy_id in recent_ids
            if policy_id in candidate_ids and policy_id not in hard_negative_set and policy_id not in champion_set
        )
        candidate_ids = tuple(dict.fromkeys([*hard_negative_ids, *champion_ids, *recent_ids]))
        self._opponent_candidate_ids = candidate_ids
        self._pfsp_pool_size = len(candidate_ids)
        self._opponent_champion_ids = champion_ids
        self._opponent_recent_ids = recent_ids
        self._opponent_hard_negative_ids = hard_negative_ids
        self._pfsp_champion_pool_size = len(champion_ids)
        self._pfsp_recent_pool_size = len(recent_ids)
        self._pfsp_hard_negative_pool_size = len(hard_negative_ids)
        models: dict[str, PolicyValueModel] = {}
        snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
        resident_policy_ids = tuple(
            dict.fromkeys(
                [*candidate_ids, *self._active_assigned_opponent_policy_ids(), *self._configured_fixed_opponent_policy_ids()]
            )
        )
        for policy_id in resident_policy_ids:
            snapshot = snapshots_by_id.get(policy_id)
            if snapshot is None:
                continue
            models[policy_id] = self._load_snapshot_model(snapshot.path)
        self._opponent_models = models
        self._opponent_model_locks = {policy_id: threading.Lock() for policy_id in models}
        if stale_demoted:
            registry.save(self._registry_path)

    def _active_assigned_opponent_policy_ids(self) -> tuple[str, ...]:
        if not hasattr(self, "_actors"):
            return ()
        active_policy_ids: list[str] = []
        for actor in getattr(self, "_actors", ()):
            policy_ids = getattr(actor, "opponent_policy_id_by_env", None)
            if policy_ids is None:
                continue
            for policy_id in np.asarray(policy_ids, dtype=object).tolist():
                policy_id_text = str(policy_id).strip()
                if not policy_id_text or policy_id_text == _MIRROR_OPPONENT_POLICY_ID:
                    continue
                active_policy_ids.append(policy_id_text)
        return tuple(dict.fromkeys(active_policy_ids))

    def _configured_fixed_opponent_policy_ids(self) -> tuple[str, ...]:
        policy_ids: list[str] = []
        if (
            int(getattr(self, "_heuristic_public_reserved_envs_per_actor", 0)) > 0
            and HEURISTIC_PUBLIC_POLICY_ID in self._opponent_heuristic_policies
        ):
            policy_ids.append(HEURISTIC_PUBLIC_POLICY_ID)
        if int(getattr(self, "_noleague_baseline_reserved_envs_per_actor", 0)) > 0:
            policy_ids.append(_NOLEAGUE_BASELINE_POLICY_ID)
        return tuple(dict.fromkeys(policy_ids))

    def _fixed_opponent_policy_slots(self) -> np.ndarray | None:
        envs_per_actor = int(self.config.envs_per_actor)
        slots = np.full((envs_per_actor,), "", dtype=object)
        cursor = 0
        heuristic_count = min(int(getattr(self, "_heuristic_public_reserved_envs_per_actor", 0)), envs_per_actor - cursor)
        if heuristic_count > 0:
            slots[cursor : cursor + heuristic_count] = HEURISTIC_PUBLIC_POLICY_ID
            cursor += heuristic_count
        baseline_count = min(int(getattr(self, "_noleague_baseline_reserved_envs_per_actor", 0)), envs_per_actor - cursor)
        if baseline_count > 0:
            slots[cursor : cursor + baseline_count] = _NOLEAGUE_BASELINE_POLICY_ID
            cursor += baseline_count
        if cursor <= 0:
            return None
        return slots

    def _fixed_opponent_policy_is_active(self, policy_id: str) -> bool:
        policy_id = str(policy_id).strip()
        if not policy_id:
            return False
        forced_policy_ids = tuple(getattr(self, "_forced_fixed_opponent_policy_ids", ()))
        if policy_id in forced_policy_ids:
            if policy_id == HEURISTIC_PUBLIC_POLICY_ID:
                return policy_id in self._opponent_heuristic_policies
            if policy_id == _NOLEAGUE_BASELINE_POLICY_ID:
                return policy_id in self._opponent_models
        reference_update = self._league_reference_update()
        if policy_id == HEURISTIC_PUBLIC_POLICY_ID:
            if self._league_config is None:
                return False
            sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
            start_updates = int(getattr(sampling_cfg, "heuristic_public_start_updates", 0))
            return (
                reference_update >= start_updates
                and HEURISTIC_PUBLIC_POLICY_ID in self._opponent_heuristic_policies
            )
        if policy_id == _NOLEAGUE_BASELINE_POLICY_ID:
            return (
                self._league_config is not None
                and reference_update >= int(self._league_config.warmup.first_updates)
                and policy_id in self._opponent_models
            )
        return policy_id in self._opponent_models or policy_id in self._opponent_heuristic_policies

    def reset_outcome_tracker(self) -> None:
        self._pfsp_epoch = int(self._outcomes.bump_epoch(drop_previous=True))
        self._pfsp_quarantined_opponents = 0

    def _league_reference_update(self) -> int:
        effective_update = int(getattr(self, "_effective_learner_update", 0))
        if effective_update > 0:
            return effective_update
        return int(getattr(self, "_current_learner_update", 0))

    def _promotion_gated_recent_reservoir_size(
        self,
        *,
        base_recent_size: int,
        champion_size: int,
        admitted_champion_ids: Sequence[str],
    ) -> int:
        base_recent_size_i = max(0, int(base_recent_size))
        if base_recent_size_i <= 0:
            return 0
        if not admitted_champion_ids:
            return min(base_recent_size_i, max(1, int(champion_size)))
        return min(
            base_recent_size_i,
            max(_PROMOTION_GATED_RECENT_RESERVOIR_MIN_SIZE, max(1, int(champion_size) // 2)),
        )

    def _filter_timeout_heavy_opponents(self, candidate_ids: Sequence[str]) -> tuple[str, ...]:
        if not candidate_ids or self._league_config is None or not bool(self._league_config.promotion_gate_enabled):
            return tuple(candidate_ids)
        timeout_threshold = float(self._league_config.promotion.gate.guardrails.max_truncation_rate)
        kept: list[str] = []
        for policy_id in candidate_ids:
            wins, losses, draws, timeouts = self._outcomes.counts(policy_id)
            total = int(wins + losses + draws + timeouts)
            if total < _PFSP_TIMEOUT_FILTER_MIN_SAMPLES:
                kept.append(policy_id)
                continue
            timeout_rate = float(timeouts) / float(total)
            if timeout_rate <= timeout_threshold:
                kept.append(policy_id)
        return tuple(kept)

    def _apply_opponent_pool_diversity_floor(
        self,
        *,
        candidate_ids: Sequence[str],
        filtered_candidate_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], int]:
        original_ids = tuple(str(policy_id) for policy_id in candidate_ids)
        filtered_ids = tuple(str(policy_id) for policy_id in filtered_candidate_ids)
        if not original_ids:
            return (), 0
        if not filtered_ids:
            return original_ids, 0
        raw_quarantined_count = max(0, len(original_ids) - len(filtered_ids))
        restored: list[str] = list(filtered_ids)
        minimum_size = min(len(original_ids), _PFSP_DIVERSITY_FLOOR_SIZE)
        if len(restored) < minimum_size:
            restored_set = set(restored)
            for policy_id in original_ids:
                if policy_id in restored_set:
                    continue
                restored.append(policy_id)
                restored_set.add(policy_id)
                if len(restored) >= minimum_size:
                    break
        return tuple(restored), raw_quarantined_count

    def _select_hard_negative_ids(self, candidate_ids: Sequence[str]) -> tuple[str, ...]:
        if not candidate_ids or self._league_config is None or not hasattr(self, "_outcomes"):
            return ()
        sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
        min_samples = int(getattr(sampling_cfg, "hard_negative_min_samples", 16))
        max_win_rate = float(getattr(sampling_cfg, "hard_negative_max_win_rate", 0.45))
        scored: list[tuple[float, int, str]] = []
        snapshots_by_id: dict[str, int] = {}
        if self._registry_path is not None and self._registry_path.is_file():
            registry = SnapshotRegistry.load(self._registry_path)
            snapshots_by_id = {snapshot.policy_id: int(snapshot.update) for snapshot in registry.snapshots}
        for policy_id in candidate_ids:
            wins, losses, draws, timeouts = self._outcomes.counts(policy_id)
            total = int(wins + losses + draws + timeouts)
            if total < min_samples:
                continue
            win_rate = float(self._outcomes.win_rate(policy_id))
            if win_rate <= max_win_rate:
                scored.append((win_rate, -int(snapshots_by_id.get(policy_id, 0)), str(policy_id)))
        scored.sort()
        return tuple(policy_id for _, _, policy_id in scored)

    def collect_update_batch(
        self,
        *,
        gamma: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
        vtrace_rho_bar: float,
        vtrace_c_bar: float,
    ) -> RuntimeBatch:
        batch_started = time.perf_counter()
        self._reset_batch_timer_metrics()
        occupancy_samples: list[float] = []
        fill_started = time.perf_counter()
        self._fill_pending_unrolls(
            target_count=int(self.config.batch_unrolls_per_update),
            occupancy_samples=occupancy_samples,
        )
        self._record_batch_timer_ms("fill_pending_unrolls", time.perf_counter() - fill_started)

        selected = self._select_pending_unrolls()
        selected_keys = {(item.actor_id, item.unroll_seq) for item in selected}
        self._pending_unrolls = deque(
            item for item in self._pending_unrolls if (item.actor_id, item.unroll_seq) not in selected_keys
        )

        build_started = time.perf_counter()
        learner_batch = self._build_learner_batch(
            selected,
            gamma=gamma,
            truncation_reward=truncation_reward,
            truncation_bootstrap_value=truncation_bootstrap_value,
            vtrace_rho_bar=vtrace_rho_bar,
            vtrace_c_bar=vtrace_c_bar,
        )
        self._record_batch_timer_ms("build_learner_batch", time.perf_counter() - build_started)
        runtime_metrics = self._runtime_metrics(selected, occupancy_samples=occupancy_samples)
        self._record_batch_timer_ms("collect_update_batch_total", time.perf_counter() - batch_started)
        if self._performance_logger is not None:
            elapsed = time.time() - self._runtime_start
            log_started = time.perf_counter()
            self._performance_logger.log(
                {
                    "kind": "runtime_performance_v1",
                    "wall_clock_seconds": elapsed,
                    **runtime_metrics,
                    **self._batch_timer_metrics,
                }
            )
            self._record_batch_timer_ms("performance_log", time.perf_counter() - log_started)
        runtime_metrics.update(self._batch_timer_metrics)
        return RuntimeBatch(learner_batch=learner_batch, runtime_metrics=runtime_metrics)

    def collect_policy_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
    ) -> RuntimeBatch:
        batch_started = time.perf_counter()
        self._reset_batch_timer_metrics()
        occupancy_samples: list[float] = []
        fill_started = time.perf_counter()
        self._fill_pending_unrolls(
            target_count=int(self.config.batch_unrolls_per_update),
            occupancy_samples=occupancy_samples,
        )
        self._record_batch_timer_ms("fill_pending_unrolls", time.perf_counter() - fill_started)

        selected = self._select_pending_unrolls()
        selected_keys = {(item.actor_id, item.unroll_seq) for item in selected}
        self._pending_unrolls = deque(
            item for item in self._pending_unrolls if (item.actor_id, item.unroll_seq) not in selected_keys
        )

        build_started = time.perf_counter()
        learner_batch = self._build_ppo_batch(
            selected,
            gamma=gamma,
            gae_lambda=gae_lambda,
            truncation_reward=truncation_reward,
            truncation_bootstrap_value=truncation_bootstrap_value,
        )
        self._record_batch_timer_ms("build_ppo_batch", time.perf_counter() - build_started)
        runtime_metrics = self._runtime_metrics(selected, occupancy_samples=occupancy_samples)
        self._record_batch_timer_ms("collect_policy_batch_total", time.perf_counter() - batch_started)
        if self._performance_logger is not None:
            elapsed = time.time() - self._runtime_start
            log_started = time.perf_counter()
            self._performance_logger.log(
                {
                    "kind": "runtime_performance_v1",
                    "wall_clock_seconds": elapsed,
                    **runtime_metrics,
                    **self._batch_timer_metrics,
                }
            )
            self._record_batch_timer_ms("performance_log", time.perf_counter() - log_started)
        runtime_metrics.update(self._batch_timer_metrics)
        return RuntimeBatch(learner_batch=learner_batch, runtime_metrics=runtime_metrics)

    def _select_pending_unrolls(self) -> list[RuntimeUnroll]:
        batch_size = int(self.config.batch_unrolls_per_update)
        if self.config.mode != "train_ordered":
            return list(self._pending_unrolls)[:batch_size]
        ordered = sorted(
            self._pending_unrolls,
            key=lambda item: (item.behavior_policy_version, item.unroll_seq, item.actor_id),
        )
        if not ordered:
            raise RuntimeError("train_ordered selection requires at least one pending unroll")
        oldest_version = int(ordered[0].behavior_policy_version)
        selected: list[RuntimeUnroll] = []
        current_group: list[RuntimeUnroll] = []
        current_seq: int | None = None
        for item in ordered:
            if int(item.behavior_policy_version) != oldest_version:
                break
            if current_seq is None or int(item.unroll_seq) == current_seq:
                current_group.append(item)
                current_seq = int(item.unroll_seq)
                continue
            if len(selected) + len(current_group) > batch_size:
                break
            selected.extend(current_group)
            current_group = [item]
            current_seq = int(item.unroll_seq)
        if current_group and len(selected) + len(current_group) <= batch_size:
            selected.extend(current_group)
        if not selected:
            raise RuntimeError("train_ordered selection could not produce a same-version batch")
        return selected

    def _next_actor_batch(self, count: int) -> list[_ActorState]:
        if count <= 0:
            return []
        actor_batch: list[_ActorState] = []
        actor_total = len(self._actors)
        batch_size = min(int(count), actor_total)
        for _ in range(batch_size):
            actor_batch.append(self._actors[self._next_actor_index])
            self._next_actor_index = (self._next_actor_index + 1) % actor_total
        return actor_batch

    def _fill_pending_unrolls(self, *, target_count: int, occupancy_samples: list[float]) -> None:
        if self._collector_result_queue is not None:
            while len(self._pending_unrolls) < int(target_count):
                occupancy_samples.append(len(self._pending_unrolls) / float(self.config.queue_capacity_unrolls))
                payload = self._collector_result_queue.get()
                if (not self._use_shared_collector_transport) or isinstance(payload, RuntimeUnroll):
                    self._pending_unrolls.append(payload)
                    continue
                actor_id = int(payload["actor_id"])
                slot = self._collector_shared_slots[actor_id]
                self._pending_unrolls.append(_read_unroll_from_shared_slot(slot, payload))
                self._collector_free_queues[actor_id].put(1)
            return
        if self._use_central_batched_collection:
            while len(self._pending_unrolls) < int(target_count):
                occupancy_samples.append(len(self._pending_unrolls) / float(self.config.queue_capacity_unrolls))
                remaining = int(target_count) - len(self._pending_unrolls)
                actors = self._next_actor_batch(remaining)
                if not actors:
                    break
                self._pending_unrolls.extend(self._collect_actor_unrolls_central(actors))
            return
        while len(self._pending_unrolls) < int(target_count):
            occupancy_samples.append(len(self._pending_unrolls) / float(self.config.queue_capacity_unrolls))
            remaining = int(target_count) - len(self._pending_unrolls)
            actors = self._next_actor_batch(remaining)
            if not actors:
                break
            if self._collector_executor is None or len(actors) == 1:
                for actor in actors:
                    self._pending_unrolls.append(self._collect_actor_unroll(actor))
                continue
            futures = [self._collector_executor.submit(self._collect_actor_unroll, actor) for actor in actors]
            for future in futures:
                self._pending_unrolls.append(future.result())

    def _start_process_collectors(self, model: PolicyValueModel) -> None:
        system_config = self.stack.config.system
        start_method = "spawn" if system_config is None else str(system_config.mp_start_method).strip()
        self._process_context = mp.get_context(start_method)
        self._collector_result_queue = self._process_context.Queue(maxsize=int(self.config.queue_capacity_unrolls))
        model_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        slot_configs: dict[int, dict[str, Any]] = {}
        if self._use_shared_collector_transport:
            hidden_size = int(getattr(model, "hidden_size", 1))
            for actor_id in range(int(self.config.actor_count)):
                slot_config = _create_shared_collector_slot_config(
                    actor_id=int(actor_id),
                    profile=str(self.config.profile),
                    unroll_length=int(self.config.unroll_length),
                    envs_per_actor=int(self.config.envs_per_actor),
                    observation_dim=int(self.observation_dim),
                    action_dim=int(self.action_dim),
                    hidden_size=hidden_size,
                    layout_name=("i16_legal_ids" if str(self.config.profile) == "fast" else "mask"),
                    legal_action_meta_width=int(self._action_meta_width),
                )
                self._collector_shared_slots[int(actor_id)] = _open_shared_collector_slot(slot_config, create=True)
                slot_configs[int(actor_id)] = slot_config
        for actor_id in range(int(self.config.actor_count)):
            control_queue = self._process_context.Queue()
            free_queue = None
            if self._use_shared_collector_transport:
                free_queue = self._process_context.Queue(maxsize=1)
                free_queue.put(1)
            process = self._process_context.Process(
                target=_collector_process_main,
                kwargs={
                    "stack": self.stack,
                    "config": self.config,
                    "model_state_dict": model_state_dict,
                    "observation_dim": int(self.observation_dim),
                    "action_dim": int(self.action_dim),
                    "observation_spec": self._observation_spec,
                    "spec_bundle": self._spec_bundle,
                    "actor_id": int(actor_id),
                    "control_queue": control_queue,
                    "free_queue": free_queue,
                    "result_queue": self._collector_result_queue,
                    "shared_slot_config": (None if not self._use_shared_collector_transport else slot_configs[int(actor_id)]),
                },
                daemon=True,
            )
            process.start()
            self._collector_control_queues.append(control_queue)
            if self._use_shared_collector_transport:
                self._collector_free_queues.append(free_queue)
            self._collector_processes.append(process)

    def _build_actor_state(self, *, model: PolicyValueModel, actor_id: int) -> _ActorState:
        env, layout_name = self._build_env(seed=_actor_seed(self.config.base_seed, actor_id))
        if self._shared_actor_model is not None:
            actor_model = self._shared_actor_model
            compiled_model = self._shared_compiled_actor_model
        else:
            actor_model = copy.deepcopy(model).to(self._device)
            actor_model.eval()
            compiled_model = _maybe_compile_runtime_actor_model(
                actor_model,
                enabled=self._compile_actor_inference,
            )
        current_batch = env.reset(seed=_actor_seed(self.config.base_seed, actor_id))
        state = _ActorState(
            actor_id=actor_id,
            env=env,
            model=actor_model,
            compiled_model=compiled_model,
            rng=np.random.default_rng(_actor_seed(self.config.base_seed, actor_id)),
            seat_hidden=actor_model.initial_seat_hidden(self.config.envs_per_actor, device=self._device),
            current_batch=current_batch,
            layout_name=layout_name,
            focal_seat_by_env=np.zeros((int(self.config.envs_per_actor),), dtype=np.int64),
            opponent_policy_id_by_env=np.full(
                (int(self.config.envs_per_actor),),
                _MIRROR_OPPONENT_POLICY_ID,
                dtype=object,
            ),
            opponent_hidden=actor_model.initial_seat_hidden(self.config.envs_per_actor, device=self._device),
            fixed_opponent_policy_id_by_env=self._fixed_opponent_policy_slots(),
        )
        self._assign_episode_roles(state, np.ones((int(self.config.envs_per_actor),), dtype=np.bool_), initial=True)
        return state

    def _build_env(self, *, seed: int) -> tuple[DecisionBoundaryEnv, str]:
        env_config = build_env_config_from_stack(self.stack, seed=int(seed))
        pool, layout_name = make_env_pool_from_config(
            env_config,
            profile=self.config.profile,  # type: ignore[arg-type]
            num_envs=int(self.config.envs_per_actor),
        )
        legality = "ids_offsets" if layout_name == "i16_legal_ids" else "mask"
        max_no_progress_decisions = None
        curriculum = self.stack.config.curriculum
        if curriculum is not None:
            raw_limit = curriculum.simulator.get("max_no_progress_decisions")
            if raw_limit is not None:
                max_no_progress_decisions = int(raw_limit)
        env = DecisionBoundaryEnv(
            pool,
            legality=legality,  # type: ignore[arg-type]
            pass_action_id=int(self.config.pass_action_id),
            engine_status_policy="hard_fail",
            # The training runtime copies each step into its own unroll buffers before
            # the next simulator call, so views are safe here and avoid a full extra
            # numpy allocation/copy per field on every reset/step.
            copy_arrays=False,
            max_decisions=int(env_config["max_decisions"]),
            max_ticks=int(env_config["max_ticks"]),
            max_no_progress_decisions=max_no_progress_decisions,
            profile_timers=self._profile_timers,
        )
        return env, str(layout_name)

    def _load_snapshot_model(self, snapshot_path: str) -> PolicyValueModel:
        if self._run_dir is None:
            raise RuntimeError("QueueRuntime cannot load opponent snapshots without a canonical run_dir")
        payload = torch.load(self._run_dir / snapshot_path, map_location="cpu", weights_only=True)
        model_state_dict = payload.get("model_state_dict")
        if not isinstance(model_state_dict, dict):
            raise RuntimeError(f"snapshot weights payload missing model_state_dict: {snapshot_path}")
        model_config = self.stack.config.model
        if model_config is None:
            raise RuntimeError("stack config is missing model config")
        model = build_policy_value_model(
            observation_dim=self.observation_dim,
            config=model_config,
            action_dim=self.action_dim,
            observation_spec=self._observation_spec,
            spec_bundle=self._spec_bundle,
        ).to(self._device)
        model.load_state_dict(model_state_dict)
        model.eval()
        return model

    def _assign_episode_roles(self, actor: _ActorState, done: np.ndarray, *, initial: bool = False) -> None:
        done_array = np.asarray(done, dtype=np.bool_)
        if done_array.shape != actor.focal_seat_by_env.shape:
            raise ValueError(f"done must have shape {actor.focal_seat_by_env.shape}, got {done_array.shape}")
        if not np.any(done_array):
            return
        if initial:
            actor.focal_seat_by_env[done_array] = (actor.actor_id + np.flatnonzero(done_array)) % 2
        else:
            actor.focal_seat_by_env[done_array] = 1 - actor.focal_seat_by_env[done_array]

        remaining_mask = done_array.copy()
        fixed_policy_ids = getattr(actor, "fixed_opponent_policy_id_by_env", None)
        fixed_heuristic_public_count = 0
        fixed_noleague_baseline_count = 0
        if fixed_policy_ids is not None:
            fixed_policy_ids = np.asarray(fixed_policy_ids, dtype=object)
            fixed_assign_mask = np.asarray(
                [
                    bool(done_flag) and bool(str(policy_id).strip()) and self._fixed_opponent_policy_is_active(str(policy_id))
                    for done_flag, policy_id in zip(done_array.tolist(), fixed_policy_ids.tolist())
                ],
                dtype=np.bool_,
            )
            if np.any(fixed_assign_mask):
                actor.opponent_policy_id_by_env[fixed_assign_mask] = fixed_policy_ids[fixed_assign_mask]
                remaining_mask = remaining_mask & ~fixed_assign_mask
                fixed_heuristic_public_count = int(
                    np.count_nonzero(fixed_policy_ids[fixed_assign_mask] == HEURISTIC_PUBLIC_POLICY_ID)
                )
                fixed_noleague_baseline_count = int(
                    np.count_nonzero(fixed_policy_ids[fixed_assign_mask] == _NOLEAGUE_BASELINE_POLICY_ID)
                )

        sampled_policy_ids = self._sample_opponent_policy_ids(count=int(np.count_nonzero(remaining_mask)), rng=actor.rng)
        actor.opponent_policy_id_by_env[remaining_mask] = np.asarray(sampled_policy_ids, dtype=object)
        if fixed_heuristic_public_count or fixed_noleague_baseline_count:
            self._pfsp_last_sampled_envs += fixed_heuristic_public_count + fixed_noleague_baseline_count
            self._pfsp_last_heuristic_public_envs += fixed_heuristic_public_count
            self._pfsp_last_noleague_baseline_envs += fixed_noleague_baseline_count

    def _sample_opponent_policy_ids(self, *, count: int, rng: np.random.Generator) -> tuple[str, ...]:
        if count <= 0:
            return ()
        if not self._league_enabled:
            self._pfsp_last_sampled_envs = 0
            self._pfsp_last_mirror_envs = count
            self._pfsp_last_heuristic_public_envs = 0
            self._pfsp_last_noleague_baseline_envs = 0
            self._pfsp_last_champion_envs = 0
            self._pfsp_last_recent_envs = 0
            self._pfsp_last_hard_negative_envs = 0
            return tuple(_MIRROR_OPPONENT_POLICY_ID for _ in range(count))
        assert self._league_config is not None
        sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
        pfsp_ready = self._pfsp_sampling_ready()
        groups: list[tuple[str, tuple[str, ...], float]] = []
        heuristic_public_start_updates = max(
            0,
            int(getattr(sampling_cfg, "heuristic_public_start_updates", 0)),
        )
        heuristic_public_weight = max(0.0, float(getattr(sampling_cfg, "heuristic_public_mix_fraction", 0.0)))
        champion_weight = max(0.0, float(getattr(sampling_cfg, "champion_mix_fraction", 0.35)))
        hard_negative_weight = max(0.0, float(getattr(sampling_cfg, "hard_negative_mix_fraction", 0.2)))
        recent_weight = max(0.0, 1.0 - heuristic_public_weight - champion_weight - hard_negative_weight)
        heuristic_policies = getattr(self, "_opponent_heuristic_policies", {})
        if (
            heuristic_public_weight > 0.0
            and self._league_reference_update() >= heuristic_public_start_updates
            and HEURISTIC_PUBLIC_POLICY_ID in heuristic_policies
        ):
            groups.append(("heuristic_public", (HEURISTIC_PUBLIC_POLICY_ID,), heuristic_public_weight))
        if pfsp_ready and self._opponent_hard_negative_ids:
            groups.append(("hard_negative", self._opponent_hard_negative_ids, hard_negative_weight))
        if pfsp_ready and self._opponent_champion_ids:
            groups.append(("champion", self._opponent_champion_ids, champion_weight))
        if pfsp_ready and self._opponent_recent_ids:
            groups.append(("recent", self._opponent_recent_ids, recent_weight))
        if not pfsp_ready:
            mirror_weight = max(0.0, 1.0 - heuristic_public_weight)
            groups.append(("mirror", (_MIRROR_OPPONENT_POLICY_ID,), mirror_weight))
        elif not groups:
            groups.append(("recent", self._opponent_candidate_ids, 1.0))
        weights = np.asarray([weight for _, _, weight in groups], dtype=np.float64)
        if not np.any(weights > 0):
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)
        sampled_group_indices = rng.choice(len(groups), size=count, replace=True, p=weights)
        sampled_policy_ids_list = [""] * count
        mirror_count = 0
        heuristic_public_count = 0
        noleague_baseline_count = 0
        hard_negative_count = 0
        champion_count = 0
        recent_count = 0
        for group_index, (group_name, group_ids, _) in enumerate(groups):
            positions = np.flatnonzero(sampled_group_indices == group_index)
            if positions.size == 0:
                continue
            if group_name in {"mirror", "heuristic_public"}:
                sampled_group_ids = tuple(group_ids[0] for _ in range(int(positions.size)))
            else:
                sampled_group_ids = sample_opponent_snapshot_ids(
                    group_ids,
                    count=int(positions.size),
                    rng=rng,
                    win_rates_by_snapshot_id={policy_id: self._outcomes.win_rate(policy_id) for policy_id in group_ids},
                    power=float(self._league_config.pfsp_power),
                    eps_uniform=float(self._league_config.pfsp_epsilon_uniform),
                )
            for idx, policy_id in zip(positions.tolist(), sampled_group_ids):
                sampled_policy_ids_list[int(idx)] = str(policy_id)
            if group_name == "mirror":
                mirror_count += int(positions.size)
            elif group_name == "heuristic_public":
                heuristic_public_count += int(positions.size)
            elif group_name == "hard_negative":
                hard_negative_count += int(positions.size)
            elif group_name == "champion":
                champion_count += int(positions.size)
            else:
                recent_count += int(positions.size)
        self._pfsp_last_sampled_envs = count - mirror_count
        self._pfsp_last_mirror_envs = mirror_count
        self._pfsp_last_heuristic_public_envs = heuristic_public_count
        self._pfsp_last_noleague_baseline_envs = noleague_baseline_count
        self._pfsp_last_hard_negative_envs = hard_negative_count
        self._pfsp_last_champion_envs = champion_count
        self._pfsp_last_recent_envs = recent_count
        return tuple(str(policy_id) for policy_id in sampled_policy_ids_list)

    def _pfsp_sampling_ready(self) -> bool:
        if not self._league_enabled or self._league_config is None or self._opponent_sampler is None:
            return False
        if self._league_reference_update() < int(self._league_config.warmup.first_updates):
            return False
        return bool(self._opponent_candidate_ids) and bool(self._opponent_models)

    def _heuristic_public_actions_from_ids(
        self,
        *,
        actor: _ActorState | None,
        heuristic_policy: HeuristicPublicPolicy,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None = None,
        counters: dict[str, int] | None = None,
    ) -> np.ndarray:
        actions = np.zeros((int(row_indices.shape[0]),), dtype=np.int64)
        row_indices_array = np.asarray(row_indices, dtype=np.int64)
        if counters is not None:
            counters["tactical_row_count"] += int(row_indices_array.shape[0])
            counters["fixed_opponent_tactical_row_count"] += int(row_indices_array.shape[0])
        if actor is not None and str(getattr(self, "_fixed_opponent_backend", "python_batched")) == "simulator_native":
            if counters is not None:
                candidate_counts = np.maximum(
                    np.asarray(legal_offsets[row_indices_array + 1], dtype=np.int64)
                    - np.asarray(legal_offsets[row_indices_array], dtype=np.int64),
                    0,
                )
                counters["packed_candidate_count"] += int(candidate_counts.sum())
            return self._heuristic_public_actions_from_pool(
                actor=actor,
                row_indices=row_indices_array,
            )
        batch_choose = getattr(heuristic_policy, "choose_actions_from_meta_batch", None)
        if callable(batch_choose):
            if legal_action_meta is None:
                subset_ids, subset_offsets = _slice_packed_rows(
                    legal_ids,
                    legal_offsets,
                    row_indices_array,
                )
                subset_meta = None
            else:
                subset_ids, subset_offsets, subset_meta = _slice_packed_rows_with_meta(
                    legal_ids,
                    legal_offsets,
                    row_indices_array,
                    legal_action_meta=legal_action_meta,
                )
            if counters is not None:
                counters["packed_candidate_count"] += int(subset_ids.shape[0])
            return np.asarray(
                batch_choose(
                    np.asarray(obs_step[row_indices_array], dtype=np.int32),
                    subset_ids,
                    subset_offsets,
                    subset_meta,
                ),
                dtype=np.int64,
            )
        for offset, row_index in enumerate(row_indices_array):
            start = int(legal_offsets[int(row_index)])
            stop = int(legal_offsets[int(row_index) + 1])
            if counters is not None:
                counters["packed_candidate_count"] += max(0, stop - start)
            actions[offset] = int(
                heuristic_policy.choose_action_from_meta(
                    np.asarray(obs_step[int(row_index)]),
                    np.asarray(legal_ids[start:stop], dtype=np.uint32),
                    (
                        None
                        if legal_action_meta is None
                        else np.asarray(legal_action_meta[start:stop], dtype=np.uint16)
                    ),
                )
            )
        return actions

    def _heuristic_public_actions_from_mask(
        self,
        *,
        actor: _ActorState | None,
        heuristic_policy: HeuristicPublicPolicy,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        legal_mask: np.ndarray,
        counters: dict[str, int] | None = None,
    ) -> np.ndarray:
        actions = np.zeros((int(row_indices.shape[0]),), dtype=np.int64)
        row_indices_array = np.asarray(row_indices, dtype=np.int64)
        if counters is not None:
            counters["tactical_row_count"] += int(row_indices_array.shape[0])
            counters["fixed_opponent_tactical_row_count"] += int(row_indices_array.shape[0])
        if actor is not None and str(getattr(self, "_fixed_opponent_backend", "python_batched")) == "simulator_native":
            if counters is not None:
                counters["packed_candidate_count"] += int(
                    np.count_nonzero(np.asarray(legal_mask[row_indices_array], dtype=np.bool_))
                )
            return self._heuristic_public_actions_from_pool(
                actor=actor,
                row_indices=row_indices_array,
            )
        for offset, row_index in enumerate(row_indices_array.tolist()):
            legal_ids = np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(np.uint32, copy=False)
            if counters is not None:
                counters["packed_candidate_count"] += int(legal_ids.shape[0])
            actions[offset] = int(heuristic_policy.choose_action(np.asarray(obs_step[int(row_index)]), legal_ids))
        return actions

    def _heuristic_public_actions_from_pool(
        self,
        *,
        actor: _ActorState | None,
        row_indices: np.ndarray,
    ) -> np.ndarray:
        if actor is None:
            raise RuntimeError("simulator_native fixed-opponent routing requires actor context")
        choose_into = getattr(getattr(actor, "env", None), "pool", None)
        choose_into = getattr(choose_into, "choose_heuristic_public_actions_into", None)
        if not callable(choose_into):
            raise RuntimeError(
                "training.fixed_opponent_backend=simulator_native requires "
                "pool.choose_heuristic_public_actions_into(...)"
            )
        env_indices = np.asarray(row_indices, dtype=np.uint32)
        chosen_actions = np.zeros((int(env_indices.shape[0]),), dtype=np.uint16)
        choose_into(env_indices, chosen_actions)
        return chosen_actions.astype(np.int64, copy=False)

    def _teacher_label_arrays(self, num_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (int(num_rows),)
        return (
            np.full(shape, -1, dtype=np.int32),
            np.full(shape, -1, dtype=np.int32),
            np.full(shape, -1, dtype=np.int32),
            np.zeros(shape, dtype=np.bool_),
        )

    def _teacher_labels_from_actions(
        self,
        *,
        row_indices: np.ndarray,
        chosen_actions: np.ndarray,
        num_rows: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        teacher_family, teacher_slot, teacher_attack_type, teacher_valid = self._teacher_label_arrays(num_rows)
        if (
            not bool(getattr(self, "_teacher_guidance_enabled", False))
            or self._teacher_action_catalog is None
        ):
            return teacher_family, teacher_slot, teacher_attack_type, teacher_valid
        for row_index, action_id in zip(
            np.asarray(row_indices, dtype=np.int64).tolist(),
            np.asarray(chosen_actions, dtype=np.int64).tolist(),
            strict=True,
        ):
            decoded = self._teacher_action_catalog.decode(int(action_id))
            family_index = self._teacher_family_index.get(decoded.family)
            if family_index is None:
                continue
            teacher_valid[int(row_index)] = True
            teacher_family[int(row_index)] = int(family_index)
            if decoded.family == "main_play_character" and decoded.stage_slot is not None:
                teacher_slot[int(row_index)] = int(decoded.stage_slot)
            elif decoded.family == "attack":
                if decoded.slot is not None:
                    teacher_slot[int(row_index)] = int(decoded.slot)
                if decoded.attack_type is not None:
                    attack_type_index = self._teacher_attack_type_index.get(decoded.attack_type)
                    if attack_type_index is not None:
                        teacher_attack_type[int(row_index)] = int(attack_type_index)
        return teacher_family, teacher_slot, teacher_attack_type, teacher_valid

    def _teacher_labels_from_ids(
        self,
        *,
        focal_rows: np.ndarray,
        decision_kind: np.ndarray,
        obs_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None = None,
        counters: dict[str, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        teacher_family, teacher_slot, teacher_attack_type, teacher_valid = self._teacher_label_arrays(
            int(decision_kind.shape[0])
        )
        if not bool(getattr(self, "_teacher_guidance_enabled", False)) or self._teacher_policy is None:
            return teacher_family, teacher_slot, teacher_attack_type, teacher_valid
        decision_kind_array = np.asarray(decision_kind, dtype=np.int32)
        tactical_rows = np.flatnonzero(
            np.asarray(focal_rows, dtype=np.bool_)
            & np.isin(decision_kind_array, tuple(_TACTICAL_TEACHER_DECISION_KINDS))
        )
        if tactical_rows.size == 0:
            return teacher_family, teacher_slot, teacher_attack_type, teacher_valid
        if counters is not None:
            counters["teacher_tactical_row_count"] += int(tactical_rows.size)
        chosen_actions = self._heuristic_public_actions_from_ids(
            actor=None,
            heuristic_policy=self._teacher_policy,
            row_indices=tactical_rows,
            obs_step=obs_step,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=legal_action_meta,
            counters=counters,
        )
        return self._teacher_labels_from_actions(
            row_indices=tactical_rows,
            chosen_actions=chosen_actions,
            num_rows=int(decision_kind.shape[0]),
        )

    def _teacher_labels_from_mask(
        self,
        *,
        focal_rows: np.ndarray,
        decision_kind: np.ndarray,
        obs_step: np.ndarray,
        legal_mask: np.ndarray,
        counters: dict[str, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        teacher_family, teacher_slot, teacher_attack_type, teacher_valid = self._teacher_label_arrays(
            int(decision_kind.shape[0])
        )
        if not bool(getattr(self, "_teacher_guidance_enabled", False)) or self._teacher_policy is None:
            return teacher_family, teacher_slot, teacher_attack_type, teacher_valid
        decision_kind_array = np.asarray(decision_kind, dtype=np.int32)
        tactical_rows = np.flatnonzero(
            np.asarray(focal_rows, dtype=np.bool_)
            & np.isin(decision_kind_array, tuple(_TACTICAL_TEACHER_DECISION_KINDS))
        )
        if tactical_rows.size == 0:
            return teacher_family, teacher_slot, teacher_attack_type, teacher_valid
        if counters is not None:
            counters["teacher_tactical_row_count"] += int(tactical_rows.size)
        chosen_actions = self._heuristic_public_actions_from_mask(
            actor=None,
            heuristic_policy=self._teacher_policy,
            row_indices=tactical_rows,
            obs_step=obs_step,
            legal_mask=legal_mask,
            counters=counters,
        )
        return self._teacher_labels_from_actions(
            row_indices=tactical_rows,
            chosen_actions=chosen_actions,
            num_rows=int(decision_kind.shape[0]),
        )

    def _write_deterministic_logits(
        self,
        *,
        logits_out: np.ndarray | None,
        row_indices: np.ndarray,
        chosen_actions: np.ndarray,
        legal_action_ids: Sequence[np.ndarray],
    ) -> None:
        if logits_out is None:
            return
        for row_index, chosen_action, legal_ids in zip(
            row_indices.tolist(),
            np.asarray(chosen_actions, dtype=np.int64).tolist(),
            legal_action_ids,
            strict=True,
        ):
            row_logits = np.full((self.action_dim,), -1.0e9, dtype=np.float32)
            legal_ids_np = np.asarray(legal_ids, dtype=np.int64)
            if legal_ids_np.size:
                row_logits[legal_ids_np] = -100.0
            row_logits[int(chosen_action)] = 0.0
            logits_out[int(row_index)] = row_logits

    def _write_deterministic_logits_from_packed(
        self,
        *,
        logits_out: np.ndarray | None,
        row_indices: np.ndarray,
        chosen_actions: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> None:
        if logits_out is None:
            return
        row_indices_array = np.asarray(row_indices, dtype=np.int64)
        chosen_actions_array = np.asarray(chosen_actions, dtype=np.int64)
        for row_index, chosen_action in zip(row_indices_array, chosen_actions_array, strict=True):
            row_logits = logits_out[int(row_index)]
            row_logits.fill(-1.0e9)
            start = int(legal_offsets[int(row_index)])
            stop = int(legal_offsets[int(row_index) + 1])
            if stop > start:
                row_logits[np.asarray(legal_ids[start:stop], dtype=np.int64)] = -100.0
            row_logits[int(chosen_action)] = 0.0

    def _fill_policy_outputs_mask(
        self,
        *,
        actor: _ActorState,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        focal_rows: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
    ) -> None:
        focal_indices = np.flatnonzero(focal_rows)
        if focal_indices.size:
            self._apply_policy_rows_mask(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
            )
        opponent_indices = np.flatnonzero(~focal_rows)
        if opponent_indices.size:
            self._apply_opponent_rows_mask(
                actor=actor,
                row_indices=opponent_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
            )

    def _fill_policy_outputs_ids(
        self,
        *,
        actor: _ActorState,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        focal_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
    ) -> None:
        focal_indices = np.flatnonzero(focal_rows)
        if focal_indices.size:
            self._apply_policy_rows_ids(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
            )
        opponent_indices = np.flatnonzero(~focal_rows)
        if opponent_indices.size:
            self._apply_opponent_rows_ids(
                actor=actor,
                row_indices=opponent_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
            )

    def _apply_opponent_rows_mask(
        self,
        *,
        actor: _ActorState,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
    ) -> None:
        for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in row_indices.tolist()}):
            policy_rows = row_indices[actor.opponent_policy_id_by_env[row_indices] == policy_id]
            if not policy_rows.size:
                continue
            if policy_id == _MIRROR_OPPONENT_POLICY_ID:
                self._apply_policy_rows_mask(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_mask=legal_mask,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                )
                continue
            heuristic_policy = getattr(self, "_opponent_heuristic_policies", {}).get(policy_id)
            if heuristic_policy is not None:
                self._advance_hidden_only(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                )
                chosen_actions = self._heuristic_public_actions_from_mask(
                    actor=actor,
                    heuristic_policy=heuristic_policy,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    legal_mask=legal_mask,
                )
                legal_action_ids = [
                    np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(np.uint32, copy=False)
                    for row_index in policy_rows.tolist()
                ]
                self._write_deterministic_logits(
                    logits_out=logits_out,
                    row_indices=policy_rows,
                    chosen_actions=chosen_actions,
                    legal_action_ids=legal_action_ids,
                )
                values_out[policy_rows] = 0.0
                if sample_actions:
                    assert actions_out is not None and logp_out is not None
                    actions_out[policy_rows] = chosen_actions
                    logp_out[policy_rows] = 0.0
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            self._advance_hidden_only(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
            )
            with self._opponent_model_locks[policy_id]:
                self._apply_policy_rows_mask(
                    model=model,
                    hidden_state=actor.opponent_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_mask=legal_mask,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                )

    def _apply_opponent_rows_ids(
        self,
        *,
        actor: _ActorState,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
        heuristic_rows_hidden_already_advanced: bool = False,
    ) -> None:
        for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in row_indices.tolist()}):
            policy_rows = row_indices[actor.opponent_policy_id_by_env[row_indices] == policy_id]
            if not policy_rows.size:
                continue
            if policy_id == _MIRROR_OPPONENT_POLICY_ID:
                self._apply_policy_rows_ids(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                )
                continue
            heuristic_policy = getattr(self, "_opponent_heuristic_policies", {}).get(policy_id)
            if heuristic_policy is not None:
                if not bool(heuristic_rows_hidden_already_advanced):
                    self._advance_hidden_only(
                        model=_actor_inference_model(actor),
                        hidden_state=actor.seat_hidden,
                        row_indices=policy_rows,
                        obs_step=obs_step,
                        actor_step=actor_step,
                    )
                chosen_actions = self._heuristic_public_actions_from_ids(
                    actor=actor,
                    heuristic_policy=heuristic_policy,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                )
                self._write_deterministic_logits_from_packed(
                    logits_out=logits_out,
                    row_indices=policy_rows,
                    chosen_actions=chosen_actions,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                )
                values_out[policy_rows] = 0.0
                if sample_actions:
                    assert actions_out is not None and logp_out is not None
                    actions_out[policy_rows] = chosen_actions
                    logp_out[policy_rows] = 0.0
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            self._advance_hidden_only(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
            )
            with self._opponent_model_locks[policy_id]:
                self._apply_policy_rows_ids(
                    model=model,
                    hidden_state=actor.opponent_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                )

    def _apply_policy_rows_mask(
        self,
        *,
        model: Any,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
    ) -> None:
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            legal_actions = (
                _structured_legal_batch_from_mask(legal_mask, row_indices)
                if bool(getattr(model, "supports_legal_candidate_scoring", False))
                else None
            )
            logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                torch.as_tensor(obs_step[row_indices], device=self._device),
                torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                hidden_state[row_indices],
                legal_actions=legal_actions,
            )
        hidden_state[row_indices] = torch.as_tensor(
            next_hidden,
            device=self._device,
            dtype=hidden_state.dtype,
        ).clone()
        logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        value_subset = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if logits_out is not None:
            logits_out[row_indices] = logits_subset
        values_out[row_indices] = value_subset
        if sample_actions:
            assert actions_out is not None and logp_out is not None
            action_subset, logp_subset, _entropy = sample_actions_from_mask(
                logits_subset,
                legal_mask[row_indices],
                rng=rng,
                pass_action_id=self.config.pass_action_id,
            )
            actions_out[row_indices] = action_subset
            logp_out[row_indices] = logp_subset

    def _apply_policy_rows_ids(
        self,
        *,
        model: Any,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
    ) -> None:
        legal_actions = (
            _structured_legal_batch_from_packed(
                legal_ids,
                legal_offsets,
                row_indices,
                legal_action_meta,
            )
            if bool(getattr(model, "supports_legal_candidate_scoring", False))
            else None
        )
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            if (
                legal_actions is not None
                and sample_actions
                and logits_out is None
                and hasattr(model, "sample_packed_seat_aware")
            ):
                action_tensor, logp_tensor, value_tensor, next_hidden = model.sample_packed_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                    legal_actions=legal_actions,
                    sample_seeds=torch.as_tensor(
                        rng.integers(0, np.iinfo(np.int64).max, size=row_indices.shape[0], dtype=np.int64),
                        device=self._device,
                        dtype=torch.long,
                    ),
                    pass_action_id=int(self.config.pass_action_id),
                )
                logits_subset = None
            elif legal_actions is None:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                )
                logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                    legal_actions=legal_actions,
                )
                logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        hidden_state[row_indices] = torch.as_tensor(
            next_hidden,
            device=self._device,
            dtype=hidden_state.dtype,
        ).clone()
        value_subset = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if logits_out is not None:
            assert logits_subset is not None
            logits_out[row_indices] = logits_subset
        values_out[row_indices] = value_subset
        if sample_actions:
            assert actions_out is not None and logp_out is not None
            if logits_subset is None:
                action_subset = action_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
                logp_subset = logp_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                subset_ids, subset_offsets = _slice_packed_rows(legal_ids, legal_offsets, row_indices)
                action_subset, logp_subset, _entropy = sample_actions_from_legal_ids(
                    logits_subset,
                    subset_ids,
                    subset_offsets,
                    rng=rng,
                    pass_action_id=self.config.pass_action_id,
                )
            actions_out[row_indices] = action_subset
            logp_out[row_indices] = logp_subset

    def _update_outcomes(
        self,
        *,
        actor: _ActorState,
        acting_seat: np.ndarray,
        terminal_batch: DecisionBoundaryBatch,
        done: np.ndarray,
    ) -> None:
        if not np.any(done):
            return
        for env_index in np.flatnonzero(done):
            opponent_policy_id = str(actor.opponent_policy_id_by_env[env_index])
            if opponent_policy_id == _MIRROR_OPPONENT_POLICY_ID:
                continue
            result = game_result_from_step(
                terminal_batch,
                env_index=int(env_index),
                acting_seat=int(acting_seat[int(env_index)]),
            )
            focal_seat = int(actor.focal_seat_by_env[int(env_index)])
            if result.truncated:
                outcome = "t"
            elif result.winner_seat is None:
                outcome = "d"
            elif int(result.winner_seat) == focal_seat:
                outcome = "w"
            else:
                outcome = "l"
            self._outcomes.update(opponent_policy_id, outcome)

    def _advance_hidden_only(
        self,
        *,
        model: Any,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
    ) -> None:
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            advance_hidden = getattr(model, "advance_seat_hidden", None)
            if callable(advance_hidden):
                next_hidden = advance_hidden(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                )
            else:
                _logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                )
        hidden_state[row_indices] = torch.as_tensor(
            next_hidden,
            device=self._device,
            dtype=hidden_state.dtype,
        ).clone()

    def _central_sample_policy_rows_ids(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
        actions_outs: Sequence[np.ndarray],
        logp_outs: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[int, _ActorState, np.ndarray]] = []
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets = [np.array([0], dtype=np.uint32)]
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        seed_parts: list[np.ndarray] = []
        model = _actor_inference_model(actors[0])
        pack_started = time.perf_counter()
        for actor_index, (actor, batch, obs_step, actor_step, row_indices) in enumerate(
            zip(
            actors,
            batches,
            obs_steps,
            actor_steps,
            row_indices_by_actor,
            strict=True,
            )
        ):
            if row_indices.size == 0:
                continue
            legal_ids, legal_offsets = _require_ids_offsets(batch)
            legal_action_meta = _optional_legal_action_meta(batch)
            subset_ids, subset_offsets, subset_meta = _slice_packed_rows_with_meta(
                legal_ids,
                legal_offsets,
                row_indices,
                legal_action_meta=legal_action_meta,
            )
            offset_base = int(packed_offsets[-1][-1])
            packed_ids.append(subset_ids)
            packed_offsets.append(np.asarray(subset_offsets[1:] + offset_base, dtype=np.uint32))
            if subset_meta is not None:
                packed_meta.append(subset_meta)
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            seed_parts.append(actor.rng.integers(0, np.iinfo(np.int64).max, size=row_indices.shape[0], dtype=np.int64))
            entries.append((actor_index, actor, row_indices))
        if not entries:
            return
        legal_actions = LegalActionBatch.from_packed(
            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
            np.concatenate(packed_offsets, axis=0),
            meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
            action_space=int(self.action_dim),
        )
        hidden_concat = torch.cat(hidden_parts, dim=0)
        self._record_batch_timer_ms("central_focal_policy_pack", time.perf_counter() - pack_started)
        model_started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            actions_tensor, logp_tensor, value_tensor, next_hidden = model.sample_packed_seat_aware(
                torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                hidden_concat,
                legal_actions=legal_actions,
                sample_seeds=torch.as_tensor(np.concatenate(seed_parts, axis=0), device=self._device, dtype=torch.long),
                pass_action_id=int(self.config.pass_action_id),
            )
        self._record_batch_timer_ms("central_focal_policy_model", time.perf_counter() - model_started)
        actions_concat = actions_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
        logp_concat = logp_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        scatter_started = time.perf_counter()
        offset = 0
        for actor_index, actor, row_indices in entries:
            count = int(row_indices.shape[0])
            actor.seat_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
            values_outs[actor_index][row_indices] = values_concat[offset : offset + count]
            actions_outs[actor_index][row_indices] = actions_concat[offset : offset + count]
            logp_outs[actor_index][row_indices] = logp_concat[offset : offset + count]
            offset += count
        self._record_batch_timer_ms("central_focal_policy_scatter", time.perf_counter() - scatter_started)

    def _central_value_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[int, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = _actor_inference_model(actors[0])
        for actor_index, (actor, obs_step, actor_step, row_indices) in enumerate(
            zip(actors, obs_steps, actor_steps, row_indices_by_actor, strict=True)
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor_index, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            value_seat_aware = getattr(model, "value_seat_aware", None)
            if callable(value_seat_aware):
                value_tensor = value_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
            else:
                _logits_tensor, value_tensor, _next_hidden = model.forward_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        offset = 0
        for actor_index, row_indices in entries:
            count = int(row_indices.shape[0])
            values_outs[actor_index][row_indices] = values_concat[offset : offset + count]
            offset += count

    def _central_advance_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[_ActorState, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = _actor_inference_model(actors[0])
        for actor, obs_step, actor_step, row_indices in zip(
            actors,
            obs_steps,
            actor_steps,
            row_indices_by_actor,
            strict=True,
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            advance_hidden = getattr(model, "advance_seat_hidden", None)
            if callable(advance_hidden):
                next_hidden = advance_hidden(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
            else:
                _logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor, row_indices in entries:
            count = int(row_indices.shape[0])
            actor.seat_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
            offset += count

    def _central_forward_all_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch] | None,
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        if not actors:
            return
        obs_concat = np.concatenate(obs_steps, axis=0)
        actor_concat = np.concatenate(actor_steps, axis=0)
        hidden_concat = torch.cat([actor.seat_hidden for actor in actors], dim=0)
        model = _actor_inference_model(actors[0])
        legal_actions = None
        if bool(getattr(model, "supports_legal_candidate_scoring", False)) and batches is not None:
            legal_actions = _concatenate_batch_legal_actions(batches, action_space=int(self.action_dim))
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                torch.as_tensor(obs_concat, device=self._device),
                torch.as_tensor(actor_concat, device=self._device, dtype=torch.long),
                hidden_concat,
                legal_actions=legal_actions,
            )
        logits_concat = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor, logits_out, values_out in zip(actors, logits_outs, values_outs, strict=True):
            count = int(logits_out.shape[0])
            logits_out[...] = logits_concat[offset : offset + count]
            values_out[...] = values_concat[offset : offset + count]
            actor.seat_hidden[...] = next_hidden_tensor[offset : offset + count]
            offset += count

    def _overwrite_central_outputs_with_opponents(
        self,
        *,
        actor: _ActorState,
        batch: DecisionBoundaryBatch,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
    ) -> None:
        self._overwrite_central_outputs_with_configured_opponents(
            actors=[actor],
            batches=[batch],
            obs_steps=[obs_step],
            actor_steps=[actor_step],
            logits_outs=[logits_out],
            values_outs=[values_out],
        )

    def _overwrite_central_outputs_with_batched_opponents(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray | None],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        overwrite_started = time.perf_counter()
        policy_groups: dict[
            str,
            list[tuple[_ActorState, DecisionBoundaryBatch, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]],
        ] = {}
        for actor, batch, obs_step, actor_step, logits_out, values_out in zip(
            actors,
            batches,
            obs_steps,
            actor_steps,
            logits_outs,
            values_outs,
            strict=True,
        ):
            focal_rows = actor_step == actor.focal_seat_by_env
            opponent_indices = np.flatnonzero(~focal_rows)
            if opponent_indices.size == 0:
                continue
            for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in opponent_indices.tolist()}):
                if policy_id == _MIRROR_OPPONENT_POLICY_ID:
                    continue
                policy_rows = opponent_indices[actor.opponent_policy_id_by_env[opponent_indices] == policy_id]
                if not policy_rows.size:
                    continue
                policy_groups.setdefault(policy_id, []).append(
                    (actor, batch, policy_rows, obs_step, actor_step, logits_out, values_out)
                )

        for policy_id, entries in sorted(policy_groups.items()):
            heuristic_policy = getattr(self, "_opponent_heuristic_policies", {}).get(policy_id)
            if heuristic_policy is not None:
                self._central_advance_actor_rows(
                    actors=[actor for actor, *_rest in entries],
                    obs_steps=[obs_step for _actor, _batch, _row_indices, obs_step, _actor_step, _logits_out, _values_out in entries],
                    actor_steps=[actor_step for _actor, _batch, _row_indices, _obs_step, actor_step, _logits_out, _values_out in entries],
                    row_indices_by_actor=[row_indices for _actor, _batch, row_indices, _obs_step, _actor_step, _logits_out, _values_out in entries],
                )
                packed_entries = [
                    entry for entry in entries if entry[1].ids_offsets is not None
                ]
                mask_entries = [
                    entry for entry in entries if entry[1].ids_offsets is None
                ]
                if packed_entries:
                    if str(getattr(self, "_fixed_opponent_backend", "python_batched")) == "simulator_native":
                        for actor, batch, row_indices, obs_step, _actor_step, logits_out, values_out in packed_entries:
                            legal_ids, legal_offsets = _require_ids_offsets(batch)
                            chosen_actions = self._heuristic_public_actions_from_ids(
                                actor=actor,
                                heuristic_policy=heuristic_policy,
                                row_indices=row_indices,
                                obs_step=obs_step,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                                legal_action_meta=_optional_legal_action_meta(batch),
                            )
                            self._write_deterministic_logits_from_packed(
                                logits_out=logits_out,
                                row_indices=row_indices,
                                chosen_actions=chosen_actions,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                            )
                            values_out[row_indices] = 0.0
                    else:
                        packed_obs_parts: list[np.ndarray] = []
                        packed_ids: list[np.ndarray] = []
                        packed_meta: list[np.ndarray] = []
                        packed_offsets = [np.array([0], dtype=np.uint32)]
                        packed_entry_counts: list[int] = []
                        for actor, batch, row_indices, obs_step, _actor_step, _logits_out, _values_out in packed_entries:
                            legal_ids, legal_offsets = _require_ids_offsets(batch)
                            subset_ids, subset_offsets, subset_meta = _slice_packed_rows_with_meta(
                                legal_ids,
                                legal_offsets,
                                row_indices,
                                legal_action_meta=_optional_legal_action_meta(batch),
                            )
                            offset_base = int(packed_offsets[-1][-1])
                            packed_ids.append(subset_ids)
                            packed_offsets.append(np.asarray(subset_offsets[1:] + offset_base, dtype=np.uint32))
                            if subset_meta is not None:
                                packed_meta.append(subset_meta)
                            packed_obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.int32))
                            packed_entry_counts.append(int(row_indices.shape[0]))
                        packed_chosen_actions = heuristic_policy.choose_actions_from_meta_batch(
                            np.concatenate(packed_obs_parts, axis=0)
                            if packed_obs_parts
                            else np.zeros((0, 0), dtype=np.int32),
                            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                            np.concatenate(packed_offsets, axis=0),
                            np.concatenate(packed_meta, axis=0) if packed_meta else None,
                        )
                        offset = 0
                        for (actor, batch, row_indices, _obs_step, _actor_step, logits_out, values_out), count in zip(
                            packed_entries,
                            packed_entry_counts,
                            strict=True,
                        ):
                            legal_ids, legal_offsets = _require_ids_offsets(batch)
                            chosen_actions = np.asarray(
                                packed_chosen_actions[offset : offset + count],
                                dtype=np.int64,
                            )
                            self._write_deterministic_logits_from_packed(
                                logits_out=logits_out,
                                row_indices=row_indices,
                                chosen_actions=chosen_actions,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                            )
                            values_out[row_indices] = 0.0
                            offset += count
                for actor, batch, row_indices, obs_step, _actor_step, logits_out, values_out in mask_entries:
                    legal_mask = _require_mask(batch)
                    chosen_actions = self._heuristic_public_actions_from_mask(
                        actor=actor,
                        heuristic_policy=heuristic_policy,
                        row_indices=row_indices,
                        obs_step=obs_step,
                        legal_mask=legal_mask,
                    )
                    legal_action_ids = [
                        np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(
                            np.uint32,
                            copy=False,
                        )
                        for row_index in row_indices.tolist()
                    ]
                    self._write_deterministic_logits(
                        logits_out=logits_out,
                        row_indices=row_indices,
                        chosen_actions=chosen_actions,
                        legal_action_ids=legal_action_ids,
                    )
                    values_out[row_indices] = 0.0
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            obs_concat = np.concatenate(
                [obs_step[row_indices] for _, _, row_indices, obs_step, _, _, _ in entries],
                axis=0,
            )
            actor_concat = np.concatenate(
                [actor_step[row_indices] for _, _, row_indices, _, actor_step, _, _ in entries],
                axis=0,
            )
            hidden_concat = torch.cat(
                [actor.opponent_hidden[row_indices] for actor, _, row_indices, _, _, _, _ in entries],
                dim=0,
            )
            with self._opponent_model_locks[policy_id]:
                with torch.inference_mode(), torch.amp.autocast(
                    device_type=self._device.type,
                    enabled=self._actor_amp_enabled,
                ):
                    logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                        torch.as_tensor(obs_concat, device=self._device),
                        torch.as_tensor(actor_concat, device=self._device, dtype=torch.long),
                        hidden_concat,
                    )
            logits_concat = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
            offset = 0
            for actor, _batch, row_indices, _obs_step, _actor_step, logits_out, values_out in entries:
                count = int(row_indices.shape[0])
                actor.opponent_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
                values_out[row_indices] = values_concat[offset : offset + count]
                if logits_out is not None:
                    logits_out[row_indices] = logits_concat[offset : offset + count]
                offset += count
        self._record_batch_timer_ms("central_fixed_opponent_overwrite", time.perf_counter() - overwrite_started)

    def _collect_actor_unrolls_central(self, actors: Sequence[_ActorState]) -> list[RuntimeUnroll]:
        central_started = time.perf_counter()
        if not actors:
            return []
        if len({str(actor.layout_name) for actor in actors}) != 1:
            return [self._collect_actor_unroll(actor) for actor in actors]

        T = int(self.config.unroll_length)
        N = int(self.config.envs_per_actor)
        obs_dtype = np.asarray(actors[0].current_batch.obs).dtype
        state_by_actor: dict[int, dict[str, Any]] = {}
        for actor in actors:
            state_by_actor[int(actor.actor_id)] = {
                "obs": np.zeros((T, N, self.observation_dim), dtype=obs_dtype),
                "actions": np.zeros((T, N), dtype=np.uint16),
                "rewards": np.zeros((T, N), dtype=np.float32),
                "terminated": np.zeros((T, N), dtype=np.bool_),
                "truncated": np.zeros((T, N), dtype=np.bool_),
                "to_play_seat": np.zeros((T, N), dtype=np.int8),
                "behavior_logp": np.zeros((T, N), dtype=np.float32),
                "values": np.zeros((T, N), dtype=np.float32),
                "episode_seed": np.zeros((T, N), dtype=np.uint64),
                "policy_train_mask": np.zeros((T, N), dtype=np.bool_),
                "teacher_family": np.full((T, N), -1, dtype=np.int32),
                "teacher_slot": np.full((T, N), -1, dtype=np.int32),
                "teacher_attack_type": np.full((T, N), -1, dtype=np.int32),
                "teacher_valid": np.zeros((T, N), dtype=np.bool_),
                "packed_ids": [],
                "packed_meta": [],
                "packed_offsets": [np.array([0], dtype=np.uint32)],
                "mask_steps": [],
                "initial_hidden_state": actor.seat_hidden.detach().cpu().numpy().copy(),
                "counters": _collector_counter_template(),
                "action_sequence_state": make_action_sequence_state(N),
            }
        timeout_limits_by_actor = {int(actor.actor_id): _timeout_limits_for_env(actor.env) for actor in actors}

        batches = [actor.current_batch for actor in actors]
        structured_central_packed = bool(
            all(actor.layout_name == "i16_legal_ids" for actor in actors)
            and bool(getattr(_actor_inference_model(actors[0]), "supports_legal_candidate_scoring", False))
        )
        for step_index in range(T):
            obs_storage_steps = [np.asarray(batch.obs) for batch in batches]
            obs_steps = [np.asarray(batch.obs, dtype=np.float32) for batch in batches]
            actor_steps = [np.asarray(batch.actor, dtype=np.int64) for batch in batches]
            if structured_central_packed:
                action_steps = [np.zeros((N,), dtype=np.int64) for _ in actors]
                logp_steps = [np.zeros((N,), dtype=np.float32) for _ in actors]
                value_steps = [np.zeros((N,), dtype=np.float32) for _ in actors]
                policy_row_indices = [
                    np.flatnonzero(actor_step == actor.focal_seat_by_env)
                    for actor, actor_step in zip(actors, actor_steps, strict=True)
                ]
                for actor, row_indices in zip(actors, policy_row_indices, strict=True):
                    state_by_actor[int(actor.actor_id)]["counters"]["focal_row_count"] += int(row_indices.shape[0])
                fuse_mirror_policy_rows = not bool(getattr(self, "_disable_mirror_policy_fusion", False))
                heuristic_policy_ids = tuple(getattr(self, "_opponent_heuristic_policies", {}).keys())
                heuristic_rows_by_actor: list[np.ndarray] = []
                mirror_rows_by_actor: list[np.ndarray] = []
                residual_rows_by_actor: list[np.ndarray] = []
                for actor, actor_step in zip(actors, actor_steps, strict=True):
                    opponent_indices = np.flatnonzero(actor_step != actor.focal_seat_by_env)
                    if opponent_indices.size == 0:
                        heuristic_rows_by_actor.append(np.zeros((0,), dtype=np.int64))
                        mirror_rows_by_actor.append(np.zeros((0,), dtype=np.int64))
                        residual_rows_by_actor.append(np.zeros((0,), dtype=np.int64))
                        continue
                    opponent_policy_ids = np.asarray(
                        actor.opponent_policy_id_by_env[opponent_indices],
                        dtype=object,
                    )
                    heuristic_mask = (
                        np.isin(opponent_policy_ids, heuristic_policy_ids)
                        if heuristic_policy_ids
                        else np.zeros(opponent_policy_ids.shape, dtype=np.bool_)
                    )
                    mirror_mask = opponent_policy_ids == _MIRROR_OPPONENT_POLICY_ID
                    heuristic_rows_by_actor.append(opponent_indices[heuristic_mask])
                    mirror_rows_by_actor.append(opponent_indices[mirror_mask])
                    residual_rows_by_actor.append(opponent_indices[~(heuristic_mask | mirror_mask)])
                for actor, heuristic_rows, mirror_rows, residual_rows in zip(
                    actors,
                    heuristic_rows_by_actor,
                    mirror_rows_by_actor,
                    residual_rows_by_actor,
                    strict=True,
                ):
                    state_by_actor[int(actor.actor_id)]["counters"]["opponent_row_count"] += int(
                        heuristic_rows.shape[0] + mirror_rows.shape[0] + residual_rows.shape[0]
                    )
                sampled_policy_rows_by_actor = [
                    (
                        np.concatenate((focal_rows, mirror_rows), axis=0).astype(np.int64, copy=False)
                        if fuse_mirror_policy_rows and mirror_rows.size > 0
                        else focal_rows
                    )
                    for focal_rows, mirror_rows in zip(policy_row_indices, mirror_rows_by_actor, strict=True)
                ]
                forward_started = time.perf_counter()
                if any(rows.size > 0 for rows in sampled_policy_rows_by_actor):
                    self._central_sample_policy_rows_ids(
                        actors=actors,
                        batches=batches,
                        obs_steps=obs_steps,
                        actor_steps=actor_steps,
                        row_indices_by_actor=sampled_policy_rows_by_actor,
                        values_outs=value_steps,
                        actions_outs=action_steps,
                        logp_outs=logp_steps,
                    )
                self._record_batch_timer_ms("central_focal_policy", time.perf_counter() - forward_started)
                per_actor_forward_ms = int(
                    ((time.perf_counter() - forward_started) * 1000.0) / max(len(actors), 1)
                )
                for state in state_by_actor.values():
                    state["counters"]["actor_policy_forward_ms"] += per_actor_forward_ms

                overwrite_started = time.perf_counter()
                if fuse_mirror_policy_rows and heuristic_policy_ids:
                    heuristic_actors: list[_ActorState] = []
                    heuristic_obs_steps: list[np.ndarray] = []
                    heuristic_actor_steps: list[np.ndarray] = []
                    heuristic_row_indices_for_advance: list[np.ndarray] = []
                    for actor, obs_step, actor_step, heuristic_rows in zip(
                        actors,
                        obs_steps,
                        actor_steps,
                        heuristic_rows_by_actor,
                        strict=True,
                    ):
                        if heuristic_rows.size == 0:
                            continue
                        heuristic_actors.append(actor)
                        heuristic_obs_steps.append(obs_step)
                        heuristic_actor_steps.append(actor_step)
                        heuristic_row_indices_for_advance.append(heuristic_rows)
                    if heuristic_actors:
                        self._central_advance_actor_rows(
                            actors=heuristic_actors,
                            obs_steps=heuristic_obs_steps,
                            actor_steps=heuristic_actor_steps,
                            row_indices_by_actor=heuristic_row_indices_for_advance,
                        )
                for actor, batch, obs_step, actor_step, value_step, action_step, logp_step, heuristic_rows, residual_rows in zip(
                    actors,
                    batches,
                    obs_steps,
                    actor_steps,
                    value_steps,
                    action_steps,
                    logp_steps,
                    heuristic_rows_by_actor,
                    residual_rows_by_actor,
                    strict=True,
                ):
                    legal_ids, legal_offsets = _require_ids_offsets(batch)
                    legal_action_meta = _optional_legal_action_meta(batch)
                    if fuse_mirror_policy_rows:
                        if heuristic_rows.size > 0:
                            self._apply_opponent_rows_ids(
                                actor=actor,
                                row_indices=heuristic_rows,
                                obs_step=obs_step,
                                actor_step=actor_step,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                                legal_action_meta=legal_action_meta,
                                logits_out=None,
                                values_out=value_step,
                                actions_out=action_step,
                                logp_out=logp_step,
                                rng=actor.rng,
                                sample_actions=True,
                                heuristic_rows_hidden_already_advanced=True,
                            )
                        opponent_rows = residual_rows
                    else:
                        opponent_rows = np.flatnonzero(actor_step != actor.focal_seat_by_env)
                    if opponent_rows.size > 0:
                        self._apply_opponent_rows_ids(
                            actor=actor,
                            row_indices=opponent_rows,
                            obs_step=obs_step,
                            actor_step=actor_step,
                            legal_ids=legal_ids,
                            legal_offsets=legal_offsets,
                            legal_action_meta=legal_action_meta,
                            logits_out=None,
                            values_out=value_step,
                            actions_out=action_step,
                            logp_out=logp_step,
                            rng=actor.rng,
                            sample_actions=True,
                        )
                self._record_batch_timer_ms("central_fixed_opponent_overwrite", time.perf_counter() - overwrite_started)
                per_actor_overwrite_ms = int(
                    ((time.perf_counter() - overwrite_started) * 1000.0) / max(len(actors), 1)
                )
                for state in state_by_actor.values():
                    state["counters"]["fixed_opponent_routing_ms"] += per_actor_overwrite_ms
                logits_steps: list[np.ndarray | None] = [None for _ in actors]
            else:
                logits_steps = [np.empty((N, self.action_dim), dtype=np.float32) for _ in actors]
                value_steps = [np.empty((N,), dtype=np.float32) for _ in actors]
                for actor, actor_step in zip(actors, actor_steps, strict=True):
                    focal_rows = np.flatnonzero(actor_step == actor.focal_seat_by_env)
                    opponent_rows = np.flatnonzero(actor_step != actor.focal_seat_by_env)
                    state_by_actor[int(actor.actor_id)]["counters"]["focal_row_count"] += int(focal_rows.shape[0])
                    state_by_actor[int(actor.actor_id)]["counters"]["opponent_row_count"] += int(opponent_rows.shape[0])
                forward_started = time.perf_counter()
                self._central_forward_all_rows(
                    actors=actors,
                    batches=batches,
                    obs_steps=obs_steps,
                    actor_steps=actor_steps,
                    logits_outs=cast(Sequence[np.ndarray], logits_steps),
                    values_outs=value_steps,
                )
                self._record_batch_timer_ms("central_focal_policy", time.perf_counter() - forward_started)
                per_actor_forward_ms = int(
                    ((time.perf_counter() - forward_started) * 1000.0) / max(len(actors), 1)
                )
                for state in state_by_actor.values():
                    state["counters"]["actor_policy_forward_ms"] += per_actor_forward_ms

                overwrite_started = time.perf_counter()
                self._overwrite_central_outputs_with_configured_opponents(
                    actors=actors,
                    batches=batches,
                    obs_steps=obs_steps,
                    actor_steps=actor_steps,
                    logits_outs=cast(Sequence[np.ndarray | None], logits_steps),
                    values_outs=value_steps,
                )
                per_actor_overwrite_ms = int(
                    ((time.perf_counter() - overwrite_started) * 1000.0) / max(len(actors), 1)
                )
                for state in state_by_actor.values():
                    state["counters"]["fixed_opponent_routing_ms"] += per_actor_overwrite_ms

            next_batches: list[DecisionBoundaryBatch] = []
            for actor_index, (actor, batch, obs_storage_step, actor_step, logits_step, value_step) in enumerate(
                zip(
                    actors,
                    batches,
                    obs_storage_steps,
                    actor_steps,
                    logits_steps,
                    value_steps,
                    strict=True,
                )
            ):
                state = state_by_actor[int(actor.actor_id)]
                obs_step = np.asarray(obs_storage_step, dtype=np.float32)
                focal_rows = actor_step == actor.focal_seat_by_env
                state["policy_train_mask"][step_index] = focal_rows
                if actor.layout_name == "i16_legal_ids":
                    legal_ids, legal_offsets = _require_ids_offsets(batch)
                    legal_action_meta = _optional_legal_action_meta(batch)
                    teacher_started = time.perf_counter()
                    teacher_family, teacher_slot, teacher_attack_type, teacher_valid = self._teacher_labels_from_ids(
                        focal_rows=focal_rows,
                        decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                        obs_step=obs_step,
                        legal_ids=legal_ids,
                        legal_offsets=legal_offsets,
                        legal_action_meta=legal_action_meta,
                        counters=state["counters"],
                    )
                    state["counters"]["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                    state["counters"]["packed_candidate_count"] += int(np.asarray(legal_ids).shape[0])
                    packed_legal_ids = np.asarray(legal_ids, dtype=np.int64)
                    packed_legal_offsets = np.asarray(legal_offsets, dtype=np.int64)
                    offset_base = int(state["packed_offsets"][-1][-1])
                    state["packed_ids"].append(np.asarray(legal_ids, dtype=np.uint32))
                    if legal_action_meta is not None:
                        state["packed_meta"].append(np.asarray(legal_action_meta, dtype=np.uint16))
                    state["packed_offsets"].append(np.asarray(legal_offsets[1:] + offset_base, dtype=np.uint32))
                    if structured_central_packed:
                        action_step = np.asarray(action_steps[actor_index], dtype=np.int64)
                        logp_step = np.asarray(logp_steps[actor_index], dtype=np.float32)
                        env_started = time.perf_counter()
                        next_batch = actor.env.step(np.asarray(action_step, dtype=np.uint32))
                        state["counters"]["actor_env_step_ms"] += int(
                            (time.perf_counter() - env_started) * 1000.0
                        )
                    elif hasattr(actor.env, "step_sample_from_logits_with_logp"):
                        sample_seeds = actor.rng.integers(0, np.iinfo(np.int64).max, size=N, dtype=np.int64)
                        env_started = time.perf_counter()
                        next_batch, fused_actions, fused_logp = actor.env.step_sample_from_logits_with_logp(
                            cast(np.ndarray, logits_step),
                            sample_seeds,
                        )
                        state["counters"]["actor_env_step_ms"] += int(
                            (time.perf_counter() - env_started) * 1000.0
                        )
                        action_step = np.asarray(fused_actions, dtype=np.int64)
                        logp_step = np.asarray(fused_logp, dtype=np.float32)
                    else:
                        env_started = time.perf_counter()
                        action_step, logp_step, _entropy = sample_actions_from_legal_ids(
                            cast(np.ndarray, logits_step),
                            legal_ids,
                            legal_offsets,
                            rng=actor.rng,
                            pass_action_id=self.config.pass_action_id,
                        )
                        next_batch = actor.env.step(np.asarray(action_step, dtype=np.uint32))
                        state["counters"]["actor_env_step_ms"] += int(
                            (time.perf_counter() - env_started) * 1000.0
                        )
                    summary_started = time.perf_counter()
                    update_action_summary_from_ids(
                        counters=state["counters"],
                        state=state["action_sequence_state"],
                        actions=action_step,
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        pass_action_id=self.config.pass_action_id,
                    )
                    state["counters"]["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
                else:
                    legal_mask = _require_mask(batch)
                    legal_mask_array = np.asarray(legal_mask, dtype=np.bool_)
                    teacher_started = time.perf_counter()
                    teacher_family, teacher_slot, teacher_attack_type, teacher_valid = self._teacher_labels_from_mask(
                        focal_rows=focal_rows,
                        decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                        obs_step=obs_step,
                        legal_mask=legal_mask_array,
                        counters=state["counters"],
                    )
                    state["counters"]["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                    state["mask_steps"].append(legal_mask_array)
                    env_started = time.perf_counter()
                    action_step, logp_step, _entropy = sample_actions_from_mask(
                        logits_step,
                        legal_mask,
                        rng=actor.rng,
                        pass_action_id=self.config.pass_action_id,
                    )
                    state["counters"]["actor_env_step_ms"] += int(
                        (time.perf_counter() - env_started) * 1000.0
                    )
                    summary_started = time.perf_counter()
                    update_action_summary_from_mask(
                        counters=state["counters"],
                        state=state["action_sequence_state"],
                        actions=action_step,
                        legal_mask=legal_mask_array,
                        pass_action_id=self.config.pass_action_id,
                    )
                    state["counters"]["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
                    env_started = time.perf_counter()
                    next_batch = actor.env.step(np.asarray(action_step, dtype=np.uint32))
                    state["counters"]["actor_env_step_ms"] += int(
                        (time.perf_counter() - env_started) * 1000.0
                    )
                done = np.logical_or(next_batch.terminated, next_batch.truncated)

                state["obs"][step_index] = obs_storage_step
                state["actions"][step_index] = np.asarray(action_step, dtype=np.uint16)
                state["rewards"][step_index] = np.asarray(next_batch.reward, dtype=np.float32)
                state["terminated"][step_index] = np.asarray(next_batch.terminated, dtype=np.bool_)
                state["truncated"][step_index] = np.asarray(next_batch.truncated, dtype=np.bool_)
                state["to_play_seat"][step_index] = actor_step.astype(np.int8, copy=False)
                state["behavior_logp"][step_index] = np.asarray(logp_step, dtype=np.float32)
                state["values"][step_index] = value_step
                state["episode_seed"][step_index] = np.asarray(next_batch.episode_seed, dtype=np.uint64)
                state["teacher_family"][step_index] = teacher_family
                state["teacher_slot"][step_index] = teacher_slot
                state["teacher_attack_type"][step_index] = teacher_attack_type
                state["teacher_valid"][step_index] = teacher_valid

                if np.any(done):
                    _accumulate_timeout_counters(
                        counters=state["counters"],
                        batch=next_batch,
                        done=done,
                        timeout_limits=timeout_limits_by_actor[int(actor.actor_id)],
                    )
                    reset_started = time.perf_counter()
                    reset_hidden = actor.model.initial_seat_hidden(int(np.count_nonzero(done)), device=self._device)
                    done_mask = torch.as_tensor(done, dtype=torch.bool, device=self._device)
                    actor.seat_hidden[done_mask] = reset_hidden
                    actor.opponent_hidden[done_mask] = reset_hidden
                    self._assign_episode_roles(actor, done.astype(np.bool_, copy=False))
                    reset_action_sequence_state(state["action_sequence_state"], done.astype(np.bool_, copy=False))
                    next_batch = self._reset_done_rows(actor, done.astype(np.bool_, copy=False))
                    state["counters"]["actor_done_reset_ms"] += int(
                        (time.perf_counter() - reset_started) * 1000.0
                    )
                next_batches.append(next_batch)
            batches = next_batches

        bootstrap_obs_steps = [np.asarray(batch.obs, dtype=np.float32) for batch in batches]
        bootstrap_actor_steps = [np.asarray(batch.actor, dtype=np.int64) for batch in batches]
        bootstrap_values = [np.empty((N,), dtype=np.float32) for _ in actors]
        bootstrap_started = time.perf_counter()
        if structured_central_packed:
            self._central_value_actor_rows(
                actors=actors,
                obs_steps=bootstrap_obs_steps,
                actor_steps=bootstrap_actor_steps,
                row_indices_by_actor=[np.arange(N, dtype=np.int64) for _ in actors],
                values_outs=bootstrap_values,
            )
        else:
            self._central_forward_all_rows(
                actors=actors,
                batches=batches,
                obs_steps=bootstrap_obs_steps,
                actor_steps=bootstrap_actor_steps,
                logits_outs=[np.empty((N, self.action_dim), dtype=np.float32) for _ in actors],
                values_outs=bootstrap_values,
            )
        bootstrap_forward_ms = int(((time.perf_counter() - bootstrap_started) * 1000.0) / max(len(actors), 1))
        for state in state_by_actor.values():
            state["counters"]["actor_bootstrap_ms"] += bootstrap_forward_ms

        if not structured_central_packed:
            overwrite_started = time.perf_counter()
            self._overwrite_central_outputs_with_configured_opponents(
                actors=actors,
                batches=batches,
                obs_steps=bootstrap_obs_steps,
                actor_steps=bootstrap_actor_steps,
                logits_outs=[None for _ in actors],
                values_outs=bootstrap_values,
            )
            bootstrap_overwrite_ms = int(
                ((time.perf_counter() - overwrite_started) * 1000.0) / max(len(actors), 1)
            )
            for state in state_by_actor.values():
                state["counters"]["fixed_opponent_routing_ms"] += bootstrap_overwrite_ms

        unrolls: list[RuntimeUnroll] = []
        for actor, batch, bootstrap_value in zip(actors, batches, bootstrap_values, strict=True):
            state = state_by_actor[int(actor.actor_id)]
            actor.current_batch = batch
            state["counters"]["copied_bytes_estimate"] += int(
                state["obs"].nbytes
                + state["actions"].nbytes
                + state["rewards"].nbytes
                + state["terminated"].nbytes
                + state["truncated"].nbytes
                + state["to_play_seat"].nbytes
                + state["behavior_logp"].nbytes
                + state["values"].nbytes
                + state["episode_seed"].nbytes
                + state["policy_train_mask"].nbytes
                + state["teacher_family"].nbytes
                + state["teacher_slot"].nbytes
                + state["teacher_attack_type"].nbytes
                + state["teacher_valid"].nbytes
                + np.asarray(batch.obs, dtype=np.float32).nbytes
                + np.asarray(batch.actor, dtype=np.int64).nbytes
                + np.asarray(bootstrap_value, dtype=np.float32).nbytes
            )
            _merge_simulator_timing_counters(state["counters"], actor.env)
            state["counters"]["collect_actor_unroll_ms"] += int(
                ((time.perf_counter() - central_started) * 1000.0) / max(len(actors), 1)
            )
            unrolls.append(
                RuntimeUnroll(
                    actor_id=actor.actor_id,
                    unroll_seq=actor.next_unroll_seq,
                    behavior_policy_version=actor.snapshot_version,
                    unroll_hash=_hash_unroll(
                        actions=state["actions"],
                        rewards=state["rewards"],
                        episode_seed=state["episode_seed"],
                    ),
                    obs=state["obs"],
                    actions=state["actions"],
                    rewards=state["rewards"],
                    terminated=state["terminated"],
                    truncated=state["truncated"],
                    to_play_seat=state["to_play_seat"],
                    behavior_logp=state["behavior_logp"],
                    values=state["values"],
                    legal_actions=(
                        LegalActionBatch.from_packed(
                            np.concatenate(state["packed_ids"], axis=0)
                            if state["packed_ids"]
                            else np.zeros((0,), dtype=np.uint32),
                            np.concatenate(state["packed_offsets"], axis=0),
                            meta=(
                                np.concatenate(state["packed_meta"], axis=0)
                                if state["packed_meta"]
                                else None
                            ),
                            action_space=int(self.action_dim),
                        )
                        if actor.layout_name == "i16_legal_ids"
                        else LegalActionBatch.from_mask(
                            np.stack(state["mask_steps"], axis=0),
                            action_space=int(self.action_dim),
                        )
                    ),
                    bootstrap_obs=np.asarray(batch.obs, dtype=np.float32),
                    bootstrap_actor=np.asarray(batch.actor, dtype=np.int64),
                    bootstrap_value=np.asarray(bootstrap_value, dtype=np.float32),
                    initial_hidden_state=state["initial_hidden_state"],
                    final_hidden_state=actor.seat_hidden.detach().cpu().numpy().copy(),
                    episode_seed=state["episode_seed"],
                    policy_train_mask=state["policy_train_mask"],
                    teacher_family=state["teacher_family"],
                    teacher_slot=state["teacher_slot"],
                    teacher_attack_type=state["teacher_attack_type"],
                    teacher_valid=state["teacher_valid"],
                    behavior_logits=None,
                    counters=dict(state["counters"]),
                )
            )
            actor.next_unroll_seq += 1
        return unrolls

    def _collect_actor_unroll(self, actor: _ActorState) -> RuntimeUnroll:
        unroll_started = time.perf_counter()
        T = int(self.config.unroll_length)
        N = int(self.config.envs_per_actor)
        obs_dtype = np.asarray(actor.current_batch.obs).dtype
        obs = np.zeros((T, N, self.observation_dim), dtype=obs_dtype)
        actions = np.zeros((T, N), dtype=np.uint16)
        rewards = np.zeros((T, N), dtype=np.float32)
        terminated = np.zeros((T, N), dtype=np.bool_)
        truncated = np.zeros((T, N), dtype=np.bool_)
        to_play_seat = np.zeros((T, N), dtype=np.int8)
        behavior_logp = np.zeros((T, N), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        episode_seed = np.zeros((T, N), dtype=np.uint64)
        policy_train_mask = np.zeros((T, N), dtype=np.bool_)
        teacher_family = np.full((T, N), -1, dtype=np.int32)
        teacher_slot = np.full((T, N), -1, dtype=np.int32)
        teacher_attack_type = np.full((T, N), -1, dtype=np.int32)
        teacher_valid = np.zeros((T, N), dtype=np.bool_)
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
        mask_steps: list[np.ndarray] = []
        counters = _collector_counter_template()
        action_sequence_state = make_action_sequence_state(N)
        timeout_limits = _timeout_limits_for_env(actor.env)

        batch = actor.current_batch
        initial_hidden_state = actor.seat_hidden.detach().cpu().numpy().copy()
        for step_index in range(T):
            obs_storage_step = np.asarray(batch.obs)
            obs_step = np.asarray(batch.obs, dtype=np.float32)
            actor_step = np.asarray(batch.actor, dtype=np.int64)
            if obs_step.shape != (N, self.observation_dim):
                raise RuntimeError(f"unexpected actor obs shape: {obs_step.shape}")
            if np.any((actor_step != 0) & (actor_step != 1)):
                raise RuntimeError(f"actor runtime only supports live seat rows, got {actor_step.tolist()}")
            focal_rows = actor_step == actor.focal_seat_by_env
            value_step = np.zeros((N,), dtype=np.float32)
            action_step = np.zeros((N,), dtype=np.int64)
            logp_step = np.zeros((N,), dtype=np.float32)
            policy_train_mask[step_index] = focal_rows
            logits_step = np.empty((N, self.action_dim), dtype=np.float32)

            if actor.layout_name == "i16_legal_ids":
                legal_ids, legal_offsets = _require_ids_offsets(batch)
                legal_action_meta = _optional_legal_action_meta(batch)
                teacher_started = time.perf_counter()
                (
                    teacher_family_step,
                    teacher_slot_step,
                    teacher_attack_type_step,
                    teacher_valid_step,
                ) = self._teacher_labels_from_ids(
                    focal_rows=focal_rows,
                    decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                    obs_step=obs_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    counters=counters,
                )
                counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                counters["packed_candidate_count"] += int(np.asarray(legal_ids).shape[0])
                packed_legal_ids = np.asarray(legal_ids, dtype=np.int64)
                packed_legal_offsets = np.asarray(legal_offsets, dtype=np.int64)
                offset_base = int(packed_offsets[-1][-1])
                if self._use_simulator_fused_logits_step and hasattr(actor.env, "step_sample_from_logits_with_logp"):
                    packed_ids.append(np.asarray(legal_ids, dtype=np.uint32))
                    if legal_action_meta is not None:
                        packed_meta.append(np.asarray(legal_action_meta, dtype=np.uint16))
                    packed_offsets.append(np.asarray(legal_offsets[1:] + offset_base, dtype=np.uint32))
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_ids(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_ids=legal_ids,
                        legal_offsets=legal_offsets,
                        legal_action_meta=legal_action_meta,
                        logits_out=logits_step,
                        values_out=value_step,
                        actions_out=None,
                        logp_out=None,
                        rng=actor.rng,
                        sample_actions=False,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    sample_seeds = actor.rng.integers(0, np.iinfo(np.int64).max, size=N, dtype=np.int64)
                    env_started = time.perf_counter()
                    next_batch, fused_actions, fused_logp = actor.env.step_sample_from_logits_with_logp(
                        logits_step,
                        sample_seeds,
                    )
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    action_step = np.asarray(fused_actions, dtype=np.int64)
                    logp_step = np.asarray(fused_logp, dtype=np.float32)
                    summary_started = time.perf_counter()
                    update_action_summary_from_ids(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        pass_action_id=self.config.pass_action_id,
                    )
                    counters["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
                else:
                    packed_ids.append(np.asarray(legal_ids, dtype=np.uint32))
                    if legal_action_meta is not None:
                        packed_meta.append(np.asarray(legal_action_meta, dtype=np.uint16))
                    packed_offsets.append(np.asarray(legal_offsets[1:] + offset_base, dtype=np.uint32))
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_ids(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_ids=legal_ids,
                        legal_offsets=legal_offsets,
                        legal_action_meta=legal_action_meta,
                        logits_out=None,
                        values_out=value_step,
                        actions_out=action_step,
                        logp_out=logp_step,
                        rng=actor.rng,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    env_started = time.perf_counter()
                    next_batch = actor.env.step(action_step.astype(np.uint32, copy=False))
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    summary_started = time.perf_counter()
                    update_action_summary_from_ids(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        pass_action_id=self.config.pass_action_id,
                    )
                    counters["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
            else:
                legal_mask = _require_mask(batch)
                teacher_started = time.perf_counter()
                (
                    teacher_family_step,
                    teacher_slot_step,
                    teacher_attack_type_step,
                    teacher_valid_step,
                ) = self._teacher_labels_from_mask(
                    focal_rows=focal_rows,
                    decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                    obs_step=obs_step,
                    legal_mask=np.asarray(legal_mask, dtype=np.bool_),
                    counters=counters,
                )
                counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                if self._use_simulator_fused_logits_step:
                    current_legal_mask = np.asarray(legal_mask, dtype=np.bool_).copy()
                    mask_steps.append(current_legal_mask)
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_mask(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_mask=current_legal_mask,
                        logits_out=logits_step,
                        values_out=value_step,
                        actions_out=None,
                        logp_out=None,
                        rng=actor.rng,
                        sample_actions=False,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    sample_seeds = actor.rng.integers(0, np.iinfo(np.int64).max, size=N, dtype=np.int64)
                    env_started = time.perf_counter()
                    next_batch, fused_actions = actor.env.step_sample_from_logits(logits_step, sample_seeds)
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    action_step = np.asarray(fused_actions, dtype=np.int64)
                    logp_step = masked_logp_from_mask(
                        logits_step,
                        current_legal_mask,
                        action_step.astype(np.uint32, copy=False),
                        pass_action_id=self.config.pass_action_id,
                    )
                    summary_started = time.perf_counter()
                    update_action_summary_from_mask(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_mask=current_legal_mask,
                        pass_action_id=self.config.pass_action_id,
                    )
                    counters["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
                else:
                    legal_mask_array = np.asarray(legal_mask, dtype=np.bool_)
                    mask_steps.append(legal_mask_array)
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_mask(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_mask=legal_mask,
                        logits_out=None,
                        values_out=value_step,
                        actions_out=action_step,
                        logp_out=logp_step,
                        rng=actor.rng,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    summary_started = time.perf_counter()
                    update_action_summary_from_mask(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_mask=legal_mask_array,
                        pass_action_id=self.config.pass_action_id,
                    )
                    counters["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
                    env_started = time.perf_counter()
                    next_batch = actor.env.step(action_step.astype(np.uint32, copy=False))
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
            done = np.logical_or(next_batch.terminated, next_batch.truncated)

            obs[step_index] = obs_storage_step
            actions[step_index] = action_step.astype(np.uint16, copy=False)
            rewards[step_index] = np.asarray(next_batch.reward, dtype=np.float32)
            terminated[step_index] = np.asarray(next_batch.terminated, dtype=np.bool_)
            truncated[step_index] = np.asarray(next_batch.truncated, dtype=np.bool_)
            to_play_seat[step_index] = actor_step.astype(np.int8, copy=False)
            behavior_logp[step_index] = logp_step
            values[step_index] = value_step
            episode_seed[step_index] = np.asarray(next_batch.episode_seed, dtype=np.uint64)
            teacher_family[step_index] = teacher_family_step
            teacher_slot[step_index] = teacher_slot_step
            teacher_attack_type[step_index] = teacher_attack_type_step
            teacher_valid[step_index] = teacher_valid_step

            if np.any(done):
                _accumulate_timeout_counters(
                    counters=counters,
                    batch=next_batch,
                    done=done,
                    timeout_limits=timeout_limits,
                )
                self._update_outcomes(
                    actor=actor,
                    acting_seat=actor_step,
                    terminal_batch=next_batch,
                    done=done.astype(np.bool_, copy=False),
                )
                reset_started = time.perf_counter()
                reset_hidden = actor.model.initial_seat_hidden(int(np.count_nonzero(done)), device=self._device)
                done_mask = torch.as_tensor(done, dtype=torch.bool, device=self._device)
                actor.seat_hidden[done_mask] = reset_hidden
                actor.opponent_hidden[done_mask] = reset_hidden
                self._assign_episode_roles(actor, done.astype(np.bool_, copy=False))
                reset_action_sequence_state(action_sequence_state, done.astype(np.bool_, copy=False))
                batch = self._reset_done_rows(actor, done.astype(np.bool_, copy=False))
                counters["actor_done_reset_ms"] += int((time.perf_counter() - reset_started) * 1000.0)
            else:
                batch = next_batch

        actor.current_batch = batch
        bootstrap_value = np.zeros((batch.obs.shape[0],), dtype=np.float32)
        bootstrap_obs = np.asarray(batch.obs, dtype=np.float32)
        bootstrap_actor = np.asarray(batch.actor, dtype=np.int64)
        valid_bootstrap_rows = (bootstrap_actor == 0) | (bootstrap_actor == 1)
        if np.any(valid_bootstrap_rows):
            bootstrap_started = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ):
                actor_model = _actor_inference_model(actor)
                value_seat_aware = getattr(actor_model, "value_seat_aware", None)
                if callable(value_seat_aware):
                    bootstrap_value_tensor = value_seat_aware(
                        torch.as_tensor(bootstrap_obs[valid_bootstrap_rows], device=self._device),
                        torch.as_tensor(bootstrap_actor[valid_bootstrap_rows], device=self._device, dtype=torch.long),
                        actor.seat_hidden[valid_bootstrap_rows],
                    )
                else:
                    _, bootstrap_value_tensor, _ = actor_model.forward_seat_aware(
                        torch.as_tensor(bootstrap_obs[valid_bootstrap_rows], device=self._device),
                        torch.as_tensor(bootstrap_actor[valid_bootstrap_rows], device=self._device, dtype=torch.long),
                        actor.seat_hidden[valid_bootstrap_rows],
                    )
            bootstrap_value[valid_bootstrap_rows] = (
                bootstrap_value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            counters["actor_bootstrap_ms"] += int((time.perf_counter() - bootstrap_started) * 1000.0)
        unroll = RuntimeUnroll(
            actor_id=actor.actor_id,
            unroll_seq=actor.next_unroll_seq,
            behavior_policy_version=actor.snapshot_version,
            unroll_hash=_hash_unroll(actions=actions, rewards=rewards, episode_seed=episode_seed),
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            to_play_seat=to_play_seat,
            behavior_logp=behavior_logp,
            values=values,
            legal_actions=(
                LegalActionBatch.from_packed(
                    np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                    np.concatenate(packed_offsets, axis=0),
                    meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
                    action_space=int(self.action_dim),
                )
                if actor.layout_name == "i16_legal_ids"
                else LegalActionBatch.from_mask(np.stack(mask_steps, axis=0), action_space=int(self.action_dim))
            ),
            bootstrap_obs=bootstrap_obs,
            bootstrap_actor=bootstrap_actor,
            bootstrap_value=bootstrap_value,
            initial_hidden_state=initial_hidden_state,
            final_hidden_state=actor.seat_hidden.detach().cpu().numpy().copy(),
            episode_seed=episode_seed,
            policy_train_mask=policy_train_mask,
            teacher_family=teacher_family,
            teacher_slot=teacher_slot,
            teacher_attack_type=teacher_attack_type,
            teacher_valid=teacher_valid,
            behavior_logits=None,
            counters=counters,
        )
        counters["copied_bytes_estimate"] += int(
            obs.nbytes
            + actions.nbytes
            + rewards.nbytes
            + terminated.nbytes
            + truncated.nbytes
            + to_play_seat.nbytes
            + behavior_logp.nbytes
            + values.nbytes
            + episode_seed.nbytes
            + policy_train_mask.nbytes
            + teacher_family.nbytes
            + teacher_slot.nbytes
            + teacher_attack_type.nbytes
            + teacher_valid.nbytes
            + bootstrap_obs.nbytes
            + bootstrap_actor.nbytes
            + bootstrap_value.nbytes
        )
        _merge_simulator_timing_counters(counters, actor.env)
        counters["collect_actor_unroll_ms"] += int((time.perf_counter() - unroll_started) * 1000.0)
        actor.next_unroll_seq += 1
        return unroll

    def _build_learner_batch(
        self,
        unrolls: Sequence[RuntimeUnroll],
        *,
        gamma: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
        vtrace_rho_bar: float,
        vtrace_c_bar: float,
    ) -> dict[str, Any]:
        concat_started = time.perf_counter()
        obs = _concat_time_major_field(unrolls, "obs")
        actions = _concat_time_major_field(unrolls, "actions")
        rewards = _concat_time_major_field(unrolls, "rewards")
        terminated = _concat_time_major_field(unrolls, "terminated")
        truncated = _concat_time_major_field(unrolls, "truncated")
        to_play_seat = _concat_time_major_field(unrolls, "to_play_seat")
        behavior_logp = _concat_time_major_field(unrolls, "behavior_logp")
        behavior_values = _concat_time_major_field(unrolls, "values")
        bootstrap_value = np.concatenate([np.asarray(unroll.bootstrap_value, dtype=np.float32) for unroll in unrolls], axis=0)
        initial_hidden_state = _concat_batch_major_field(unrolls, "initial_hidden_state")
        policy_train_mask = _concat_time_major_field(unrolls, "policy_train_mask")
        teacher_family = _concat_optional_time_major_field(unrolls, "teacher_family")
        teacher_slot = _concat_optional_time_major_field(unrolls, "teacher_slot")
        teacher_attack_type = _concat_optional_time_major_field(unrolls, "teacher_attack_type")
        teacher_valid = _concat_optional_time_major_field(unrolls, "teacher_valid")
        legal_actions = _concatenate_legal_actions(unrolls, action_space=int(self.action_dim))
        self._record_batch_timer_ms("legal_concatenation", time.perf_counter() - concat_started)
        legal_mask = None if legal_actions.mask is None else legal_actions.mask
        discounts = np.logical_not(terminated).astype(np.float32) * float(gamma)
        if not truncation_bootstrap_value:
            discounts *= np.logical_not(truncated).astype(np.float32)

        return {
            "obs": obs,
            "actions": actions,
            "legal_actions": legal_actions,
            "legal_mask": legal_mask,
            "legal_action_meta": legal_actions.meta,
            "to_play_seat": to_play_seat,
            "actor": to_play_seat,
            "initial_hidden_state": initial_hidden_state,
            "rewards": rewards,
            "discounts": discounts,
            "behavior_logp": behavior_logp,
            "behavior_values": behavior_values,
            "bootstrap_value": bootstrap_value,
            "vtrace_rho_bar": float(vtrace_rho_bar),
            "vtrace_c_bar": float(vtrace_c_bar),
            "policy_train_mask": policy_train_mask,
            "teacher_family": teacher_family,
            "teacher_slot": teacher_slot,
            "teacher_attack_type": teacher_attack_type,
            "teacher_valid": teacher_valid,
        }

    def _build_ppo_batch(
        self,
        unrolls: Sequence[RuntimeUnroll],
        *,
        gamma: float,
        gae_lambda: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
    ) -> dict[str, Any]:
        bootstrap_values = [self._bootstrap_values(unroll) for unroll in unrolls]
        concat_started = time.perf_counter()
        obs = _concat_time_major_field(unrolls, "obs")
        actions = _concat_time_major_field(unrolls, "actions")
        rewards = _concat_time_major_field(unrolls, "rewards")
        terminated = _concat_time_major_field(unrolls, "terminated")
        truncated = _concat_time_major_field(unrolls, "truncated")
        to_play_seat = _concat_time_major_field(unrolls, "to_play_seat")
        old_logp = _concat_time_major_field(unrolls, "behavior_logp")
        old_values = _concat_time_major_field(unrolls, "values")
        initial_hidden_state = _concat_batch_major_field(unrolls, "initial_hidden_state")
        policy_train_mask = _concat_time_major_field(unrolls, "policy_train_mask")
        teacher_family = _concat_optional_time_major_field(unrolls, "teacher_family")
        teacher_slot = _concat_optional_time_major_field(unrolls, "teacher_slot")
        teacher_attack_type = _concat_optional_time_major_field(unrolls, "teacher_attack_type")
        teacher_valid = _concat_optional_time_major_field(unrolls, "teacher_valid")
        legal_actions = _concatenate_legal_actions(unrolls, action_space=int(self.action_dim))
        self._record_batch_timer_ms("legal_concatenation", time.perf_counter() - concat_started)
        legal_mask = None if legal_actions.mask is None else legal_actions.mask

        discounts = np.logical_not(terminated).astype(np.float32) * float(gamma)
        if not truncation_bootstrap_value:
            discounts *= np.logical_not(truncated).astype(np.float32)

        bootstrap_value = np.concatenate(bootstrap_values, axis=0)
        advantages = _gae_advantages(
            rewards=rewards,
            values=old_values,
            bootstrap_value=bootstrap_value,
            discounts=discounts,
            gae_lambda=float(gae_lambda),
        )
        returns = advantages + old_values

        return {
            "obs": obs,
            "actions": actions,
            "legal_actions": legal_actions,
            "legal_mask": legal_mask,
            "legal_action_meta": legal_actions.meta,
            "to_play_seat": to_play_seat,
            "actor": to_play_seat,
            "initial_hidden_state": initial_hidden_state,
            "rewards": rewards,
            "discounts": discounts,
            "old_logp": old_logp,
            "old_values": old_values,
            "returns": returns,
            "advantages": advantages,
            "policy_train_mask": policy_train_mask,
            "teacher_family": teacher_family,
            "teacher_slot": teacher_slot,
            "teacher_attack_type": teacher_attack_type,
            "teacher_valid": teacher_valid,
        }

    def _bootstrap_values(self, unroll: RuntimeUnroll) -> np.ndarray:
        bootstrap_value = np.zeros((unroll.bootstrap_obs.shape[0],), dtype=np.float32)
        valid_rows = (unroll.bootstrap_actor == 0) | (unroll.bootstrap_actor == 1)
        if not np.any(valid_rows):
            return bootstrap_value
        if self._bootstrap_models is not None:
            actor_model = self._bootstrap_models[int(unroll.actor_id)]
        else:
            actor_model = self._actors[int(unroll.actor_id)].model
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=self._actor_amp_enabled,
        ):
            value_seat_aware = getattr(actor_model, "value_seat_aware", None)
            if callable(value_seat_aware):
                value_tensor = value_seat_aware(
                    torch.as_tensor(unroll.bootstrap_obs[valid_rows], device=self._device),
                    torch.as_tensor(unroll.bootstrap_actor[valid_rows], device=self._device, dtype=torch.long),
                    torch.as_tensor(unroll.final_hidden_state[valid_rows], device=self._device),
                )
            else:
                _, value_tensor, _ = actor_model.forward_seat_aware(
                    torch.as_tensor(unroll.bootstrap_obs[valid_rows], device=self._device),
                    torch.as_tensor(unroll.bootstrap_actor[valid_rows], device=self._device, dtype=torch.long),
                    torch.as_tensor(unroll.final_hidden_state[valid_rows], device=self._device),
                )
        bootstrap_value[valid_rows] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return bootstrap_value

    def _runtime_metrics(
        self,
        selected: Sequence[RuntimeUnroll],
        *,
        occupancy_samples: Sequence[float],
    ) -> dict[str, float]:
        batch_env_steps = sum(int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in selected)
        now = time.time()
        elapsed = max(now - self._runtime_start, 1e-6)
        elapsed_window = max(now - self._runtime_last_metrics_time, 1e-6)
        self._runtime_last_metrics_time = now
        self._runtime_cumulative_env_steps += int(batch_env_steps)
        policy_lags = [
            float(self._last_published_snapshot_version - unroll.behavior_policy_version) for unroll in selected
        ]
        occupancy = np.asarray(tuple(occupancy_samples) or (0.0,), dtype=np.float64)
        lag_array = np.asarray(policy_lags or (0.0,), dtype=np.float64)
        effective_update = int(getattr(self, "_effective_learner_update", 0))
        counter_totals: dict[str, float] = {}
        for unroll in selected:
            counters = getattr(unroll, "counters", None)
            if counters is None:
                continue
            for key, value in counters.items():
                counter_totals[key] = counter_totals.get(key, 0.0) + float(value)
        tactical_rows = float(counter_totals.get("tactical_row_count", 0.0))
        packed_candidates = float(counter_totals.get("packed_candidate_count", 0.0))
        row_count_total = float(counter_totals.get("focal_row_count", 0.0) + counter_totals.get("opponent_row_count", 0.0))
        if row_count_total <= 0.0:
            row_count_total = tactical_rows
        metrics = {
            "actor_env_steps_per_sec": float(batch_env_steps / elapsed_window),
            "actor_env_steps_per_sec_cumulative": float(self._runtime_cumulative_env_steps / elapsed),
            "batch_env_steps": float(batch_env_steps),
            "queue_occupancy_p50": float(np.percentile(occupancy, 50)),
            "queue_occupancy_p90": float(np.percentile(occupancy, 90)),
            "policy_version_lag_p50": float(np.percentile(lag_array, 50)),
            "policy_version_lag_p90": float(np.percentile(lag_array, 90)),
            "league_effective_update": float(effective_update),
            "league_update_lag": float(max(0, int(getattr(self, "_current_learner_update", 0)) - effective_update)),
            "pfsp_pool_size": float(self._pfsp_pool_size),
            "pfsp_quarantined_opponents": float(self._pfsp_quarantined_opponents),
            "pfsp_champion_pool_size": float(self._pfsp_champion_pool_size),
            "pfsp_recent_pool_size": float(self._pfsp_recent_pool_size),
            "pfsp_hard_negative_pool_size": float(self._pfsp_hard_negative_pool_size),
            "pfsp_sampled_envs": float(self._pfsp_last_sampled_envs),
            "pfsp_mirror_envs": float(self._pfsp_last_mirror_envs),
            "pfsp_heuristic_public_envs": float(self._pfsp_last_heuristic_public_envs),
            "pfsp_noleague_baseline_envs": float(getattr(self, "_pfsp_last_noleague_baseline_envs", 0.0)),
            "pfsp_champion_envs": float(self._pfsp_last_champion_envs),
            "pfsp_recent_envs": float(self._pfsp_last_recent_envs),
            "pfsp_hard_negative_envs": float(self._pfsp_last_hard_negative_envs),
            "pfsp_epoch": float(self._pfsp_epoch),
            "tactical_row_count": tactical_rows,
            "packed_candidate_count": packed_candidates,
            "copied_bytes_estimate": float(counter_totals.get("copied_bytes_estimate", 0.0)),
            "avg_legal_actions_per_row": float(packed_candidates / max(row_count_total, 1.0)),
            **{f"collector_{key}": value for key, value in counter_totals.items()},
        }
        for key, value in counter_totals.items():
            if key.startswith("simulator_") and key.endswith("_ns"):
                metrics[f"timer_{key[:-3]}_ms"] = float(value / 1_000_000.0)
            elif key.startswith("simulator_python_"):
                metrics[f"timer_{key}_ms"] = float(value / 1_000_000.0)
        return metrics

    def _reset_done_rows(self, actor: _ActorState, done: np.ndarray) -> DecisionBoundaryBatch:
        try:
            return actor.env.reset_done(done)
        except RuntimeError:
            initial_hidden = actor.model.initial_seat_hidden(
                int(self.config.envs_per_actor),
                device=self._device,
            ).clone()
            actor.seat_hidden = initial_hidden.clone()
            actor.opponent_hidden = initial_hidden
            full_reset = np.ones(actor.focal_seat_by_env.shape, dtype=np.bool_)
            self._assign_episode_roles(actor, full_reset, initial=True)
            fallback_seed = int(actor.rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
            return actor.env.reset(seed=fallback_seed)

    def _reset_actor_state_for_fixed_opponents(self, actor: _ActorState) -> None:
        full_reset = np.ones(actor.focal_seat_by_env.shape, dtype=np.bool_)
        initial_hidden = actor.model.initial_seat_hidden(
            int(self.config.envs_per_actor),
            device=self._device,
        ).clone()
        actor.seat_hidden = initial_hidden.clone()
        actor.opponent_hidden = initial_hidden
        self._assign_episode_roles(actor, full_reset, initial=True)
        actor.current_batch = self._reset_done_rows(actor, full_reset)


def build_runtime_config(
    *,
    stack: StackConfig,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
    minimal_batch: bool = False,
) -> QueueRuntimeConfig:
    system = stack.config.system
    training = stack.config.training
    if system is None or training is None:
        raise RuntimeError("stack config is missing system or training blocks")

    configured_actor_count = int(system.actor_process_count)
    configured_envs_per_actor = int(system.envs_per_actor)
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=int(num_envs),
        runtime_mode=runtime_mode,
        configured_actor_count=configured_actor_count,
        configured_envs_per_actor=configured_envs_per_actor,
    )

    batch_unrolls_per_update = int(training.batch_unrolls_per_update)
    queue_capacity_unrolls = max(int(system.actor_queue_capacity_unrolls), batch_unrolls_per_update)
    if minimal_batch:
        batch_unrolls_per_update = int(actor_count)
        queue_capacity_unrolls = int(actor_count)

    return QueueRuntimeConfig(
        mode=runtime_mode,
        actor_count=actor_count,
        envs_per_actor=envs_per_actor,
        unroll_length=int(unroll_length),
        batch_unrolls_per_update=batch_unrolls_per_update,
        queue_capacity_unrolls=queue_capacity_unrolls,
        profile=profile,
        base_seed=int(seed),
        pass_action_id=int(pass_action_id),
        actor_reload_interval_updates=max(1, int(training.actor_reload_interval_updates)),
    )


def _resolve_actor_topology(
    *,
    num_envs: int,
    runtime_mode: QueueRuntimeMode,
    configured_actor_count: int,
    configured_envs_per_actor: int,
) -> tuple[int, int]:
    if runtime_mode != "train_async_fast":
        if int(num_envs) == int(configured_actor_count) * int(configured_envs_per_actor):
            return int(configured_actor_count), int(configured_envs_per_actor)
        return 1, int(num_envs)

    candidate_max_actors = max(1, int(configured_actor_count))
    divisors = [actor_count for actor_count in range(1, candidate_max_actors + 1) if num_envs % actor_count == 0]
    if not divisors:
        return 1, int(num_envs)

    def _score(actor_count: int) -> tuple[int, int, int]:
        envs_per_actor = int(num_envs // actor_count)
        in_band = 32 <= envs_per_actor <= 64
        band_penalty = 0 if in_band else 1
        target = 64 if in_band else 48
        return (band_penalty, abs(target - envs_per_actor), actor_count)

    best_actor_count = min(divisors, key=_score)
    return int(best_actor_count), int(num_envs // best_actor_count)


def _actor_seed(base_seed: int, actor_id: int) -> int:
    return int(np.uint64(base_seed) ^ (np.uint64(actor_id + 1) << np.uint64(32)))


def _concat_time_major_field(unrolls: Sequence[RuntimeUnroll], field_name: str) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    template = np.asarray(getattr(unrolls[0], field_name))
    total_batch = sum(int(np.asarray(getattr(unroll, field_name)).shape[1]) for unroll in unrolls)
    result = np.empty((template.shape[0], total_batch, *template.shape[2:]), dtype=template.dtype)
    offset = 0
    for unroll in unrolls:
        value = np.asarray(getattr(unroll, field_name))
        width = int(value.shape[1])
        result[:, offset : offset + width, ...] = value
        offset += width
    return result


def _concat_optional_time_major_field(unrolls: Sequence[RuntimeUnroll], field_name: str) -> np.ndarray | None:
    present_values = [getattr(unroll, field_name) for unroll in unrolls if getattr(unroll, field_name) is not None]
    if not present_values:
        return None
    template = np.asarray(present_values[0])
    total_batch = sum(
        int(np.asarray(getattr(unroll, field_name) if getattr(unroll, field_name) is not None else template).shape[1])
        for unroll in unrolls
    )
    result = np.empty((template.shape[0], total_batch, *template.shape[2:]), dtype=template.dtype)
    offset = 0
    for unroll in unrolls:
        raw_value = getattr(unroll, field_name)
        value = template if raw_value is None else np.asarray(raw_value, dtype=template.dtype)
        width = int(value.shape[1])
        result[:, offset : offset + width, ...] = value
        offset += width
    return result


def _concat_batch_major_field(unrolls: Sequence[RuntimeUnroll], field_name: str) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    template = np.asarray(getattr(unrolls[0], field_name))
    total_batch = sum(int(np.asarray(getattr(unroll, field_name)).shape[0]) for unroll in unrolls)
    result = np.empty((total_batch, *template.shape[1:]), dtype=template.dtype)
    offset = 0
    for unroll in unrolls:
        value = np.asarray(getattr(unroll, field_name))
        width = int(value.shape[0])
        result[offset : offset + width, ...] = value
        offset += width
    return result


def _concatenate_legal_actions(unrolls: Sequence[RuntimeUnroll], *, action_space: int) -> LegalActionBatch:
    packed_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
    mask_parts: list[np.ndarray] = []
    saw_packed = False
    saw_mask = False

    for unroll in unrolls:
        legal_actions = unroll.legal_actions
        if legal_actions.ids is not None and legal_actions.offsets is not None:
            saw_packed = True
            offset_base = int(packed_offsets[-1][-1])
            packed_offsets.append(np.asarray(legal_actions.offsets[1:] + offset_base, dtype=np.uint32))
            continue

        saw_mask = True
        mask_parts.append(
            legal_actions.to_mask(
                expected_shape=(int(unroll.obs.shape[0]), int(unroll.obs.shape[1])),
                action_space=int(action_space),
            )
        )

    if saw_packed and not saw_mask:
        total_time_steps = int(unrolls[0].obs.shape[0])
        for unroll in unrolls[1:]:
            if int(unroll.obs.shape[0]) != total_time_steps:
                raise RuntimeError("packed legal-action concatenation requires aligned unroll lengths")
        if not all(unroll.legal_actions.row_count == int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in unrolls):
            packed_ids: list[np.ndarray] = []
            packed_meta: list[np.ndarray] = []
            packed_offsets = [np.array([0], dtype=np.uint32)]
            any_meta = any(unroll.legal_actions.meta is not None for unroll in unrolls)
            for unroll in unrolls:
                legal_actions = unroll.legal_actions
                assert legal_actions.ids is not None and legal_actions.offsets is not None
                row_limit = int(unroll.obs.shape[0] * unroll.obs.shape[1])
                offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
                ids_limit = int(offsets[min(row_limit, max(offsets.size - 1, 0))])
                ids = np.asarray(legal_actions.ids[:ids_limit], dtype=np.uint32)
                offset_base = int(packed_offsets[-1][-1])
                packed_ids.append(ids)
                packed_offsets.append(np.asarray(offsets[1 : row_limit + 1] + offset_base, dtype=np.uint32))
                if any_meta and legal_actions.meta is not None:
                    packed_meta.append(np.asarray(legal_actions.meta[:ids_limit], dtype=np.uint16))
            return LegalActionBatch.from_packed(
                np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                np.concatenate(packed_offsets, axis=0),
                meta=(
                    np.concatenate(packed_meta, axis=0)
                    if packed_meta
                    else (np.zeros((0, _infer_packed_meta_width(unrolls)), dtype=np.uint16) if any_meta else None)
                ),
                action_space=int(action_space),
            )

        total_ids = sum(int(np.asarray(unroll.legal_actions.ids, dtype=np.uint32).size) for unroll in unrolls)
        total_rows = sum(int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in unrolls)
        total_batch = sum(int(unroll.obs.shape[1]) for unroll in unrolls)
        ordered_packed_ids = np.empty((total_ids,), dtype=np.uint32)
        any_meta = any(unroll.legal_actions.meta is not None for unroll in unrolls)
        ordered_packed_meta = (
            np.empty((total_ids, _infer_packed_meta_width(unrolls)), dtype=np.uint16)
            if any_meta and total_ids > 0
            else None
        )
        ordered_packed_offsets = np.empty((total_rows + 1,), dtype=np.uint32)
        ordered_packed_offsets[0] = 0
        ordered_widths = np.empty((total_time_steps, total_batch), dtype=np.uint32)
        batch_offset = 0
        for unroll in unrolls:
            legal_actions = unroll.legal_actions
            assert legal_actions.offsets is not None
            env_count = int(unroll.obs.shape[1])
            widths = np.diff(np.asarray(legal_actions.offsets, dtype=np.uint32)).reshape(total_time_steps, env_count)
            ordered_widths[:, batch_offset : batch_offset + env_count] = widths
            batch_offset += env_count
        ordered_packed_offsets[1:] = np.cumsum(ordered_widths.reshape(-1), dtype=np.uint64).astype(np.uint32, copy=False)
        ids_offset = 0
        for time_index in range(total_time_steps):
            for unroll in unrolls:
                legal_actions = unroll.legal_actions
                assert legal_actions.ids is not None and legal_actions.offsets is not None
                env_count = int(unroll.obs.shape[1])
                row_base = int(time_index * env_count)
                offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
                ids = np.asarray(legal_actions.ids, dtype=np.uint32)
                meta = None if legal_actions.meta is None else np.asarray(legal_actions.meta, dtype=np.uint16)
                start = int(offsets[row_base])
                end = int(offsets[row_base + env_count])
                width = end - start
                if width > 0:
                    ordered_packed_ids[ids_offset : ids_offset + width] = ids[start:end]
                    if ordered_packed_meta is not None:
                        if meta is None:
                            ordered_packed_meta[ids_offset : ids_offset + width] = np.iinfo(np.uint16).max
                        else:
                            ordered_packed_meta[ids_offset : ids_offset + width] = meta[start:end]
                ids_offset += width

        return LegalActionBatch.from_packed(
            ordered_packed_ids[:ids_offset],
            ordered_packed_offsets,
            meta=None if ordered_packed_meta is None else ordered_packed_meta[:ids_offset],
            action_space=int(action_space),
        )

    if saw_packed:
        mask_parts = [
            unroll.legal_actions.to_mask(
                expected_shape=(int(unroll.obs.shape[0]), int(unroll.obs.shape[1])),
                action_space=int(action_space),
            )
            for unroll in unrolls
        ]

    if not mask_parts:
        raise RuntimeError("runtime learner batch requires at least one legal-action payload")
    return LegalActionBatch.from_mask(np.concatenate(mask_parts, axis=1), action_space=int(action_space))


def _require_ids_offsets(batch: DecisionBoundaryBatch) -> tuple[np.ndarray, np.ndarray]:
    if batch.ids_offsets is None:
        raise RuntimeError("QueueRuntime requires ids_offsets legality batches")
    legal_ids, legal_offsets = batch.ids_offsets
    return np.asarray(legal_ids, dtype=np.uint32), np.asarray(legal_offsets, dtype=np.uint32)


def _optional_legal_action_meta(batch: DecisionBoundaryBatch) -> np.ndarray | None:
    if batch.legal_action_meta is None:
        return None
    return np.asarray(batch.legal_action_meta, dtype=np.uint16)


def _require_mask(batch: DecisionBoundaryBatch) -> np.ndarray:
    if batch.mask is None:
        raise RuntimeError("QueueRuntime expected dense mask legality for this actor batch")
    return np.asarray(batch.mask, dtype=np.bool_)


def _concatenate_batch_legal_actions(
    batches: Sequence[DecisionBoundaryBatch],
    *,
    action_space: int,
) -> LegalActionBatch | None:
    if not batches:
        return None
    if all(batch.mask is not None for batch in batches):
        masks = [np.asarray(batch.mask, dtype=np.bool_) for batch in batches]
        return LegalActionBatch.from_mask(
            np.expand_dims(np.concatenate(masks, axis=0), axis=0),
            action_space=int(action_space),
        )
    if all(batch.ids_offsets is not None for batch in batches):
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets = [np.array([0], dtype=np.uint32)]
        for batch in batches:
            legal_ids, legal_offsets = _require_ids_offsets(batch)
            offset_base = int(packed_offsets[-1][-1])
            packed_ids.append(np.asarray(legal_ids, dtype=np.uint32))
            legal_action_meta = _optional_legal_action_meta(batch)
            if legal_action_meta is not None:
                packed_meta.append(np.asarray(legal_action_meta, dtype=np.uint16))
            packed_offsets.append(np.asarray(legal_offsets[1:] + offset_base, dtype=np.uint32))
        return LegalActionBatch.from_packed(
            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
            np.concatenate(packed_offsets, axis=0),
            meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
            action_space=int(action_space),
        )
    return None


def _slice_packed_rows(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_ids: list[np.ndarray] = []
    offsets = [0]
    for row_index in row_indices.tolist():
        start = int(legal_offsets[int(row_index)])
        stop = int(legal_offsets[int(row_index) + 1])
        row_ids = np.asarray(legal_ids[start:stop], dtype=np.uint32)
        selected_ids.append(row_ids)
        offsets.append(offsets[-1] + int(row_ids.size))
    return (
        np.concatenate(selected_ids, axis=0) if selected_ids else np.zeros((0,), dtype=np.uint32),
        np.asarray(offsets, dtype=np.uint32),
    )


def _slice_packed_rows_with_meta(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    *,
    legal_action_meta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    subset_ids, subset_offsets = _slice_packed_rows(legal_ids, legal_offsets, row_indices)
    subset_meta = None
    if legal_action_meta is not None:
        selected_meta: list[np.ndarray] = []
        for row_index in row_indices.tolist():
            start = int(legal_offsets[int(row_index)])
            stop = int(legal_offsets[int(row_index) + 1])
            selected_meta.append(np.asarray(legal_action_meta[start:stop], dtype=np.uint16))
        subset_meta = (
            np.concatenate(selected_meta, axis=0)
            if selected_meta
            else np.zeros((0, legal_action_meta.shape[1]), dtype=np.uint16)
        )
    return subset_ids, subset_offsets, subset_meta


def _structured_legal_batch_from_mask(legal_mask: np.ndarray, row_indices: np.ndarray) -> LegalActionBatch:
    row_mask = np.asarray(legal_mask[row_indices], dtype=np.bool_)
    return LegalActionBatch.from_mask(np.expand_dims(row_mask, axis=0))


def _structured_legal_batch_from_packed(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    legal_action_meta: np.ndarray | None = None,
) -> LegalActionBatch:
    subset_ids, subset_offsets, subset_meta = _slice_packed_rows_with_meta(
        legal_ids,
        legal_offsets,
        row_indices,
        legal_action_meta=legal_action_meta,
    )
    return LegalActionBatch.from_packed(subset_ids, subset_offsets, meta=subset_meta)


def _infer_packed_meta_width(unrolls: Sequence[RuntimeUnroll]) -> int:
    for unroll in unrolls:
        if unroll.legal_actions.meta is not None:
            return int(np.asarray(unroll.legal_actions.meta).shape[1])
    return _DEFAULT_ACTION_META_WIDTH


def _hash_unroll(*, actions: np.ndarray, rewards: np.ndarray, episode_seed: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (actions, rewards, episode_seed):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _hash_state_dict(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(str(key).encode("utf-8"))
        value = state_dict[key]
        tensor = value.detach().cpu().contiguous() if torch.is_tensor(value) else torch.as_tensor(value)
        array = np.asarray(tensor.numpy())
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _gae_advantages(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: np.ndarray,
    discounts: np.ndarray,
    gae_lambda: float,
) -> np.ndarray:
    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    discounts_array = np.asarray(discounts, dtype=np.float32)
    bootstrap_array = np.asarray(bootstrap_value, dtype=np.float32)
    advantages = np.zeros_like(rewards_array, dtype=np.float32)
    gae = np.zeros((rewards_array.shape[1],), dtype=np.float32)
    next_values = bootstrap_array
    for timestep in range(rewards_array.shape[0] - 1, -1, -1):
        delta = rewards_array[timestep] + (discounts_array[timestep] * next_values) - values_array[timestep]
        gae = delta + (discounts_array[timestep] * float(gae_lambda) * gae)
        advantages[timestep] = gae
        next_values = values_array[timestep]
    return advantages
