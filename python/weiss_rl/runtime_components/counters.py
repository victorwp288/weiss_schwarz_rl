"""Counter helpers shared by queue-runtime collectors.

The runtime records these counters while collecting actor unrolls. Keeping the
logic here makes timeout accounting and simulator timing drains testable without
constructing the full queue runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.core.termination_reason import classify_episode_end_reason


def collector_counter_template() -> dict[str, int]:
    """Return a fresh collector counter map with all known runtime keys."""

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
        "pass_with_nonpass_penalty_count": 0,
        "pass_with_nonpass_penalty_total_micros": 0,
        "mulligan_select_with_confirm_penalty_count": 0,
        "mulligan_select_with_confirm_penalty_total_micros": 0,
        "mulligan_force_confirm_after_select_rows": 0,
        "mulligan_force_confirm_after_select_actions": 0,
        "main_move_only_force_pass_rows": 0,
        "main_move_only_force_pass_actions": 0,
        "attack_available_force_attack_rows": 0,
        "attack_available_force_attack_actions": 0,
        "max_consecutive_main_moves": 0,
        "focal_row_count": 0,
        "opponent_row_count": 0,
        "tactical_row_count": 0,
        "teacher_tactical_row_count": 0,
        "fixed_opponent_tactical_row_count": 0,
        "trajectory_retention_rows": 0,
        "packed_candidate_count": 0,
        "pfsp_sampled_envs": 0,
        "pfsp_mirror_envs": 0,
        "pfsp_heuristic_public_envs": 0,
        "pfsp_heuristic_public_variant_envs": 0,
        "pfsp_noleague_baseline_envs": 0,
        "pfsp_champion_envs": 0,
        "pfsp_recent_envs": 0,
        "pfsp_hard_negative_envs": 0,
        "pfsp_warmup_snapshot_envs": 0,
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


def accumulate_actor_role_row_counters(
    *,
    counters: dict[str, int],
    actor_step: np.ndarray,
    focal_seat_by_env: np.ndarray,
) -> tuple[int, int]:
    """Accumulate focal/opponent row counts for a collector step."""

    actor_arr = np.asarray(actor_step, dtype=np.int64)
    focal_arr = np.asarray(focal_seat_by_env, dtype=np.int64)
    if actor_arr.shape != focal_arr.shape:
        raise ValueError(
            f"actor_step and focal_seat_by_env must have matching shapes, got {actor_arr.shape} and {focal_arr.shape}"
        )
    focal_rows = int(np.count_nonzero(actor_arr == focal_arr))
    opponent_rows = int(actor_arr.size - focal_rows)
    counters["focal_row_count"] = int(counters.get("focal_row_count", 0)) + focal_rows
    counters["opponent_row_count"] = int(counters.get("opponent_row_count", 0)) + opponent_rows
    return focal_rows, opponent_rows


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def timeout_limits_for_env(env: Any) -> dict[str, int | None]:
    return {
        "max_decisions": optional_int(getattr(env, "max_decisions", None)),
        "max_ticks": optional_int(getattr(env, "max_ticks", None)),
        "max_no_progress_decisions": optional_int(getattr(env, "max_no_progress_decisions", None)),
    }


def merge_simulator_timing_counters(counters: dict[str, int], env: Any) -> None:
    drain_timing_counters = getattr(env, "drain_timing_counters", None)
    if not callable(drain_timing_counters):
        return
    for key, value in drain_timing_counters().items():
        counters[f"simulator_{str(key)}"] = counters.get(f"simulator_{str(key)}", 0) + int(value)


def accumulate_timeout_counters(
    *,
    counters: dict[str, int],
    batch: Any,
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


def packed_legal_views_from_step_out(step_out: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    legal_offsets = np.asarray(step_out.legal_offsets, dtype=np.uint32)
    used = 0 if legal_offsets.size == 0 else int(legal_offsets[-1])
    legal_ids = np.asarray(step_out.legal_ids, dtype=np.uint32)[:used]
    raw_meta = getattr(step_out, "legal_action_meta", None)
    legal_action_meta = None if raw_meta is None else np.asarray(raw_meta, dtype=np.uint16)[:used]
    return legal_ids, legal_offsets, legal_action_meta
