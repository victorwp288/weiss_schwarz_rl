from __future__ import annotations

import time
from typing import Any

import numpy as np

DEFAULT_ACTION_META_WIDTH = 4


def obs_numpy_dtype_for_profile(profile: str) -> np.dtype[Any]:
    normalized = str(profile).strip().lower()
    if normalized == "debug":
        return np.dtype(np.int32)
    return np.dtype(np.int16)


def shared_segment_spec(
    *,
    actor_id: int,
    slot_id: int,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    size = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    return {
        "name": f"weissrl_{actor_id}_{slot_id}_{name}_{time.time_ns()}",
        "shape": tuple(int(dim) for dim in shape),
        "dtype": dtype.str,
        "size": size,
    }


def create_shared_collector_slot_config(
    *,
    actor_id: int,
    slot_id: int = 0,
    profile: str,
    unroll_length: int,
    envs_per_actor: int,
    observation_dim: int,
    action_dim: int,
    hidden_size: int,
    layout_name: str,
    legal_action_meta_width: int = DEFAULT_ACTION_META_WIDTH,
) -> dict[str, Any]:
    rows = int(unroll_length * envs_per_actor)
    obs_dtype = obs_numpy_dtype_for_profile(profile)
    specs = {
        "obs": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="obs",
            shape=(int(unroll_length), int(envs_per_actor), int(observation_dim)),
            dtype=obs_dtype,
        ),
        "actions": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="actions",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.uint16),
        ),
        "rewards": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="rewards",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.float32),
        ),
        "terminated": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="terminated",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.bool_),
        ),
        "truncated": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="truncated",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.bool_),
        ),
        "to_play_seat": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="to_play_seat",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.int8),
        ),
        "behavior_logp": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="behavior_logp",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.float32),
        ),
        "values": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="values",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.float32),
        ),
        "bootstrap_obs": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="bootstrap_obs",
            shape=(int(envs_per_actor), int(observation_dim)),
            dtype=np.dtype(np.float32),
        ),
        "bootstrap_actor": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="bootstrap_actor",
            shape=(int(envs_per_actor),),
            dtype=np.dtype(np.int64),
        ),
        "bootstrap_value": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="bootstrap_value",
            shape=(int(envs_per_actor),),
            dtype=np.dtype(np.float32),
        ),
        "initial_hidden_state": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="initial_hidden_state",
            shape=(int(envs_per_actor), 2, int(hidden_size)),
            dtype=np.dtype(np.float32),
        ),
        "final_hidden_state": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="final_hidden_state",
            shape=(int(envs_per_actor), 2, int(hidden_size)),
            dtype=np.dtype(np.float32),
        ),
        "episode_seed": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="episode_seed",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.uint64),
        ),
        "policy_train_mask": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="policy_train_mask",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.bool_),
        ),
        "opponent_context_index": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="opponent_context_index",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.int16),
        ),
        "teacher_family": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="teacher_family",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.int32),
        ),
        "teacher_slot": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="teacher_slot",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.int32),
        ),
        "teacher_move_source": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="teacher_move_source",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.int32),
        ),
        "teacher_attack_type": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="teacher_attack_type",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.int32),
        ),
        "teacher_action": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="teacher_action",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.int32),
        ),
        "teacher_valid": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="teacher_valid",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.bool_),
        ),
        "trajectory_retention_valid": shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="trajectory_retention_valid",
            shape=(int(unroll_length), int(envs_per_actor)),
            dtype=np.dtype(np.bool_),
        ),
    }
    if str(layout_name) == "i16_legal_ids":
        specs["legal_ids"] = shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="legal_ids",
            shape=(rows * int(action_dim),),
            dtype=np.dtype(np.uint32),
        )
        specs["legal_action_meta"] = shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="legal_action_meta",
            shape=(rows * int(action_dim), int(legal_action_meta_width)),
            dtype=np.dtype(np.uint16),
        )
        specs["legal_offsets"] = shared_segment_spec(
            actor_id=actor_id, slot_id=slot_id, name="legal_offsets", shape=(rows + 1,), dtype=np.dtype(np.uint32)
        )
    else:
        specs["legal_mask"] = shared_segment_spec(
            actor_id=actor_id,
            slot_id=slot_id,
            name="legal_mask",
            shape=(int(unroll_length), int(envs_per_actor), int(action_dim)),
            dtype=np.dtype(np.bool_),
        )
    return {
        "actor_id": int(actor_id),
        "slot_id": int(slot_id),
        "layout_name": str(layout_name),
        "specs": specs,
    }


__all__ = [
    "DEFAULT_ACTION_META_WIDTH",
    "create_shared_collector_slot_config",
    "obs_numpy_dtype_for_profile",
    "shared_segment_spec",
]
