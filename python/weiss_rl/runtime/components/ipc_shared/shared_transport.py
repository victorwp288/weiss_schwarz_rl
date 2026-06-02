"""Runtime-specific shared collector transport facades."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.runtime.components import shared as runtime_shared
from weiss_rl.runtime.components.types import RuntimeUnroll

SharedCollectorSlot = runtime_shared.SharedCollectorSlot


def obs_numpy_dtype_for_profile(profile: str) -> np.dtype[Any]:
    return runtime_shared.obs_numpy_dtype_for_profile(profile)


def shared_segment_spec(
    *,
    actor_id: int,
    slot_id: int,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    return runtime_shared.shared_segment_spec(actor_id=actor_id, slot_id=slot_id, name=name, shape=shape, dtype=dtype)


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
    legal_action_meta_width: int = runtime_shared.DEFAULT_ACTION_META_WIDTH,
) -> dict[str, Any]:
    return runtime_shared.create_shared_collector_slot_config(
        actor_id=actor_id,
        slot_id=slot_id,
        profile=profile,
        unroll_length=unroll_length,
        envs_per_actor=envs_per_actor,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        layout_name=layout_name,
        legal_action_meta_width=legal_action_meta_width,
    )


def open_shared_collector_slot(config: dict[str, Any], *, create: bool = False) -> SharedCollectorSlot:
    return runtime_shared.open_shared_collector_slot(config, create=create)


def shared_unroll_metadata(unroll: RuntimeUnroll, *, slot_id: int | None = None) -> dict[str, Any]:
    return runtime_shared.shared_unroll_metadata(unroll, slot_id=slot_id)


def write_unroll_to_shared_slot(slot: SharedCollectorSlot, unroll: RuntimeUnroll) -> None:
    runtime_shared.write_unroll_to_shared_slot(slot, unroll)


def read_unroll_from_shared_slot(slot: SharedCollectorSlot, metadata: dict[str, Any]) -> RuntimeUnroll:
    return runtime_shared.read_unroll_from_shared_slot(slot, metadata, unroll_type=RuntimeUnroll)
