"""Runtime-specific shared collector transport facades."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.runtime.components.shared_memory.config import (
    DEFAULT_ACTION_META_WIDTH,
)
from weiss_rl.runtime.components.shared_memory.config import (
    create_shared_collector_slot_config as _create_shared_collector_slot_config,
)
from weiss_rl.runtime.components.shared_memory.config import (
    obs_numpy_dtype_for_profile as _obs_numpy_dtype_for_profile,
)
from weiss_rl.runtime.components.shared_memory.config import (
    shared_segment_spec as _shared_segment_spec,
)
from weiss_rl.runtime.components.shared_memory.io import (
    open_shared_collector_slot as _open_shared_collector_slot,
)
from weiss_rl.runtime.components.shared_memory.io import (
    read_unroll_from_shared_slot as _read_unroll_from_shared_slot,
)
from weiss_rl.runtime.components.shared_memory.io import (
    shared_unroll_metadata as _shared_unroll_metadata,
)
from weiss_rl.runtime.components.shared_memory.io import (
    write_unroll_to_shared_slot as _write_unroll_to_shared_slot,
)
from weiss_rl.runtime.components.shared_memory.slots import SharedCollectorSlot
from weiss_rl.runtime.components.types import RuntimeUnroll


def obs_numpy_dtype_for_profile(profile: str) -> np.dtype[Any]:
    return _obs_numpy_dtype_for_profile(profile)


def shared_segment_spec(
    *,
    actor_id: int,
    slot_id: int,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    return _shared_segment_spec(actor_id=actor_id, slot_id=slot_id, name=name, shape=shape, dtype=dtype)


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
    return _create_shared_collector_slot_config(
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
    return _open_shared_collector_slot(config, create=create)


def shared_unroll_metadata(unroll: RuntimeUnroll, *, slot_id: int | None = None) -> dict[str, Any]:
    return _shared_unroll_metadata(unroll, slot_id=slot_id)


def write_unroll_to_shared_slot(slot: SharedCollectorSlot, unroll: RuntimeUnroll) -> None:
    _write_unroll_to_shared_slot(slot, unroll)


def read_unroll_from_shared_slot(slot: SharedCollectorSlot, metadata: dict[str, Any]) -> RuntimeUnroll:
    return _read_unroll_from_shared_slot(slot, metadata, unroll_type=RuntimeUnroll)
