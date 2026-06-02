"""Compatibility facade for shared-memory collector transport helpers."""

from __future__ import annotations

from weiss_rl.runtime.components.shared_memory.config import (
    DEFAULT_ACTION_META_WIDTH,
    create_shared_collector_slot_config,
    obs_numpy_dtype_for_profile,
    shared_segment_spec,
)
from weiss_rl.runtime.components.shared_memory.io import (
    open_shared_collector_slot,
    read_unroll_from_shared_slot,
    shared_unroll_metadata,
    write_unroll_to_shared_slot,
)
from weiss_rl.runtime.components.shared_memory.slots import SharedCollectorSlot, SharedPendingUnroll

__all__ = [
    "DEFAULT_ACTION_META_WIDTH",
    "SharedCollectorSlot",
    "SharedPendingUnroll",
    "create_shared_collector_slot_config",
    "obs_numpy_dtype_for_profile",
    "open_shared_collector_slot",
    "read_unroll_from_shared_slot",
    "shared_segment_spec",
    "shared_unroll_metadata",
    "write_unroll_to_shared_slot",
]
