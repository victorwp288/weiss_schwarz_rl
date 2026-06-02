from __future__ import annotations

from collections.abc import Callable
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.shared_memory.slots import SharedCollectorSlot


def open_shared_collector_slot(config: dict[str, Any], *, create: bool = False) -> SharedCollectorSlot:
    specs = dict(config["specs"])
    segments: list[shared_memory.SharedMemory] = []
    arrays: dict[str, np.ndarray] = {}
    for key, spec in specs.items():
        shape = tuple(int(dim) for dim in spec["shape"])
        dtype = np.dtype(spec["dtype"])
        segment = shared_memory.SharedMemory(name=spec["name"], create=create, size=int(spec["size"]))
        segments.append(segment)
        arrays[key] = np.ndarray(shape, dtype=dtype, buffer=segment.buf)
    return SharedCollectorSlot(
        actor_id=int(config["actor_id"]),
        slot_id=int(config.get("slot_id", 0)),
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
        opponent_context_index=arrays["opponent_context_index"],
        teacher_family=arrays["teacher_family"],
        teacher_slot=arrays["teacher_slot"],
        teacher_move_source=arrays["teacher_move_source"],
        teacher_attack_type=arrays["teacher_attack_type"],
        teacher_action=arrays["teacher_action"],
        teacher_valid=arrays["teacher_valid"],
        trajectory_retention_valid=arrays["trajectory_retention_valid"],
        legal_ids=arrays.get("legal_ids"),
        legal_action_meta=arrays.get("legal_action_meta"),
        legal_offsets=arrays.get("legal_offsets"),
        legal_mask=arrays.get("legal_mask"),
        _segments=tuple(segments),
    )


def shared_unroll_metadata(unroll: Any, *, slot_id: int | None = None) -> dict[str, Any]:
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
    if slot_id is not None:
        metadata["slot_id"] = int(slot_id)
    if unroll.counters:
        metadata["counters"] = {str(key): int(value) for key, value in unroll.counters.items()}
    metadata["has_teacher_labels"] = bool(
        unroll.teacher_family is not None
        and unroll.teacher_slot is not None
        and unroll.teacher_attack_type is not None
        and unroll.teacher_valid is not None
    )
    metadata["has_teacher_move_source_label"] = bool(unroll.teacher_move_source is not None)
    metadata["has_teacher_action_label"] = bool(unroll.teacher_action is not None)
    metadata["has_trajectory_retention_label"] = bool(unroll.trajectory_retention_valid is not None)
    metadata["has_opponent_context_index"] = bool(getattr(unroll, "opponent_context_index", None) is not None)
    return metadata


def write_unroll_to_shared_slot(slot: SharedCollectorSlot, unroll: Any) -> None:
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
    if getattr(unroll, "opponent_context_index", None) is None:
        slot.opponent_context_index.fill(0)
    else:
        slot.opponent_context_index[...] = unroll.opponent_context_index
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
    if unroll.teacher_move_source is None:
        slot.teacher_move_source.fill(-1)
    else:
        slot.teacher_move_source[...] = unroll.teacher_move_source
    if unroll.teacher_action is None:
        slot.teacher_action.fill(-1)
    else:
        slot.teacher_action[...] = unroll.teacher_action
    if unroll.trajectory_retention_valid is None:
        slot.trajectory_retention_valid.fill(False)
    else:
        slot.trajectory_retention_valid[...] = unroll.trajectory_retention_valid
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


def read_unroll_from_shared_slot(
    slot: SharedCollectorSlot,
    metadata: dict[str, Any],
    *,
    unroll_type: Callable[..., Any],
) -> Any:
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
    return unroll_type(
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
        opponent_context_index=(
            np.array(slot.opponent_context_index, copy=True)
            if bool(metadata.get("has_opponent_context_index", False))
            else None
        ),
        teacher_family=(
            np.array(slot.teacher_family, copy=True) if bool(metadata.get("has_teacher_labels", False)) else None
        ),
        teacher_slot=(
            np.array(slot.teacher_slot, copy=True) if bool(metadata.get("has_teacher_labels", False)) else None
        ),
        teacher_move_source=(
            np.array(slot.teacher_move_source, copy=True)
            if bool(metadata.get("has_teacher_move_source_label", False))
            else None
        ),
        teacher_attack_type=(
            np.array(slot.teacher_attack_type, copy=True) if bool(metadata.get("has_teacher_labels", False)) else None
        ),
        teacher_action=(
            np.array(slot.teacher_action, copy=True) if bool(metadata.get("has_teacher_action_label", False)) else None
        ),
        teacher_valid=(
            np.array(slot.teacher_valid, copy=True) if bool(metadata.get("has_teacher_labels", False)) else None
        ),
        trajectory_retention_valid=(
            np.array(slot.trajectory_retention_valid, copy=True)
            if bool(metadata.get("has_trajectory_retention_label", False))
            else None
        ),
        behavior_logits=None,
        counters=(
            None
            if not isinstance(metadata.get("counters"), dict)
            else {str(key): int(value) for key, value in dict(metadata["counters"]).items()}
        ),
    )


__all__ = [
    "open_shared_collector_slot",
    "read_unroll_from_shared_slot",
    "shared_unroll_metadata",
    "write_unroll_to_shared_slot",
]
