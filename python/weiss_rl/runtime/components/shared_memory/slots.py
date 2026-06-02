from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from weiss_rl.core.legal_actions import LegalActionBatch


@dataclass(slots=True)
class SharedCollectorSlot:
    actor_id: int
    slot_id: int
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
    opponent_context_index: np.ndarray
    teacher_family: np.ndarray
    teacher_slot: np.ndarray
    teacher_move_source: np.ndarray
    teacher_attack_type: np.ndarray
    teacher_action: np.ndarray
    teacher_valid: np.ndarray
    trajectory_retention_valid: np.ndarray
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
                with suppress(FileNotFoundError):
                    segment.unlink()


@dataclass(slots=True)
class SharedPendingUnroll:
    actor_id: int
    slot_id: int
    behavior_policy_version: int
    unroll_seq: int
    unroll_hash: str
    slot: SharedCollectorSlot
    action_space: int | None
    legal_kind: str
    legal_ids_size: int
    has_legal_action_meta: bool
    has_teacher_labels: bool
    has_teacher_move_source_label: bool
    has_teacher_action_label: bool
    has_trajectory_retention_label: bool
    has_opponent_context_index: bool = False
    counters: dict[str, int] | None = None
    _legal_actions: LegalActionBatch | None = None

    @classmethod
    def from_metadata(cls, slot: SharedCollectorSlot, metadata: dict[str, Any]) -> SharedPendingUnroll:
        return cls(
            actor_id=int(metadata["actor_id"]),
            slot_id=int(metadata.get("slot_id", 0)),
            behavior_policy_version=int(metadata["behavior_policy_version"]),
            unroll_seq=int(metadata["unroll_seq"]),
            unroll_hash=str(metadata["unroll_hash"]),
            slot=slot,
            action_space=None if metadata.get("action_space") is None else int(metadata["action_space"]),
            legal_kind=str(metadata.get("legal_kind", "mask")),
            legal_ids_size=int(metadata.get("legal_ids_size", 0)),
            has_legal_action_meta=bool(metadata.get("has_legal_action_meta", False)),
            has_teacher_labels=bool(metadata.get("has_teacher_labels", False)),
            has_teacher_move_source_label=bool(metadata.get("has_teacher_move_source_label", False)),
            has_teacher_action_label=bool(metadata.get("has_teacher_action_label", False)),
            has_trajectory_retention_label=bool(metadata.get("has_trajectory_retention_label", False)),
            has_opponent_context_index=bool(metadata.get("has_opponent_context_index", False)),
            counters=(
                None
                if not isinstance(metadata.get("counters"), dict)
                else {str(key): int(value) for key, value in dict(metadata["counters"]).items()}
            ),
        )

    @property
    def obs(self) -> np.ndarray:
        return self.slot.obs

    @property
    def actions(self) -> np.ndarray:
        return self.slot.actions

    @property
    def rewards(self) -> np.ndarray:
        return self.slot.rewards

    @property
    def terminated(self) -> np.ndarray:
        return self.slot.terminated

    @property
    def truncated(self) -> np.ndarray:
        return self.slot.truncated

    @property
    def to_play_seat(self) -> np.ndarray:
        return self.slot.to_play_seat

    @property
    def behavior_logp(self) -> np.ndarray:
        return self.slot.behavior_logp

    @property
    def values(self) -> np.ndarray:
        return self.slot.values

    @property
    def bootstrap_obs(self) -> np.ndarray:
        return self.slot.bootstrap_obs

    @property
    def bootstrap_actor(self) -> np.ndarray:
        return self.slot.bootstrap_actor

    @property
    def bootstrap_value(self) -> np.ndarray:
        return self.slot.bootstrap_value

    @property
    def initial_hidden_state(self) -> np.ndarray:
        return self.slot.initial_hidden_state

    @property
    def final_hidden_state(self) -> np.ndarray:
        return self.slot.final_hidden_state

    @property
    def episode_seed(self) -> np.ndarray:
        return self.slot.episode_seed

    @property
    def policy_train_mask(self) -> np.ndarray:
        return self.slot.policy_train_mask

    @property
    def opponent_context_index(self) -> np.ndarray | None:
        return self.slot.opponent_context_index if self.has_opponent_context_index else None

    @property
    def teacher_family(self) -> np.ndarray | None:
        return self.slot.teacher_family if self.has_teacher_labels else None

    @property
    def teacher_slot(self) -> np.ndarray | None:
        return self.slot.teacher_slot if self.has_teacher_labels else None

    @property
    def teacher_move_source(self) -> np.ndarray | None:
        return self.slot.teacher_move_source if self.has_teacher_move_source_label else None

    @property
    def teacher_attack_type(self) -> np.ndarray | None:
        return self.slot.teacher_attack_type if self.has_teacher_labels else None

    @property
    def teacher_action(self) -> np.ndarray | None:
        return self.slot.teacher_action if self.has_teacher_action_label else None

    @property
    def teacher_valid(self) -> np.ndarray | None:
        return self.slot.teacher_valid if self.has_teacher_labels else None

    @property
    def trajectory_retention_valid(self) -> np.ndarray | None:
        return self.slot.trajectory_retention_valid if self.has_trajectory_retention_label else None

    @property
    def behavior_logits(self) -> None:
        return None

    @property
    def legal_actions(self) -> LegalActionBatch:
        cached = self._legal_actions
        if cached is not None:
            return cached
        if self.legal_kind == "packed":
            assert self.slot.legal_ids is not None and self.slot.legal_offsets is not None
            cached = LegalActionBatch.from_packed(
                self.slot.legal_ids[: self.legal_ids_size],
                self.slot.legal_offsets,
                meta=(
                    None
                    if self.slot.legal_action_meta is None or not self.has_legal_action_meta
                    else self.slot.legal_action_meta[: self.legal_ids_size]
                ),
                action_space=self.action_space,
            )
        else:
            assert self.slot.legal_mask is not None
            cached = LegalActionBatch.from_mask(self.slot.legal_mask, action_space=self.action_space)
        self._legal_actions = cached
        return cached


__all__ = ["SharedCollectorSlot", "SharedPendingUnroll"]
