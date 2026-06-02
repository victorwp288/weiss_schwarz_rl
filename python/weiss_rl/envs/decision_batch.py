"""Decision-boundary batch types and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

LegalMode = Literal["mask", "ids_offsets"]
EngineStatusPolicy = Literal["best_effort_reset", "hard_fail", "passthrough"]
CopyCasting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
EpisodeIdentitySource = Literal["simulator", "derived", "pool_seed_only", "missing"]

_VALID_ENGINE_STATUS_POLICIES = frozenset({"best_effort_reset", "hard_fail", "passthrough"})
_VALID_EPISODE_IDENTITY_SOURCES = frozenset({"simulator", "derived", "pool_seed_only", "missing"})


@dataclass(slots=True)
class EngineStatusCounters:
    """Training-side counters for simulator engine-status faults."""

    fault_rows: int = 0
    best_effort_reset_rows: int = 0


@dataclass(frozen=True, slots=True)
class DecisionBoundaryBatch:
    """Stable batch object returned by `DecisionBoundaryEnv.reset()` and `step()`."""

    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play: np.ndarray
    actor: np.ndarray
    decision_id: np.ndarray
    engine_status: np.ndarray
    decision_count: np.ndarray
    tick_count: np.ndarray
    episode_seed: np.ndarray
    episode_key: np.ndarray
    decision_kind: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    no_progress_count: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint32))
    main_move_action: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.bool_))
    main_pass_action: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.bool_))
    episode_identity_source: EpisodeIdentitySource = "missing"
    action_space: int | None = None
    mask: np.ndarray | None = None
    ids_offsets: tuple[np.ndarray, np.ndarray] | None = None
    legal_action_meta: np.ndarray | None = None

    def __post_init__(self) -> None:
        num_envs = _require_obs_rows(self.obs)
        _require_vector(self.reward, "reward", num_envs)
        _require_vector(self.terminated, "terminated", num_envs)
        _require_vector(self.truncated, "truncated", num_envs)
        _require_vector(self.to_play, "to_play", num_envs)
        _require_vector(self.actor, "actor", num_envs)
        if int(self.decision_kind.shape[0]) == 0 and self.decision_kind.ndim == 1:
            object.__setattr__(self, "decision_kind", np.zeros((num_envs,), dtype=np.int32))
        _require_vector(self.decision_kind, "decision_kind", num_envs)
        _require_vector(self.decision_id, "decision_id", num_envs)
        _require_vector(self.engine_status, "engine_status", num_envs)
        _require_vector(self.decision_count, "decision_count", num_envs)
        _require_vector(self.tick_count, "tick_count", num_envs)
        _require_vector(self.episode_seed, "episode_seed", num_envs)
        _require_vector(self.episode_key, "episode_key", num_envs)
        if int(self.no_progress_count.shape[0]) == 0 and self.no_progress_count.ndim == 1:
            object.__setattr__(self, "no_progress_count", np.zeros((num_envs,), dtype=np.uint32))
        _require_vector(self.no_progress_count, "no_progress_count", num_envs)
        if int(self.main_move_action.shape[0]) == 0 and self.main_move_action.ndim == 1:
            object.__setattr__(self, "main_move_action", np.zeros((num_envs,), dtype=np.bool_))
        if int(self.main_pass_action.shape[0]) == 0 and self.main_pass_action.ndim == 1:
            object.__setattr__(self, "main_pass_action", np.zeros((num_envs,), dtype=np.bool_))
        _require_vector(self.main_move_action, "main_move_action", num_envs)
        _require_vector(self.main_pass_action, "main_pass_action", num_envs)
        if self.episode_identity_source not in _VALID_EPISODE_IDENTITY_SOURCES:
            expected = ", ".join(sorted(_VALID_EPISODE_IDENTITY_SOURCES))
            raise ValueError(f"episode_identity_source must be one of: {expected}")
        _require_legality(self.mask, self.ids_offsets, self.legal_action_meta, num_envs)
        if self.action_space is not None:
            action_space = int(self.action_space)
            if action_space <= 0:
                raise ValueError("action_space must be positive")
            if self.mask is not None:
                mask_action_space = int(self.mask.shape[-1])
                if mask_action_space != action_space:
                    raise ValueError(f"action_space mismatch: expected {action_space}, got {mask_action_space}")
            elif self.ids_offsets is not None:
                legal_ids, _legal_offsets = self.ids_offsets
                if legal_ids.size and np.any(np.asarray(legal_ids, dtype=np.int64) >= action_space):
                    raise ValueError(f"ids_offsets legal ids must be in [0, {action_space})")
            object.__setattr__(self, "action_space", action_space)
        elif self.mask is not None:
            object.__setattr__(self, "action_space", int(self.mask.shape[-1]))

    @property
    def num_envs(self) -> int:
        return int(self.reward.shape[0])


def _normalize_legality(legality: str) -> LegalMode:
    if legality == "mask":
        return "mask"
    if legality == "ids_offsets":
        return "ids_offsets"
    raise ValueError("legality must be 'mask' or 'ids_offsets'")


def _normalize_engine_status_policy(policy: str) -> EngineStatusPolicy:
    if policy not in _VALID_ENGINE_STATUS_POLICIES:
        expected = ", ".join(sorted(_VALID_ENGINE_STATUS_POLICIES))
        raise ValueError(f"engine_status_policy must be one of: {expected}")
    return cast(EngineStatusPolicy, policy)


def _engine_status_codes(engine_status: Any, *, num_envs: int | None = None) -> np.ndarray:
    codes = np.ravel(np.asarray(engine_status, dtype=np.uint8))
    if num_envs is not None and int(codes.shape[0]) != num_envs:
        raise ValueError(f"engine_status must have shape ({num_envs},)")
    return np.ascontiguousarray(codes, dtype=np.uint8)


def _count_fault_rows(engine_status: Any) -> int:
    return int(np.count_nonzero(_engine_status_codes(engine_status) != 0))


def _require_obs_rows(obs: np.ndarray) -> int:
    if obs.ndim == 0:
        raise ValueError("obs must have a batch dimension")
    return int(obs.shape[0])


def _require_vector(values: np.ndarray, name: str, num_envs: int) -> None:
    if values.ndim != 1 or int(values.shape[0]) != num_envs:
        raise ValueError(f"{name} must have shape ({num_envs},)")


def _require_legality(
    mask: np.ndarray | None,
    ids_offsets: tuple[np.ndarray, np.ndarray] | None,
    legal_action_meta: np.ndarray | None,
    num_envs: int,
) -> None:
    has_mask = mask is not None
    has_ids = ids_offsets is not None
    if has_mask == has_ids:
        raise ValueError("exactly one legal representation must be present: mask or ids_offsets")

    if mask is not None:
        if mask.ndim != 2 or int(mask.shape[0]) != num_envs:
            raise ValueError(f"mask must have shape ({num_envs}, action_space)")
        return

    assert ids_offsets is not None
    legal_ids, legal_offsets = ids_offsets
    if legal_ids.ndim != 1:
        raise ValueError("ids_offsets legal_ids must be 1D")
    if legal_offsets.ndim != 1 or int(legal_offsets.shape[0]) != num_envs + 1:
        raise ValueError(f"ids_offsets legal_offsets must have shape ({num_envs + 1},)")
    if legal_action_meta is not None:
        if legal_action_meta.ndim != 2:
            raise ValueError("legal_action_meta must be 2D when present")
        if int(legal_action_meta.shape[0]) != int(legal_ids.shape[0]):
            raise ValueError(
                "legal_action_meta must align 1:1 with packed legal ids: "
                f"expected first dim {legal_ids.shape[0]}, got {legal_action_meta.shape[0]}"
            )


__all__ = [
    "CopyCasting",
    "DecisionBoundaryBatch",
    "EngineStatusCounters",
    "EngineStatusPolicy",
    "EpisodeIdentitySource",
    "LegalMode",
    "_count_fault_rows",
    "_engine_status_codes",
    "_normalize_engine_status_policy",
    "_normalize_legality",
]
