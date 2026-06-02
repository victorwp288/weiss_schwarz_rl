"""Batch packing and episode-identity helpers for decision-boundary environments."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.envs.decision_batch import DecisionBoundaryBatch, EpisodeIdentitySource, LegalMode

_U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix_u64(values: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer with simulator-parity seeding step.

    Keep this bit-for-bit aligned with ``weiss_sim.runner._mix_u64()`` so the
    training-side fallback episode-key derivation matches the simulator exactly.
    """

    mixed = np.asarray(values, dtype=np.uint64).copy()
    mixed = (mixed + np.uint64(0x9E3779B97F4A7C15)) & _U64_MASK
    mixed ^= mixed >> np.uint64(30)
    mixed = (mixed * np.uint64(0xBF58476D1CE4E5B9)) & _U64_MASK
    mixed ^= mixed >> np.uint64(27)
    mixed = (mixed * np.uint64(0x94D049BB133111EB)) & _U64_MASK
    mixed ^= mixed >> np.uint64(31)
    return mixed & _U64_MASK


def _derive_episode_key(episode_seed: np.ndarray, episode_index: np.ndarray, env_index: np.ndarray) -> np.ndarray:
    combo = (np.asarray(episode_index, dtype=np.uint64) << np.uint64(32)) ^ np.asarray(env_index, dtype=np.uint64)
    return _mix_u64(np.asarray(episode_seed, dtype=np.uint64) ^ _mix_u64(combo))


def _batch_episode_identity(
    step: Any,
    *,
    pool: Any | None,
    num_envs: int,
    copy_arrays: bool,
) -> tuple[np.ndarray, np.ndarray, EpisodeIdentitySource]:
    episode_seed = getattr(step, "episode_seed", None)
    episode_key = getattr(step, "episode_key", None)
    if episode_seed is not None and episode_key is not None:
        return (
            _array_view_or_copy(episode_seed, dtype=np.uint64, copy_arrays=copy_arrays),
            _array_view_or_copy(episode_key, dtype=np.uint64, copy_arrays=copy_arrays),
            "simulator",
        )

    if pool is None:
        zeros = np.zeros((num_envs,), dtype=np.uint64)
        return zeros, zeros.copy(), "missing"

    pool_episode_seed = getattr(pool, "episode_seed_batch", None)
    if not callable(pool_episode_seed):
        zeros = np.zeros((num_envs,), dtype=np.uint64)
        return zeros, zeros.copy(), "missing"

    episode_seed_array = _array_view_or_copy(pool_episode_seed(), dtype=np.uint64, copy_arrays=copy_arrays)
    pool_episode_index = getattr(pool, "episode_index_batch", None)
    pool_env_index = getattr(pool, "env_index_batch", None)
    if callable(pool_episode_index) and callable(pool_env_index):
        episode_key_array = _derive_episode_key(
            episode_seed_array,
            np.asarray(pool_episode_index(), dtype=np.uint64),
            np.asarray(pool_env_index(), dtype=np.uint64),
        )
        return episode_seed_array, np.asarray(episode_key_array, dtype=np.uint64), "derived"

    if episode_key is not None:
        return (
            episode_seed_array,
            _array_view_or_copy(episode_key, dtype=np.uint64, copy_arrays=copy_arrays),
            "simulator",
        )

    return episode_seed_array, np.zeros((num_envs,), dtype=np.uint64), "pool_seed_only"


def _array_view_or_copy(
    values: Any,
    *,
    dtype: np.dtype[Any] | type[Any] | None = None,
    copy_arrays: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    return np.array(array, copy=True) if copy_arrays else array


def _packed_legal_ids_prefix(legal_ids: Any, legal_offsets: Any, *, copy_arrays: bool) -> np.ndarray:
    ids = np.asarray(legal_ids)
    offsets = np.asarray(legal_offsets)
    used = 0 if offsets.size == 0 else int(offsets[-1])
    if used < 0 or used > ids.shape[0]:
        raise RuntimeError(f"packed legal_ids prefix out of bounds: used={used}, capacity={ids.shape[0]}")
    prefix = ids[:used]
    return np.array(prefix, copy=True) if copy_arrays else prefix


def _packed_legal_action_meta_prefix(
    legal_action_meta: Any | None,
    legal_offsets: Any,
    *,
    copy_arrays: bool,
) -> np.ndarray | None:
    if legal_action_meta is None:
        return None
    meta = np.asarray(legal_action_meta)
    offsets = np.asarray(legal_offsets)
    if meta.ndim != 2:
        raise RuntimeError(f"packed legal_action_meta must be 2D, got {meta.shape}")
    used = 0 if offsets.size == 0 else int(offsets[-1])
    if used < 0 or used > meta.shape[0]:
        raise RuntimeError(f"packed legal_action_meta prefix out of bounds: used={used}, capacity={meta.shape[0]}")
    prefix = meta[:used]
    return np.array(prefix, copy=True) if copy_arrays else prefix


def _step_counter_array(
    step: Any,
    *,
    field_name: str,
    num_envs: int,
    copy_arrays: bool,
) -> np.ndarray:
    values = getattr(step, field_name, None)
    if values is None:
        return np.zeros((num_envs,), dtype=np.uint32)
    array = _array_view_or_copy(values, dtype=np.uint32, copy_arrays=copy_arrays)
    if array.ndim != 1 or int(array.shape[0]) != num_envs:
        raise ValueError(f"{field_name} must have shape ({num_envs},)")
    return array


def _step_flag_array(
    step: Any,
    *,
    field_name: str,
    num_envs: int,
    copy_arrays: bool,
) -> np.ndarray:
    values = getattr(step, field_name, None)
    if values is None:
        return np.zeros((num_envs,), dtype=np.bool_)
    array = _array_view_or_copy(values, dtype=np.bool_, copy_arrays=copy_arrays)
    if array.ndim != 1 or int(array.shape[0]) != num_envs:
        raise ValueError(f"{field_name} must have shape ({num_envs},)")
    return array


def _pack_batch(
    step: Any,
    *,
    legality: LegalMode,
    pool: Any | None = None,
    copy_arrays: bool = True,
) -> DecisionBoundaryBatch:
    actor = _array_view_or_copy(step.actor, copy_arrays=copy_arrays)
    num_envs = int(actor.shape[0])
    decision_count = _step_counter_array(
        step,
        field_name="decision_count",
        num_envs=num_envs,
        copy_arrays=copy_arrays,
    )
    tick_count = _step_counter_array(
        step,
        field_name="tick_count",
        num_envs=num_envs,
        copy_arrays=copy_arrays,
    )
    no_progress_count = _step_counter_array(
        step,
        field_name="no_progress_count",
        num_envs=num_envs,
        copy_arrays=copy_arrays,
    )
    main_move_action = _step_flag_array(
        step,
        field_name="main_move_action",
        num_envs=num_envs,
        copy_arrays=copy_arrays,
    )
    main_pass_action = _step_flag_array(
        step,
        field_name="main_pass_action",
        num_envs=num_envs,
        copy_arrays=copy_arrays,
    )
    episode_seed, episode_key, episode_identity_source = _batch_episode_identity(
        step,
        pool=pool,
        num_envs=num_envs,
        copy_arrays=copy_arrays,
    )
    if legality == "mask":
        mask = getattr(step, "masks", None)
        if mask is None:
            raise RuntimeError("mask layout did not return masks")
        mask_action_space = int(np.asarray(mask).shape[-1])
        return DecisionBoundaryBatch(
            obs=_array_view_or_copy(step.obs, copy_arrays=copy_arrays),
            reward=_array_view_or_copy(step.rewards, copy_arrays=copy_arrays),
            terminated=_array_view_or_copy(step.terminated, copy_arrays=copy_arrays),
            truncated=_array_view_or_copy(step.truncated, copy_arrays=copy_arrays),
            to_play=np.array(actor, copy=True) if copy_arrays else actor,
            actor=actor,
            decision_kind=_array_view_or_copy(step.decision_kind, copy_arrays=copy_arrays),
            decision_id=_array_view_or_copy(step.decision_id, copy_arrays=copy_arrays),
            engine_status=_array_view_or_copy(step.engine_status, copy_arrays=copy_arrays),
            decision_count=decision_count,
            tick_count=tick_count,
            episode_seed=episode_seed,
            episode_key=episode_key,
            episode_identity_source=episode_identity_source,
            action_space=mask_action_space,
            no_progress_count=no_progress_count,
            main_move_action=main_move_action,
            main_pass_action=main_pass_action,
            mask=_array_view_or_copy(mask, copy_arrays=copy_arrays),
        )

    legal_ids = getattr(step, "legal_ids", None)
    legal_offsets = getattr(step, "legal_offsets", None)
    if legal_ids is None or legal_offsets is None:
        raise RuntimeError("ids_offsets layout did not return legal_ids/legal_offsets")
    legal_action_meta = getattr(step, "legal_action_meta", None)
    return DecisionBoundaryBatch(
        obs=_array_view_or_copy(step.obs, copy_arrays=copy_arrays),
        reward=_array_view_or_copy(step.rewards, copy_arrays=copy_arrays),
        terminated=_array_view_or_copy(step.terminated, copy_arrays=copy_arrays),
        truncated=_array_view_or_copy(step.truncated, copy_arrays=copy_arrays),
        to_play=np.array(actor, copy=True) if copy_arrays else actor,
        actor=actor,
        decision_kind=_array_view_or_copy(step.decision_kind, copy_arrays=copy_arrays),
        decision_id=_array_view_or_copy(step.decision_id, copy_arrays=copy_arrays),
        engine_status=_array_view_or_copy(step.engine_status, copy_arrays=copy_arrays),
        decision_count=decision_count,
        tick_count=tick_count,
        episode_seed=episode_seed,
        episode_key=episode_key,
        episode_identity_source=episode_identity_source,
        action_space=int(pool.action_space) if pool is not None and hasattr(pool, "action_space") else None,
        no_progress_count=no_progress_count,
        main_move_action=main_move_action,
        main_pass_action=main_pass_action,
        ids_offsets=(
            _packed_legal_ids_prefix(legal_ids, legal_offsets, copy_arrays=copy_arrays),
            _array_view_or_copy(legal_offsets, copy_arrays=copy_arrays),
        ),
        legal_action_meta=_packed_legal_action_meta_prefix(
            legal_action_meta,
            legal_offsets,
            copy_arrays=copy_arrays,
        ),
    )


__all__ = [
    "_array_view_or_copy",
    "_batch_episode_identity",
    "_derive_episode_key",
    "_pack_batch",
]
