"""
Fixed-shape trajectory buffers for actor-to-learner handoff.
M3-04: replace list-backed storage with array-backed T x N unroll buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weiss_rl.core.masking import assert_strictly_increasing_legal_ids
from weiss_rl.trajectory.schema import LegalRepr, TrajectoryChunkMeta


def _require_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")


def _legal_ids_dtype(action_space: int) -> np.dtype[np.unsignedinteger]:
    if action_space <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if action_space <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    raise ValueError("action_space is too large for packed legal ids")


def _coerce_legal_ids(legal_ids: np.ndarray, *, action_space: int) -> np.ndarray:
    legal_ids_arr = np.asarray(legal_ids)
    if legal_ids_arr.ndim != 1:
        raise ValueError("legal_ids must be 1D packed")
    if legal_ids_arr.dtype == np.bool_ or not np.issubdtype(legal_ids_arr.dtype, np.integer):
        raise ValueError("legal_ids must be an integer array")

    signed = legal_ids_arr.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("legal_ids must be >= 0")
    if np.any(signed >= action_space):
        raise ValueError(f"legal_ids must be < action_space ({action_space})")
    return signed


def _validate_packed_legal_ids(legal_ids: np.ndarray, legal_offsets: np.ndarray, *, action_space: int) -> np.ndarray:
    signed = _coerce_legal_ids(legal_ids, action_space=action_space)
    if int(legal_offsets[-1]) != int(signed.shape[0]):
        raise ValueError("legal_offsets[-1] must equal len(legal_ids)")
    for row_index in range(int(legal_offsets.shape[0]) - 1):
        start = int(legal_offsets[row_index])
        end = int(legal_offsets[row_index + 1])
        assert_strictly_increasing_legal_ids(signed[start:end])
    return signed


@dataclass(slots=True)
class UnrollBatch:
    """Array-backed unroll with shape (T, N, ...)."""

    T: int
    N: int
    obs_len: int
    action_space: int
    meta: TrajectoryChunkMeta

    # Required per-step stored fields (canonical §7.2)
    obs: np.ndarray  # (T, N, obs_len) int16/int32
    to_play_seat: np.ndarray  # (T, N) int8
    decision_id: np.ndarray  # (T, N) int32
    action: np.ndarray  # (T, N) uint32
    reward: np.ndarray  # (T, N) float32
    terminated: np.ndarray  # (T, N) bool
    truncated: np.ndarray  # (T, N) bool
    engine_status: np.ndarray  # (T, N) int32
    episode_seed: np.ndarray  # (T, N) uint64
    episode_key: np.ndarray  # (T, N) uint64 (or bytes later, but keep uint64 for now)
    behavior_logp: np.ndarray  # (T, N) float32

    # Optional time-scale disambiguation (M1-11 / M2-12)
    k_raw_decisions: np.ndarray | None = None  # (T, N) int16/int32

    # Legality storage (one of these, depending on meta.legal_repr)
    legal_mask: np.ndarray | None = None  # (T, N, A) uint8/bool
    legal_ids: np.ndarray | None = None  # (L,) uint16/uint32 packed
    legal_action_meta: np.ndarray | None = None  # (L, M) uint16 packed, aligned with legal_ids
    legal_offsets: np.ndarray | None = None  # (T, N+1) uint32 offsets per (t,row)

    _write_t: int = 0  # internal cursor
    _packed_legal_write: int = 0  # next free index in legal_ids


def allocate_unroll_batch(
    *,
    T: int,
    N: int,
    obs_len: int,
    action_space: int,
    obs_dtype: str,
    legal_repr: LegalRepr,
    meta: TrajectoryChunkMeta | None = None,
    # For ids_offsets, you either:
    #  - preallocate a max packed capacity (recommended for fixed footprint), or
    #  - start empty and append, which defeats the point of M3-04.
    max_packed_legal: int = 0,
    legal_action_meta_width: int = 4,
) -> UnrollBatch:
    if T <= 0 or N <= 0:
        raise ValueError("T and N must be > 0")
    if obs_len <= 0 or action_space <= 0:
        raise ValueError("obs_len and action_space must be > 0")

    if meta is None:
        meta = TrajectoryChunkMeta()
    meta.obs_dtype = obs_dtype
    meta.legal_repr = legal_repr

    if obs_dtype == "i16":
        obs_arr = np.zeros((T, N, obs_len), dtype=np.int16)
    elif obs_dtype == "i32":
        obs_arr = np.zeros((T, N, obs_len), dtype=np.int32)
    else:
        raise ValueError("obs_dtype must be 'i16' or 'i32'")

    batch = UnrollBatch(
        T=T,
        N=N,
        obs_len=obs_len,
        action_space=action_space,
        meta=meta,
        obs=obs_arr,
        to_play_seat=np.zeros((T, N), dtype=np.int8),
        decision_id=np.zeros((T, N), dtype=np.int32),
        action=np.zeros((T, N), dtype=np.uint32),
        reward=np.zeros((T, N), dtype=np.float32),
        terminated=np.zeros((T, N), dtype=np.bool_),
        truncated=np.zeros((T, N), dtype=np.bool_),
        engine_status=np.zeros((T, N), dtype=np.int32),
        episode_seed=np.zeros((T, N), dtype=np.uint64),
        episode_key=np.zeros((T, N), dtype=np.uint64),
        behavior_logp=np.zeros((T, N), dtype=np.float32),
        _write_t=0,
        _packed_legal_write=0,
    )

    if legal_repr == "mask":
        batch.legal_mask = np.zeros((T, N, action_space), dtype=np.uint8)
    elif legal_repr == "ids_offsets":
        if max_packed_legal <= 0:
            raise ValueError("max_packed_legal must be set for legal_repr='ids_offsets'")
        batch.legal_ids = np.zeros((max_packed_legal,), dtype=_legal_ids_dtype(action_space))
        batch.legal_action_meta = np.full(
            (max_packed_legal, int(legal_action_meta_width)),
            np.iinfo(np.uint16).max,
            dtype=np.uint16,
        )
        batch.legal_offsets = np.zeros((T, N + 1), dtype=np.uint32)
    elif legal_repr == "none":
        pass
    else:
        raise ValueError("legal_repr must be one of: ids_offsets, mask, none")

    return batch


def write_step_mask(
    batch: UnrollBatch,
    *,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    to_play_seat: np.ndarray,
    decision_id: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    engine_status: np.ndarray,
    episode_seed: np.ndarray,
    episode_key: np.ndarray,
    behavior_logp: np.ndarray,
    k_raw_decisions: np.ndarray | None = None,
) -> None:
    """Write a single timestep t for mask legality layout."""
    if batch._write_t >= batch.T:
        raise RuntimeError("unroll buffer overflow: too many steps written")

    t = batch._write_t

    _require_shape("obs", np.asarray(obs), (batch.N, batch.obs_len))
    _require_shape("legal_mask", np.asarray(legal_mask), (batch.N, batch.action_space))
    _require_shape("to_play_seat", np.asarray(to_play_seat), (batch.N,))
    _require_shape("decision_id", np.asarray(decision_id), (batch.N,))
    _require_shape("action", np.asarray(action), (batch.N,))
    _require_shape("reward", np.asarray(reward), (batch.N,))
    _require_shape("terminated", np.asarray(terminated), (batch.N,))
    _require_shape("truncated", np.asarray(truncated), (batch.N,))
    _require_shape("engine_status", np.asarray(engine_status), (batch.N,))
    _require_shape("episode_seed", np.asarray(episode_seed), (batch.N,))
    _require_shape("episode_key", np.asarray(episode_key), (batch.N,))
    _require_shape("behavior_logp", np.asarray(behavior_logp), (batch.N,))

    if batch.legal_mask is None:
        raise RuntimeError("batch.legal_mask is None but write_step_mask was called")
    if batch.meta.legal_repr != "mask":
        raise RuntimeError(f"write_step_mask called for legal_repr={batch.meta.legal_repr}")

    batch.obs[t] = obs
    batch.legal_mask[t] = legal_mask
    batch.to_play_seat[t] = to_play_seat
    batch.decision_id[t] = decision_id
    batch.action[t] = action
    batch.reward[t] = reward
    batch.terminated[t] = terminated
    batch.truncated[t] = truncated
    batch.engine_status[t] = engine_status
    batch.episode_seed[t] = episode_seed
    batch.episode_key[t] = episode_key
    batch.behavior_logp[t] = behavior_logp

    if k_raw_decisions is not None:
        if batch.k_raw_decisions is None:
            batch.k_raw_decisions = np.zeros((batch.T, batch.N), dtype=np.int16)
        _require_shape("k_raw_decisions", np.asarray(k_raw_decisions), (batch.N,))
        batch.k_raw_decisions[t] = k_raw_decisions

    batch._write_t += 1


def write_step_ids_offsets(
    batch: UnrollBatch,
    *,
    obs: np.ndarray,
    legal_ids: np.ndarray,
    legal_action_meta: np.ndarray | None = None,
    legal_offsets: np.ndarray,
    to_play_seat: np.ndarray,
    decision_id: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    engine_status: np.ndarray,
    episode_seed: np.ndarray,
    episode_key: np.ndarray,
    behavior_logp: np.ndarray,
    k_raw_decisions: np.ndarray | None = None,
) -> None:
    """Write a single timestep t for packed ids+offsets layout."""
    if batch._write_t >= batch.T:
        raise RuntimeError("unroll buffer overflow: too many steps written")

    t = batch._write_t

    _require_shape("obs", np.asarray(obs), (batch.N, batch.obs_len))
    _require_shape("legal_offsets", np.asarray(legal_offsets), (batch.N + 1,))
    _require_shape("to_play_seat", np.asarray(to_play_seat), (batch.N,))
    _require_shape("decision_id", np.asarray(decision_id), (batch.N,))
    _require_shape("action", np.asarray(action), (batch.N,))
    _require_shape("reward", np.asarray(reward), (batch.N,))
    _require_shape("terminated", np.asarray(terminated), (batch.N,))
    _require_shape("truncated", np.asarray(truncated), (batch.N,))
    _require_shape("engine_status", np.asarray(engine_status), (batch.N,))
    _require_shape("episode_seed", np.asarray(episode_seed), (batch.N,))
    _require_shape("episode_key", np.asarray(episode_key), (batch.N,))
    _require_shape("behavior_logp", np.asarray(behavior_logp), (batch.N,))

    if batch.legal_ids is None or batch.legal_offsets is None:
        raise RuntimeError("batch legal_ids/offsets are None but write_step_ids_offsets was called")
    if batch.meta.legal_repr != "ids_offsets":
        raise RuntimeError(f"write_step_ids_offsets called for legal_repr={batch.meta.legal_repr}")

    legal_offsets_arr = np.asarray(legal_offsets, dtype=np.int64)
    if int(legal_offsets_arr[0]) != 0:
        raise ValueError("legal_offsets must start at 0")
    if np.any(legal_offsets_arr[1:] < legal_offsets_arr[:-1]):
        raise ValueError("legal_offsets must be nondecreasing")
    legal_ids_arr = _validate_packed_legal_ids(
        legal_ids,
        legal_offsets_arr,
        action_space=batch.action_space,
    )
    if legal_action_meta is not None:
        if batch.legal_action_meta is None:
            raise RuntimeError("batch legal_action_meta is None but packed legal action meta was provided")
        legal_action_meta_arr = np.asarray(legal_action_meta, dtype=np.uint16)
        if legal_action_meta_arr.ndim != 2:
            raise ValueError("legal_action_meta must be 2D")
        if int(legal_action_meta_arr.shape[0]) != int(legal_ids_arr.shape[0]):
            raise ValueError("legal_action_meta must align 1:1 with legal_ids")
        if int(legal_action_meta_arr.shape[1]) != int(batch.legal_action_meta.shape[1]):
            raise ValueError(
                "legal_action_meta width mismatch: "
                f"expected {batch.legal_action_meta.shape[1]}, got {legal_action_meta_arr.shape[1]}"
            )
    else:
        legal_action_meta_arr = None
    if int(legal_offsets_arr[-1]) != int(legal_ids_arr.shape[0]):
        raise ValueError("legal_offsets[-1] must equal len(legal_ids)")

    seg_len = int(legal_offsets_arr[-1])
    # Copy packed ids into preallocated legal_ids and record offsets for this timestep.
    # We store offsets per row relative to the packed segment start for this timestep.
    packed_capacity = batch.legal_ids.shape[0]

    # Compute base pointer for timestep t in the packed array by reserving contiguous space:
    # Append this timestep's packed legal_ids into the global preallocated buffer and
    # store per-row offsets shifted by the segment base pointer.
    base = int(batch._packed_legal_write)

    end = base + seg_len
    if end > packed_capacity:
        raise RuntimeError("packed legal_ids capacity exceeded; increase max_packed_legal")
    batch._packed_legal_write = end

    batch.legal_ids[base:end] = legal_ids_arr[:seg_len].astype(batch.legal_ids.dtype, copy=False)
    if batch.legal_action_meta is not None:
        batch.legal_action_meta[base:end] = np.iinfo(batch.legal_action_meta.dtype).max
        if legal_action_meta_arr is not None and seg_len > 0:
            batch.legal_action_meta[base:end] = legal_action_meta_arr[:seg_len]

    # Store per-row offsets shifted by base into (T, N+1)
    batch.legal_offsets[t, :] = (legal_offsets_arr + base).astype(np.uint32, copy=False)

    batch.obs[t] = obs
    batch.to_play_seat[t] = to_play_seat
    batch.decision_id[t] = decision_id
    batch.action[t] = action
    batch.reward[t] = reward
    batch.terminated[t] = terminated
    batch.truncated[t] = truncated
    batch.engine_status[t] = engine_status
    batch.episode_seed[t] = episode_seed
    batch.episode_key[t] = episode_key
    batch.behavior_logp[t] = behavior_logp

    if k_raw_decisions is not None:
        if batch.k_raw_decisions is None:
            batch.k_raw_decisions = np.zeros((batch.T, batch.N), dtype=np.int16)
        _require_shape("k_raw_decisions", np.asarray(k_raw_decisions), (batch.N,))
        batch.k_raw_decisions[t] = k_raw_decisions

    batch._write_t += 1


def finalize_unroll(batch: UnrollBatch) -> UnrollBatch:
    """Validate that the unroll is complete before learner handoff."""
    if batch._write_t != batch.T:
        raise RuntimeError(f"unroll incomplete: wrote {batch._write_t} steps, expected {batch.T}")

    # Track used length and expose during finalize
    if batch.meta.legal_repr == "ids_offsets":
        if batch.legal_ids is None or batch.legal_offsets is None:
            raise RuntimeError("legal_repr='ids_offsets' but legal_ids/legal_offsets missing")
        used = int(batch._packed_legal_write)
        if used < 0 or used > int(batch.legal_ids.shape[0]):
            raise RuntimeError("packed legal write cursor out of bounds")
        # Optional: trim to used length so learner doesn’t see garbage tail
        batch.legal_ids = batch.legal_ids[:used].copy()

    # Extra safety checks
    if batch.meta.legal_repr == "mask" and batch.legal_mask is None:
        raise RuntimeError("legal_repr='mask' but legal_mask is None")

    return batch
