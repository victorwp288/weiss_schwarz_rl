"""Decision-boundary environment wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from typing import Any, Literal, cast

import numpy as np

LegalMode = Literal["mask", "ids_offsets"]
EngineStatusPolicy = Literal["best_effort_reset", "hard_fail"]
CopyCasting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]

_SIM_LAYOUTS: dict[LegalMode, str] = {
    "mask": "mask",
    "ids_offsets": "i16_legal_ids",
}
_STEP_OUT_CLASS_NAMES: dict[LegalMode, str] = {
    "mask": "BatchOutMinimal",
    "ids_offsets": "BatchOutMinimalI16LegalIds",
}
_RESET_OUT_CLASS_NAMES: dict[LegalMode, str] = {
    "mask": "BatchOutMinimal",
    "ids_offsets": "BatchOutMinimalNoMask",
}
_RESET_METHOD_NAMES: dict[LegalMode, str] = {
    "mask": "auto_reset_on_error_codes_into",
    "ids_offsets": "auto_reset_on_error_codes_into_nomask",
}
_RESET_DONE_METHOD_NAMES: dict[LegalMode, str] = {
    "mask": "reset_done_into",
    "ids_offsets": "reset_done_into_i16_legal_ids",
}
_RESET_WITH_EPISODE_SEED_METHOD_NAMES: dict[LegalMode, str] = {
    "mask": "reset_indices_with_episode_seeds_into",
    "ids_offsets": "reset_indices_with_episode_seeds_into_i16_legal_ids",
}
_COMMON_OUT_FIELDS = (
    "rewards",
    "terminated",
    "truncated",
    "actor",
    "decision_kind",
    "decision_id",
    "engine_status",
    "spec_hash",
)
_VALID_ENGINE_STATUS_POLICIES = frozenset({"best_effort_reset", "hard_fail"})
_DEFAULT_PASS_ACTION_ID = 51
_U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


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
    episode_seed: np.ndarray
    episode_key: np.ndarray
    mask: np.ndarray | None = None
    ids_offsets: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        num_envs = _require_obs_rows(self.obs)
        _require_vector(self.reward, "reward", num_envs)
        _require_vector(self.terminated, "terminated", num_envs)
        _require_vector(self.truncated, "truncated", num_envs)
        _require_vector(self.to_play, "to_play", num_envs)
        _require_vector(self.actor, "actor", num_envs)
        _require_vector(self.decision_id, "decision_id", num_envs)
        _require_vector(self.engine_status, "engine_status", num_envs)
        _require_vector(self.episode_seed, "episode_seed", num_envs)
        _require_vector(self.episode_key, "episode_key", num_envs)
        _require_legality(self.mask, self.ids_offsets, num_envs)

    @property
    def num_envs(self) -> int:
        return int(self.reward.shape[0])


class DecisionBoundaryEnv:
    """Low-level EnvPool-backed wrapper around `weiss_sim.rl.reset_rl/step_rl`."""

    def __init__(
        self,
        pool: Any,
        *,
        legality: LegalMode = "mask",
        pass_action_id: int = _DEFAULT_PASS_ACTION_ID,
        engine_status_policy: EngineStatusPolicy = "best_effort_reset",
        counters: EngineStatusCounters | None = None,
    ) -> None:
        self.pool = pool
        self.legality = _normalize_legality(legality)
        self.pass_action_id = int(pass_action_id)
        self.engine_status_policy = _normalize_engine_status_policy(engine_status_policy)
        self.counters = counters
        self._last_batch: DecisionBoundaryBatch | None = None
        self._step_out: Any | None = None
        self._reset_out: Any | None = None

    @classmethod
    def create(
        cls,
        *,
        legality: LegalMode = "mask",
        engine_status_policy: EngineStatusPolicy = "best_effort_reset",
        counters: EngineStatusCounters | None = None,
        **kwargs: Any,
    ) -> "DecisionBoundaryEnv":
        if "layout" in kwargs:
            raise TypeError("DecisionBoundaryEnv.create() does not accept layout=; use legality= instead")

        weiss_sim = _load_weiss_sim()
        normalized_legality = _normalize_legality(legality)
        pool, _ = weiss_sim.make_pool(layout=_SIM_LAYOUTS[normalized_legality], **kwargs)
        return cls(
            pool,
            legality=normalized_legality,
            pass_action_id=int(weiss_sim.PASS_ACTION_ID),
            engine_status_policy=engine_status_policy,
            counters=counters,
        )

    @property
    def num_envs(self) -> int:
        return int(self.pool.envs_len)

    @property
    def action_space(self) -> int:
        return int(self.pool.action_space)

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        weiss_sim = _load_weiss_sim()
        step_out = self._require_step_out(weiss_sim)
        if seed is None:
            step = weiss_sim.rl.reset_rl(
                self.pool,
                layout=_SIM_LAYOUTS[self.legality],
                out=step_out,
            )
        else:
            resetter = getattr(self.pool, _RESET_WITH_EPISODE_SEED_METHOD_NAMES[self.legality], None)
            if not callable(resetter):
                raise RuntimeError(
                    f"pool must expose {_RESET_WITH_EPISODE_SEED_METHOD_NAMES[self.legality]} for seeded resets"
                )
            env_indices = list(range(self.num_envs))
            episode_seeds = [int(seed)] * self.num_envs
            resetter(env_indices, episode_seeds, step_out)
            step = step_out
        batch = _pack_batch(step, legality=self.legality, pool=self.pool)
        self._last_batch = batch
        return batch

    def step(self, actions: Sequence[int] | np.ndarray | int) -> DecisionBoundaryBatch:
        batch = self._require_batch()
        action_array = _coerce_actions(actions, num_envs=self.num_envs, action_space=self.action_space)
        _validate_actions(action_array, batch, pass_action_id=self.pass_action_id)

        weiss_sim = _load_weiss_sim()
        step = weiss_sim.rl.step_rl(
            self.pool,
            action_array,
            layout=_SIM_LAYOUTS[self.legality],
            out=self._require_step_out(weiss_sim),
        )
        self._handle_engine_status(step, weiss_sim=weiss_sim)
        next_batch = _pack_batch(step, legality=self.legality, pool=self.pool)
        self._last_batch = next_batch
        return next_batch

    def reset_done(self, done: np.ndarray) -> DecisionBoundaryBatch:
        done_array = np.asarray(done, dtype=np.bool_)
        if done_array.ndim != 1 or int(done_array.shape[0]) != self.num_envs:
            raise ValueError(f"done must have shape ({self.num_envs},)")
        if not np.any(done_array):
            return self._require_batch()

        resetter = getattr(self.pool, _RESET_DONE_METHOD_NAMES[self.legality], None)
        if not callable(resetter):
            raise RuntimeError(f"pool must expose {_RESET_DONE_METHOD_NAMES[self.legality]} for done-row resets")

        weiss_sim = _load_weiss_sim()
        step_out = self._require_step_out(weiss_sim)
        resetter(np.ascontiguousarray(done_array), step_out)
        batch = _pack_batch(step_out, legality=self.legality, pool=self.pool)
        self._last_batch = batch
        return batch

    def close(self) -> None:
        close_fn = getattr(self.pool, "close", None)
        if callable(close_fn):
            close_fn()

    def _require_batch(self) -> DecisionBoundaryBatch:
        if self._last_batch is None:
            raise RuntimeError("reset() must be called before step()")
        return self._last_batch

    def _require_step_out(self, weiss_sim: Any) -> Any:
        if self._step_out is None:
            self._step_out = _make_sim_out(
                weiss_sim,
                class_name=_STEP_OUT_CLASS_NAMES[self.legality],
                num_envs=self.num_envs,
            )
        return self._step_out

    def _require_reset_out(self, weiss_sim: Any) -> Any:
        if self._reset_out is None:
            self._reset_out = _make_sim_out(
                weiss_sim,
                class_name=_RESET_OUT_CLASS_NAMES[self.legality],
                num_envs=self.num_envs,
            )
        return self._reset_out

    def _handle_engine_status(self, step: Any, *, weiss_sim: Any) -> None:
        engine_status = getattr(step, "engine_status", None)
        if engine_status is None:
            return

        fault_rows = _count_fault_rows(engine_status)
        if fault_rows == 0:
            return

        if self.counters is not None:
            self.counters.fault_rows += fault_rows

        if self.engine_status_policy == "hard_fail":
            raise RuntimeError(f"engine_status!=0 (fault_rows={fault_rows})")

        reset_rows = self._apply_best_effort_reset(engine_status, weiss_sim=weiss_sim)
        if self.counters is not None:
            self.counters.best_effort_reset_rows += reset_rows

    def _apply_best_effort_reset(self, engine_status: Any, *, weiss_sim: Any) -> int:
        resetter = getattr(self.pool, _RESET_METHOD_NAMES[self.legality], None)
        if not callable(resetter):
            return 0

        codes = _engine_status_codes(engine_status, num_envs=self.num_envs)
        if self.legality == "mask":
            reported_rows = resetter(codes, self._require_step_out(weiss_sim))
            return 0 if reported_rows is None else int(reported_rows)

        reset_out = self._require_reset_out(weiss_sim)
        reported_rows = resetter(codes, reset_out)
        if reported_rows is None or int(reported_rows) == 0:
            return 0

        step_out = self._require_step_out(weiss_sim)
        fault_rows = codes != 0
        _copy_common_out_fields(src=reset_out, dst=step_out, rows=fault_rows)
        _copy_obs_into(src=reset_out.obs, dst=step_out.obs, rows=fault_rows)

        refill_legal_ids = getattr(self.pool, "legal_action_ids_into", None)
        if not callable(refill_legal_ids):
            raise RuntimeError("pool must expose legal_action_ids_into for ids_offsets best-effort reset")
        refill_out = _make_sim_out(
            weiss_sim,
            class_name=_STEP_OUT_CLASS_NAMES[self.legality],
            num_envs=self.num_envs,
        )
        refill_legal_ids(refill_out.legal_ids, refill_out.legal_offsets)
        _merge_packed_legality_rows(
            dst=step_out,
            current=step_out,
            replacement=refill_out,
            rows=fault_rows,
        )
        return int(reported_rows)


def _load_weiss_sim() -> Any:
    try:
        return importlib.import_module("weiss_sim")
    except ImportError as exc:
        raise RuntimeError(
            "weiss_sim is required to use DecisionBoundaryEnv. Install it or set PYTHONPATH to the simulator's "
            "python package."
        ) from exc


def _make_sim_out(weiss_sim: Any, *, class_name: str, num_envs: int) -> Any:
    out_cls = getattr(weiss_sim, class_name, None)
    if out_cls is None:
        raise RuntimeError(f"weiss_sim is missing required output buffer class {class_name}")
    return out_cls(num_envs)


def _copy_common_out_fields(*, src: Any, dst: Any, rows: np.ndarray | None = None) -> None:
    for field_name in _COMMON_OUT_FIELDS:
        _copy_rows(dst=getattr(dst, field_name), src=getattr(src, field_name), rows=rows)


def _copy_obs_into(*, src: np.ndarray, dst: np.ndarray, rows: np.ndarray | None = None) -> None:
    if np.issubdtype(dst.dtype, np.integer):
        bounds = np.iinfo(dst.dtype)
        clipped = np.clip(src, bounds.min, bounds.max)
        _copy_rows(dst=dst, src=clipped.astype(dst.dtype, copy=False), rows=rows, casting="unsafe")
        return
    _copy_rows(dst=dst, src=src, rows=rows)


def _copy_rows(
    *,
    dst: np.ndarray,
    src: np.ndarray,
    rows: np.ndarray | None = None,
    casting: CopyCasting = "same_kind",
) -> None:
    if rows is None:
        np.copyto(dst, src, casting=casting)
        return
    dst[rows] = np.asarray(src)[rows].astype(dst.dtype, copy=False)


def _merge_packed_legality_rows(*, dst: Any, current: Any, replacement: Any, rows: np.ndarray) -> None:
    current_ids = np.asarray(current.legal_ids)
    current_offsets = np.asarray(current.legal_offsets)
    replacement_ids = np.asarray(replacement.legal_ids)
    replacement_offsets = np.asarray(replacement.legal_offsets)

    merged_ids_parts: list[np.ndarray] = []
    merged_offsets = np.zeros((rows.shape[0] + 1,), dtype=current_offsets.dtype)
    cursor = 0
    for row_index, replace_row in enumerate(rows.tolist()):
        if replace_row:
            row_ids = replacement_ids[int(replacement_offsets[row_index]) : int(replacement_offsets[row_index + 1])]
        else:
            row_ids = current_ids[int(current_offsets[row_index]) : int(current_offsets[row_index + 1])]
        row_ids = np.array(row_ids, copy=True)
        merged_ids_parts.append(row_ids)
        cursor += int(row_ids.size)
        merged_offsets[row_index + 1] = cursor

    merged_ids = (
        np.concatenate(merged_ids_parts, axis=0).astype(current_ids.dtype, copy=False)
        if merged_ids_parts
        else np.zeros((0,), dtype=current_ids.dtype)
    )
    _write_packed_legality(dst=dst, legal_ids=merged_ids, legal_offsets=merged_offsets)


def _write_packed_legality(*, dst: Any, legal_ids: np.ndarray, legal_offsets: np.ndarray) -> None:
    dst_ids = np.asarray(dst.legal_ids)
    dst_offsets = np.asarray(dst.legal_offsets)

    if dst_offsets.shape != legal_offsets.shape:
        raise RuntimeError(
            f"packed legal_offsets shape mismatch: expected {dst_offsets.shape}, got {legal_offsets.shape}"
        )
    if legal_ids.size > dst_ids.shape[0]:
        raise RuntimeError(f"packed legal_ids buffer too small: capacity={dst_ids.shape[0]}, required={legal_ids.size}")

    np.copyto(dst_offsets, legal_offsets.astype(dst_offsets.dtype, copy=False), casting="unsafe")
    if legal_ids.size:
        np.copyto(dst_ids[: legal_ids.size], legal_ids.astype(dst_ids.dtype, copy=False), casting="unsafe")
    if legal_ids.size < dst_ids.shape[0]:
        dst_ids[legal_ids.size :] = 0


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


def _batch_episode_identity(step: Any, *, pool: Any | None, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    episode_seed = getattr(step, "episode_seed", None)
    episode_key = getattr(step, "episode_key", None)
    if episode_seed is not None and episode_key is not None:
        return (
            np.array(episode_seed, dtype=np.uint64, copy=True),
            np.array(episode_key, dtype=np.uint64, copy=True),
        )

    if pool is None:
        zeros = np.zeros((num_envs,), dtype=np.uint64)
        return zeros, zeros.copy()

    pool_episode_seed = getattr(pool, "episode_seed_batch", None)
    if not callable(pool_episode_seed):
        zeros = np.zeros((num_envs,), dtype=np.uint64)
        return zeros, zeros.copy()

    episode_seed_array = np.array(pool_episode_seed(), dtype=np.uint64, copy=True)
    pool_episode_index = getattr(pool, "episode_index_batch", None)
    pool_env_index = getattr(pool, "env_index_batch", None)
    if callable(pool_episode_index) and callable(pool_env_index):
        episode_key_array = _derive_episode_key(
            episode_seed_array,
            np.asarray(pool_episode_index(), dtype=np.uint64),
            np.asarray(pool_env_index(), dtype=np.uint64),
        )
        return episode_seed_array, np.array(episode_key_array, dtype=np.uint64, copy=True)

    if episode_key is not None:
        return episode_seed_array, np.array(episode_key, dtype=np.uint64, copy=True)

    return episode_seed_array, np.zeros((num_envs,), dtype=np.uint64)


def _packed_legal_ids_prefix(legal_ids: Any, legal_offsets: Any) -> np.ndarray:
    ids = np.asarray(legal_ids)
    offsets = np.asarray(legal_offsets)
    used = 0 if offsets.size == 0 else int(offsets[-1])
    if used < 0 or used > ids.shape[0]:
        raise RuntimeError(f"packed legal_ids prefix out of bounds: used={used}, capacity={ids.shape[0]}")
    return np.array(ids[:used], copy=True)


def _pack_batch(step: Any, *, legality: LegalMode, pool: Any | None = None) -> DecisionBoundaryBatch:
    actor = np.array(step.actor, copy=True)
    num_envs = int(actor.shape[0])
    episode_seed, episode_key = _batch_episode_identity(step, pool=pool, num_envs=num_envs)
    if legality == "mask":
        mask = getattr(step, "masks", None)
        if mask is None:
            raise RuntimeError("mask layout did not return masks")
        return DecisionBoundaryBatch(
            obs=np.array(step.obs, copy=True),
            reward=np.array(step.rewards, copy=True),
            terminated=np.array(step.terminated, copy=True),
            truncated=np.array(step.truncated, copy=True),
            to_play=np.array(actor, copy=True),
            actor=actor,
            decision_id=np.array(step.decision_id, copy=True),
            engine_status=np.array(step.engine_status, copy=True),
            episode_seed=episode_seed,
            episode_key=episode_key,
            mask=np.array(mask, copy=True),
        )

    legal_ids = getattr(step, "legal_ids", None)
    legal_offsets = getattr(step, "legal_offsets", None)
    if legal_ids is None or legal_offsets is None:
        raise RuntimeError("ids_offsets layout did not return legal_ids/legal_offsets")
    return DecisionBoundaryBatch(
        obs=np.array(step.obs, copy=True),
        reward=np.array(step.rewards, copy=True),
        terminated=np.array(step.terminated, copy=True),
        truncated=np.array(step.truncated, copy=True),
        to_play=np.array(actor, copy=True),
        actor=actor,
        decision_id=np.array(step.decision_id, copy=True),
        engine_status=np.array(step.engine_status, copy=True),
        episode_seed=episode_seed,
        episode_key=episode_key,
        ids_offsets=(_packed_legal_ids_prefix(legal_ids, legal_offsets), np.array(legal_offsets, copy=True)),
    )


def _coerce_actions(
    actions: Sequence[int] | np.ndarray | int,
    *,
    num_envs: int,
    action_space: int,
) -> np.ndarray:
    if isinstance(actions, np.ndarray):
        action_array = actions
    elif np.isscalar(actions):
        action_array = np.asarray([actions])
    else:
        action_array = np.asarray(list(cast(Sequence[int], actions)))

    if action_array.ndim != 1 or int(action_array.shape[0]) != num_envs:
        raise ValueError(f"actions must have shape ({num_envs},)")
    if not np.issubdtype(action_array.dtype, np.integer):
        raise TypeError("actions must be integers")

    signed = action_array.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("actions must be >= 0")
    if np.any(signed >= action_space):
        raise ValueError(f"actions must be < action_space ({action_space})")
    return signed.astype(np.uint32, copy=False)


def _validate_actions(
    actions: np.ndarray,
    batch: DecisionBoundaryBatch,
    *,
    pass_action_id: int,
) -> None:
    if batch.mask is not None:
        _validate_mask_actions(actions, batch.mask, pass_action_id=pass_action_id)
        return

    assert batch.ids_offsets is not None
    legal_ids, legal_offsets = batch.ids_offsets
    _validate_packed_actions(actions, legal_ids, legal_offsets, pass_action_id=pass_action_id)


def _validate_mask_actions(actions: np.ndarray, mask: np.ndarray, *, pass_action_id: int) -> None:
    for env_index, action in enumerate(actions.tolist()):
        legal_row = mask[env_index]
        if not bool(np.any(legal_row)):
            _require_pass_action(action, env_index, pass_action_id)
            continue
        if not bool(legal_row[action]):
            raise ValueError(f"illegal action {action} for env {env_index}")


def _validate_packed_actions(
    actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    pass_action_id: int,
) -> None:
    for env_index, action in enumerate(actions.tolist()):
        start = int(legal_offsets[env_index])
        end = int(legal_offsets[env_index + 1])
        if start == end:
            _require_pass_action(action, env_index, pass_action_id)
            continue

        env_legal_ids = legal_ids[start:end]
        position = int(np.searchsorted(env_legal_ids, action))
        is_legal = position < env_legal_ids.size and int(env_legal_ids[position]) == action
        if not is_legal:
            raise ValueError(f"illegal action {action} for env {env_index}")


def _require_pass_action(action: int, env_index: int, pass_action_id: int) -> None:
    if action != pass_action_id:
        raise ValueError(f"env {env_index} has no legal actions; expected pass action {pass_action_id}, got {action}")
