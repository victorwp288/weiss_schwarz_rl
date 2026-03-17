"""Decision-boundary environment wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from typing import Any, Literal, cast

import numpy as np

LegalMode = Literal["mask", "ids_offsets"]
EngineStatusPolicy = Literal["best_effort_reset", "hard_fail"]

_SIM_LAYOUTS: dict[LegalMode, str] = {
    "mask": "mask",
    "ids_offsets": "i16_legal_ids",
}
_VALID_ENGINE_STATUS_POLICIES = frozenset({"best_effort_reset", "hard_fail"})
_DEFAULT_PASS_ACTION_ID = 51


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
        if seed is not None:
            raise ValueError("DecisionBoundaryEnv.reset(seed=...) is not supported yet; pass None")

        weiss_sim = _load_weiss_sim()
        step = weiss_sim.rl.reset_rl(self.pool, layout=_SIM_LAYOUTS[self.legality])
        batch = _pack_batch(step, legality=self.legality)
        self._last_batch = batch
        return batch

    def step(self, actions: Sequence[int] | np.ndarray | int) -> DecisionBoundaryBatch:
        batch = self._require_batch()
        action_array = _coerce_actions(actions, num_envs=self.num_envs, action_space=self.action_space)
        _validate_actions(action_array, batch, pass_action_id=self.pass_action_id)

        weiss_sim = _load_weiss_sim()
        step = weiss_sim.rl.step_rl(self.pool, action_array, layout=_SIM_LAYOUTS[self.legality])
        self._handle_engine_status(step)
        next_batch = _pack_batch(step, legality=self.legality)
        self._last_batch = next_batch
        return next_batch

    def close(self) -> None:
        close_fn = getattr(self.pool, "close", None)
        if callable(close_fn):
            close_fn()

    def _require_batch(self) -> DecisionBoundaryBatch:
        if self._last_batch is None:
            raise RuntimeError("reset() must be called before step()")
        return self._last_batch

    def _handle_engine_status(self, step: Any) -> None:
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

        reset_rows = self._apply_best_effort_reset(engine_status, step)
        if self.counters is not None:
            self.counters.best_effort_reset_rows += reset_rows

    def _apply_best_effort_reset(self, engine_status: Any, step: Any) -> int:
        resetter = getattr(self.pool, "auto_reset_on_error_codes_into", None)
        if not callable(resetter):
            return 0

        reported_rows = resetter(_engine_status_codes(engine_status), step)
        return 0 if reported_rows is None else int(reported_rows)


def _load_weiss_sim() -> Any:
    try:
        return importlib.import_module("weiss_sim")
    except ImportError as exc:
        raise RuntimeError(
            "weiss_sim is required to use DecisionBoundaryEnv. Install it or set PYTHONPATH to the simulator's "
            "python package."
        ) from exc


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


def _engine_status_codes(engine_status: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(engine_status)).astype(np.int32, copy=False)


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


def _pack_batch(step: Any, *, legality: LegalMode) -> DecisionBoundaryBatch:
    actor = np.asarray(step.actor)
    if legality == "mask":
        mask = getattr(step, "masks", None)
        if mask is None:
            raise RuntimeError("mask layout did not return masks")
        return DecisionBoundaryBatch(
            obs=np.asarray(step.obs),
            reward=np.asarray(step.rewards),
            terminated=np.asarray(step.terminated),
            truncated=np.asarray(step.truncated),
            to_play=actor,
            actor=actor,
            decision_id=np.asarray(step.decision_id),
            engine_status=np.asarray(step.engine_status),
            mask=np.asarray(mask),
        )

    legal_ids = getattr(step, "legal_ids", None)
    legal_offsets = getattr(step, "legal_offsets", None)
    if legal_ids is None or legal_offsets is None:
        raise RuntimeError("ids_offsets layout did not return legal_ids/legal_offsets")
    return DecisionBoundaryBatch(
        obs=np.asarray(step.obs),
        reward=np.asarray(step.rewards),
        terminated=np.asarray(step.terminated),
        truncated=np.asarray(step.truncated),
        to_play=actor,
        actor=actor,
        decision_id=np.asarray(step.decision_id),
        engine_status=np.asarray(step.engine_status),
        ids_offsets=(np.asarray(legal_ids), np.asarray(legal_offsets)),
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
