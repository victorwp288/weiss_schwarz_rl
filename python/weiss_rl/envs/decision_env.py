"""Decision-boundary environment wrapper."""

from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from weiss_rl.envs.decision_action_validation import _coerce_actions, _validate_actions
from weiss_rl.envs.decision_batch import (
    DecisionBoundaryBatch,
    EngineStatusCounters,
    EngineStatusPolicy,
    LegalMode,
    _count_fault_rows,
    _engine_status_codes,
    _normalize_engine_status_policy,
    _normalize_legality,
)
from weiss_rl.envs.decision_batch_packing import _derive_episode_key, _pack_batch
from weiss_rl.envs.decision_sim_buffers import (
    _copy_common_out_fields,
    _copy_obs_into,
    _make_sim_out,
    _merge_packed_legality_rows,
)

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
_DEFAULT_PASS_ACTION_ID = 51


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
        copy_arrays: bool = True,
        max_decisions: int | None = None,
        max_ticks: int | None = None,
        max_no_progress_decisions: int | None = None,
        profile_timers: bool = False,
    ) -> None:
        self.pool = pool
        self.legality = _normalize_legality(legality)
        self.pass_action_id = int(pass_action_id)
        self.engine_status_policy = _normalize_engine_status_policy(engine_status_policy)
        self.counters = counters
        self.copy_arrays = bool(copy_arrays)
        self.max_decisions = None if max_decisions is None else int(max_decisions)
        self.max_ticks = None if max_ticks is None else int(max_ticks)
        self.max_no_progress_decisions = None if max_no_progress_decisions is None else int(max_no_progress_decisions)
        self._last_batch: DecisionBoundaryBatch | None = None
        self._step_out: Any | None = None
        self._reset_out: Any | None = None
        self._action_buffer: np.ndarray | None = None
        self._action_logp_buffer: np.ndarray | None = None
        self._simulator_timing_enabled = bool(profile_timers)
        self._python_timing_counters: dict[str, int] = {}
        self._configure_simulator_timing(self._simulator_timing_enabled)

    @classmethod
    def create(
        cls,
        *,
        legality: LegalMode = "mask",
        engine_status_policy: EngineStatusPolicy = "best_effort_reset",
        counters: EngineStatusCounters | None = None,
        profile_timers: bool = False,
        **kwargs: Any,
    ) -> DecisionBoundaryEnv:
        if "layout" in kwargs:
            raise TypeError("DecisionBoundaryEnv.create() does not accept layout=; use legality= instead")

        weiss_sim = _load_weiss_sim()
        normalized_legality = _normalize_legality(legality)
        pool, _ = weiss_sim.make_pool(layout=_SIM_LAYOUTS[normalized_legality], **kwargs)
        curriculum = kwargs.get("curriculum")
        max_no_progress_decisions = None
        if isinstance(curriculum, dict):
            raw_limit = curriculum.get("max_no_progress_decisions")
            if raw_limit is not None:
                max_no_progress_decisions = int(raw_limit)
        return cls(
            pool,
            legality=normalized_legality,
            pass_action_id=int(weiss_sim.PASS_ACTION_ID),
            engine_status_policy=engine_status_policy,
            counters=counters,
            copy_arrays=True,
            max_decisions=None if "max_decisions" not in kwargs else int(kwargs["max_decisions"]),
            max_ticks=None if "max_ticks" not in kwargs else int(kwargs["max_ticks"]),
            max_no_progress_decisions=max_no_progress_decisions,
            profile_timers=profile_timers,
        )

    @property
    def num_envs(self) -> int:
        return int(self.pool.envs_len)

    @property
    def action_space(self) -> int:
        return int(self.pool.action_space)

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        started = time.perf_counter_ns()
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
        batch = _pack_batch(step, legality=self.legality, pool=self.pool, copy_arrays=self.copy_arrays)
        self._last_batch = batch
        self._record_python_timing("python_reset", time.perf_counter_ns() - started)
        return batch

    def step(self, actions: Sequence[int] | np.ndarray | int) -> DecisionBoundaryBatch:
        started = time.perf_counter_ns()
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
        next_batch = _pack_batch(step, legality=self.legality, pool=self.pool, copy_arrays=self.copy_arrays)
        self._last_batch = next_batch
        self._record_python_timing("python_step", time.perf_counter_ns() - started)
        return next_batch

    def step_sample_from_logits(
        self,
        logits: object,
        seeds: int | Sequence[int] | np.ndarray,
    ) -> tuple[DecisionBoundaryBatch, np.ndarray]:
        started = time.perf_counter_ns()
        self._require_batch()
        weiss_sim = _load_weiss_sim()
        actions = self._require_action_buffer()
        step, actions = weiss_sim.rl.step_rl_sample_from_logits(
            self.pool,
            logits,
            seeds,
            layout=_SIM_LAYOUTS[self.legality],
            actions=actions,
            out=self._require_step_out(weiss_sim),
        )
        self._handle_engine_status(step, weiss_sim=weiss_sim)
        next_batch = _pack_batch(step, legality=self.legality, pool=self.pool, copy_arrays=self.copy_arrays)
        self._last_batch = next_batch
        self._record_python_timing("python_step_sample_from_logits", time.perf_counter_ns() - started)
        return next_batch, np.array(actions, copy=self.copy_arrays)

    def step_sample_from_logits_with_logp(
        self,
        logits: object,
        seeds: int | Sequence[int] | np.ndarray,
    ) -> tuple[DecisionBoundaryBatch, np.ndarray, np.ndarray]:
        started = time.perf_counter_ns()
        self._require_batch()
        if self.legality != "ids_offsets":
            raise RuntimeError("step_sample_from_logits_with_logp requires legality='ids_offsets'")
        weiss_sim = _load_weiss_sim()
        actions = self._require_action_buffer()
        action_logp = self._require_action_logp_buffer()
        step, actions, action_logp = weiss_sim.rl.step_rl_sample_from_logits_with_logp(
            self.pool,
            logits,
            seeds,
            layout=_SIM_LAYOUTS[self.legality],
            actions=actions,
            action_logp=action_logp,
            out=self._require_step_out(weiss_sim),
        )
        self._handle_engine_status(step, weiss_sim=weiss_sim)
        next_batch = _pack_batch(step, legality=self.legality, pool=self.pool, copy_arrays=self.copy_arrays)
        self._last_batch = next_batch
        self._record_python_timing(
            "python_step_sample_from_logits_with_logp",
            time.perf_counter_ns() - started,
        )
        return (
            next_batch,
            np.array(actions, copy=self.copy_arrays),
            np.array(action_logp, copy=self.copy_arrays),
        )

    def reset_done(self, done: np.ndarray) -> DecisionBoundaryBatch:
        started = time.perf_counter_ns()
        done_array = np.asarray(done, dtype=np.bool_)
        if done_array.ndim != 1 or int(done_array.shape[0]) != self.num_envs:
            raise ValueError(f"done must have shape ({self.num_envs},)")
        if not np.any(done_array):
            self._record_python_timing("python_reset_done", time.perf_counter_ns() - started)
            return self._require_batch()

        resetter = getattr(self.pool, _RESET_DONE_METHOD_NAMES[self.legality], None)
        if not callable(resetter):
            raise RuntimeError(f"pool must expose {_RESET_DONE_METHOD_NAMES[self.legality]} for done-row resets")

        weiss_sim = _load_weiss_sim()
        step_out = self._require_step_out(weiss_sim)
        resetter(np.ascontiguousarray(done_array), step_out)
        batch = _pack_batch(step_out, legality=self.legality, pool=self.pool, copy_arrays=self.copy_arrays)
        self._last_batch = batch
        self._record_python_timing("python_reset_done", time.perf_counter_ns() - started)
        return batch

    def close(self) -> None:
        close_fn = getattr(self.pool, "close", None)
        if callable(close_fn):
            close_fn()

    def drain_timing_counters(self) -> dict[str, int]:
        snapshot: dict[str, int] = dict(self._python_timing_counters)
        self._python_timing_counters.clear()
        if not self._simulator_timing_enabled:
            return snapshot
        getter = getattr(self.pool, "timing_counters", None)
        if callable(getter):
            raw_snapshot = getter()
            snapshot.update(
                {str(key): int(value) for key, value in dict(raw_snapshot).items() if str(key) != "timing_enabled"}
            )
        resetter = getattr(self.pool, "reset_timing_counters", None)
        if callable(resetter):
            resetter()
        return snapshot

    def _require_batch(self) -> DecisionBoundaryBatch:
        if self._last_batch is None:
            raise RuntimeError("reset() must be called before step()")
        return self._last_batch

    def _configure_simulator_timing(self, enabled: bool) -> None:
        self._python_timing_counters.clear()
        setter = getattr(self.pool, "set_timing_enabled", None)
        if callable(setter):
            setter(bool(enabled))
        resetter = getattr(self.pool, "reset_timing_counters", None)
        if bool(enabled) and callable(resetter):
            resetter()

    def _record_python_timing(self, key: str, elapsed_ns: int) -> None:
        if not self._simulator_timing_enabled:
            return
        self._python_timing_counters[key] = self._python_timing_counters.get(key, 0) + max(
            int(elapsed_ns),
            0,
        )

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

    def _require_action_buffer(self) -> np.ndarray:
        if self._action_buffer is None:
            self._action_buffer = np.empty((self.num_envs,), dtype=np.uint32)
        return self._action_buffer

    def _require_action_logp_buffer(self) -> np.ndarray:
        if self._action_logp_buffer is None:
            self._action_logp_buffer = np.empty((self.num_envs,), dtype=np.float32)
        return self._action_logp_buffer

    def _handle_engine_status(self, step: Any, *, weiss_sim: Any | None = None) -> None:
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
        if self.engine_status_policy == "passthrough":
            return

        if weiss_sim is None:
            raise RuntimeError("best_effort_reset is not supported for PackedTrainingDecisionBoundaryEnv")
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
        refill_legal_action_meta = getattr(self.pool, "legal_action_meta_into", None)
        if callable(refill_legal_action_meta) and hasattr(refill_out, "legal_action_meta"):
            refill_legal_action_meta(refill_out.legal_action_meta)
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


__all__ = [
    "DecisionBoundaryBatch",
    "DecisionBoundaryEnv",
    "EngineStatusCounters",
    "_derive_episode_key",
    "_merge_packed_legality_rows",
    "_pack_batch",
    "_validate_actions",
]
