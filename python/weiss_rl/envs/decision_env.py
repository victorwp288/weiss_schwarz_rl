"""Decision-boundary environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


EngineStatusPolicy = Literal["best_effort_reset", "hard_fail"]
_VALID_ENGINE_STATUS_POLICIES = frozenset({"best_effort_reset", "hard_fail"})


@dataclass(slots=True)
class EngineStatusCounters:
    """Training-side counters for simulator engine-status faults."""

    fault_rows: int = 0
    best_effort_reset_rows: int = 0


def _engine_status_codes(engine_status: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(engine_status)).astype(np.int32, copy=False)


def _count_fault_rows(engine_status: Any) -> int:
    return int(np.count_nonzero(_engine_status_codes(engine_status) != 0))


@dataclass(slots=True)
class DecisionBoundaryEnv:
    """Thin wrapper around weiss_sim environments for policy-step interaction."""

    sim: Any
    engine_status_policy: EngineStatusPolicy = "best_effort_reset"
    counters: EngineStatusCounters | None = None

    def __post_init__(self) -> None:
        if self.engine_status_policy not in _VALID_ENGINE_STATUS_POLICIES:
            expected = ", ".join(sorted(_VALID_ENGINE_STATUS_POLICIES))
            raise ValueError(f"engine_status_policy must be one of: {expected}")

    @classmethod
    def create(
        cls,
        *,
        engine_status_policy: EngineStatusPolicy = "best_effort_reset",
        counters: EngineStatusCounters | None = None,
        **kwargs: Any,
    ) -> "DecisionBoundaryEnv":
        import weiss_sim  # type: ignore

        sim = weiss_sim.make(**kwargs)
        return cls(sim=sim, engine_status_policy=engine_status_policy, counters=counters)

    def reset(self, seed: int | None = None):
        return self.sim.reset(seed=seed)

    def step(self, actions):
        out = self.sim.step(actions)
        engine_status = getattr(out, "engine_status", None)
        if engine_status is None:
            return out

        fault_rows = _count_fault_rows(engine_status)
        if fault_rows == 0:
            return out

        if self.counters is not None:
            self.counters.fault_rows += fault_rows

        if self.engine_status_policy == "hard_fail":
            raise RuntimeError(f"engine_status!=0 (fault_rows={fault_rows})")

        if self.counters is not None:
            self.counters.best_effort_reset_rows += self._apply_best_effort_reset(engine_status, out)
        else:
            self._apply_best_effort_reset(engine_status, out)
        return out

    def close(self) -> None:
        close_fn = getattr(self.sim, "close", None)
        if callable(close_fn):
            close_fn()

    def _apply_best_effort_reset(self, engine_status: Any, out: Any) -> int:
        pool = getattr(self.sim, "pool", None)
        resetter = getattr(pool, "auto_reset_on_error_codes_into", None)
        if not callable(resetter):
            return 0

        reported_rows = resetter(_engine_status_codes(engine_status), out)
        return 0 if reported_rows is None else int(reported_rows)
