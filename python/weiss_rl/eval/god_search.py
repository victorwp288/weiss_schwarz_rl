"""Decision-time search helpers for exploratory strong-player evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

GodSearchMode = Literal["disabled", "same_world_prefix_rollout"]
GodSearchRolloutPolicy = Literal["eval", "argmax", "sample"]


@dataclass(frozen=True, slots=True)
class GodSearchConfig:
    """Configuration for the isolated god-search eval track.

    ``same_world_prefix_rollout`` reconstructs the simulator state from the
    episode seed and prior action prefix, then evaluates top-K candidate root
    actions by rolling forward in that sampled world. This is intentionally
    named so reports do not confuse it with a blind belief-state policy.
    """

    mode: GodSearchMode = "disabled"
    top_k: int = 4
    rollouts_per_action: int = 1
    max_rollout_decisions: int = 0
    max_search_decisions_per_game: int = 0
    rollout_policy: GodSearchRolloutPolicy = "eval"
    apply_to_focal_only: bool = True
    verify_prefix_replay: bool = True
    fail_on_prefix_mismatch: bool = True
    trace_limit: int = 24

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "same_world_prefix_rollout"}:
            raise ValueError(f"unknown god-search mode: {self.mode!r}")
        if self.top_k < 1:
            raise ValueError(f"god-search top_k must be >= 1, got {self.top_k}")
        if self.rollouts_per_action < 1:
            raise ValueError(f"god-search rollouts_per_action must be >= 1, got {self.rollouts_per_action}")
        if self.max_rollout_decisions < 0:
            raise ValueError(f"god-search max_rollout_decisions must be >= 0, got {self.max_rollout_decisions}")
        if self.max_search_decisions_per_game < 0:
            raise ValueError(
                f"god-search max_search_decisions_per_game must be >= 0, got {self.max_search_decisions_per_game}"
            )
        if self.rollout_policy not in {"eval", "argmax", "sample"}:
            raise ValueError(f"unknown god-search rollout_policy: {self.rollout_policy!r}")
        if self.trace_limit < 0:
            raise ValueError(f"god-search trace_limit must be >= 0, got {self.trace_limit}")

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> GodSearchConfig:
        if not payload:
            return cls()
        mode = str(payload.get("mode", "disabled") or "disabled").strip()
        rollout_policy = str(payload.get("rollout_policy", "eval") or "eval").strip()
        return cls(
            mode=mode,  # type: ignore[arg-type]
            top_k=int(payload.get("top_k", 4)),
            rollouts_per_action=int(payload.get("rollouts_per_action", 1)),
            max_rollout_decisions=int(payload.get("max_rollout_decisions", 0)),
            max_search_decisions_per_game=int(payload.get("max_search_decisions_per_game", 0)),
            rollout_policy=rollout_policy,  # type: ignore[arg-type]
            apply_to_focal_only=bool(payload.get("apply_to_focal_only", True)),
            verify_prefix_replay=bool(payload.get("verify_prefix_replay", True)),
            fail_on_prefix_mismatch=bool(payload.get("fail_on_prefix_mismatch", True)),
            trace_limit=int(payload.get("trace_limit", 24)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "top_k": int(self.top_k),
            "rollouts_per_action": int(self.rollouts_per_action),
            "max_rollout_decisions": int(self.max_rollout_decisions),
            "max_search_decisions_per_game": int(self.max_search_decisions_per_game),
            "rollout_policy": self.rollout_policy,
            "apply_to_focal_only": bool(self.apply_to_focal_only),
            "verify_prefix_replay": bool(self.verify_prefix_replay),
            "fail_on_prefix_mismatch": bool(self.fail_on_prefix_mismatch),
            "trace_limit": int(self.trace_limit),
        }


@dataclass(slots=True)
class GodSearchStats:
    trace_limit: int = 24
    search_decisions: int = 0
    changed_decisions: int = 0
    skipped_non_focal: int = 0
    skipped_no_model: int = 0
    skipped_single_candidate: int = 0
    candidate_evaluations: int = 0
    rollout_games: int = 0
    terminal_rollouts: int = 0
    truncated_rollouts: int = 0
    horizon_cutoffs: int = 0
    prefix_replay_failures: int = 0
    traces: list[dict[str, Any]] = field(default_factory=list)

    def add_trace(self, trace: dict[str, Any]) -> None:
        if len(self.traces) < int(self.trace_limit):
            self.traces.append(trace)

    def to_json_dict(self, *, config: GodSearchConfig) -> dict[str, Any]:
        return {
            "kind": "god_search_diagnostics_v1",
            "config": config.to_json_dict(),
            "counters": {
                "search_decisions": int(self.search_decisions),
                "changed_decisions": int(self.changed_decisions),
                "skipped_non_focal": int(self.skipped_non_focal),
                "skipped_no_model": int(self.skipped_no_model),
                "skipped_single_candidate": int(self.skipped_single_candidate),
                "candidate_evaluations": int(self.candidate_evaluations),
                "rollout_games": int(self.rollout_games),
                "terminal_rollouts": int(self.terminal_rollouts),
                "truncated_rollouts": int(self.truncated_rollouts),
                "horizon_cutoffs": int(self.horizon_cutoffs),
                "prefix_replay_failures": int(self.prefix_replay_failures),
            },
            "changed_fraction": (
                None if self.search_decisions <= 0 else float(self.changed_decisions) / float(self.search_decisions)
            ),
            "traces": list(self.traces),
        }


def top_k_legal_actions(logits: np.ndarray, legal_ids: np.ndarray, *, top_k: int) -> tuple[int, ...]:
    """Return legal action IDs ordered by descending logit, tie-breaking by action id."""

    logits_array = np.asarray(logits, dtype=np.float32)
    legal_ids_array = np.asarray(legal_ids, dtype=np.int64)
    if legal_ids_array.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids_array.size == 0:
        return ()
    if np.any(legal_ids_array < 0) or np.any(legal_ids_array >= logits_array.shape[0]):
        raise ValueError("legal_ids contain action ids outside the logits range")
    legal_logits = logits_array[legal_ids_array]
    if not np.all(np.isfinite(legal_logits)):
        raise ValueError("legal logits must be finite")
    order = np.lexsort((legal_ids_array, -legal_logits))
    selected = legal_ids_array[order[: max(1, int(top_k))]]
    return tuple(int(action) for action in selected.tolist())
