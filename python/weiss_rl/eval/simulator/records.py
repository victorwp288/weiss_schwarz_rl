"""Evaluation records shared by final eval, dev eval, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

OutcomeToken = Literal["W", "L", "D", "T"]


class EvalGameRunner(Protocol):
    def run_game(self, scheduled_game: ScheduledGame) -> GameResult: ...


@dataclass(slots=True)
class EvalSamplerAnomalies:
    cdf_renormalizations: int = 0


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    pair_index: int
    swap_index: int
    episode_index: int
    episode_seed: int
    focal_policy_id: str
    opponent_policy_id: str
    seat0_policy_id: str
    seat1_policy_id: str
    focal_seat: int
    seat0_deck: str | None = None
    seat1_deck: str | None = None


@dataclass(frozen=True, slots=True)
class GameResult:
    episode_seed: int
    terminated: bool
    truncated: bool
    winner_seat: int | None
    engine_status: int = 0
    decision_count: int = 0
    tick_count: int = 0
    no_progress_count: int = 0
    termination_reason: str | None = None
    total_actions: int = 0
    pass_actions: int = 0
    main_move_actions: int = 0
    pass_with_nonpass_available: int = 0
    max_consecutive_main_moves: int = 0
    simulator_episode_key: int | bytes | None = None
    replay_sample: ReplaySampleResult | None = None


@dataclass(frozen=True, slots=True)
class ReplaySampleResult:
    pair_index: int
    swap_index: int
    episode_index: int
    focal_policy_id: str
    opponent_policy_id: str
    raw_replay_path: str | None
    bundle_path: str
    verification_report_path: str
    verification_status: str
    replay_key64: str
    matched: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvalGameRecord:
    pair_index: int
    swap_index: int
    episode_index: int
    episode_seed: int
    episode_key: str
    episode_key64: int
    config_hash256: str
    spec_hash256: str
    focal_policy_id: str
    opponent_policy_id: str
    seat0_policy_id: str
    seat1_policy_id: str
    focal_seat: int
    outcome: OutcomeToken
    terminated: bool
    truncated: bool
    engine_status: int
    seat0_deck: str | None = None
    seat1_deck: str | None = None
    decision_count: int = 0
    tick_count: int = 0
    no_progress_count: int = 0
    termination_reason: str = "terminated"
    total_actions: int = 0
    pass_actions: int = 0
    main_move_actions: int = 0
    pass_with_nonpass_available: int = 0
    max_consecutive_main_moves: int = 0
    run_id256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_hash256": self.config_hash256,
            "engine_status": self.engine_status,
            "episode_index": self.episode_index,
            "episode_key": self.episode_key,
            "episode_key64": self.episode_key64,
            "episode_seed": self.episode_seed,
            "focal_policy_id": self.focal_policy_id,
            "focal_seat": self.focal_seat,
            "opponent_policy_id": self.opponent_policy_id,
            "outcome": self.outcome,
            "pair_index": self.pair_index,
            "seat0_policy_id": self.seat0_policy_id,
            "seat1_policy_id": self.seat1_policy_id,
            "spec_hash256": self.spec_hash256,
            "swap_index": self.swap_index,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "decision_count": self.decision_count,
            "tick_count": self.tick_count,
            "no_progress_count": self.no_progress_count,
            "termination_reason": self.termination_reason,
            "total_actions": self.total_actions,
            "pass_actions": self.pass_actions,
            "main_move_actions": self.main_move_actions,
            "pass_with_nonpass_available": self.pass_with_nonpass_available,
            "max_consecutive_main_moves": self.max_consecutive_main_moves,
        }
        if self.seat0_deck is not None:
            payload["seat0_deck"] = self.seat0_deck
        if self.seat1_deck is not None:
            payload["seat1_deck"] = self.seat1_deck
        if self.run_id256 is not None:
            payload["run_id256"] = self.run_id256
        return payload


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    episodes_path: Path
    records: tuple[EvalGameRecord, ...]
    summary: MatchupSummary


@dataclass(slots=True)
class MatchupSummary:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncations: int = 0
    engine_errors: int = 0
    natural_timeouts: int = 0
    no_progress_timeouts: int = 0
    decision_limit_timeouts: int = 0
    tick_limit_timeouts: int = 0
    timeout_unknown: int = 0
    total_actions: int = 0
    pass_actions: int = 0
    main_move_actions: int = 0
    pass_with_nonpass_available: int = 0
    max_consecutive_main_moves: int = 0


__all__ = [
    "EvalGameRecord",
    "EvalGameRunner",
    "EvalRunResult",
    "EvalSamplerAnomalies",
    "GameResult",
    "MatchupSummary",
    "OutcomeToken",
    "ReplaySampleResult",
    "ScheduledGame",
]
