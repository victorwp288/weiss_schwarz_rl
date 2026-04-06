"""Online outcome tracker for PFSP sliding-window win rates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


OutcomeToken = str


@dataclass(slots=True)
class _WindowCounts:
    outcomes: Deque[OutcomeToken]
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def push(self, outcome: OutcomeToken, *, maxlen: int) -> None:
        normalized = _normalize_outcome(outcome)
        if len(self.outcomes) == maxlen:
            self._remove(self.outcomes.popleft())
        self.outcomes.append(normalized)
        self._add(normalized)

    def total(self) -> int:
        return self.wins + self.losses + self.draws

    def win_rate(self, *, draw_value: float) -> float:
        total = self.total()
        if total == 0:
            return 0.5
        return float(self.wins + draw_value * self.draws) / float(total)

    def counts(self) -> tuple[int, int, int]:
        return self.wins, self.losses, self.draws

    def _add(self, outcome: OutcomeToken) -> None:
        if outcome == "w":
            self.wins += 1
        elif outcome == "l":
            self.losses += 1
        else:
            self.draws += 1

    def _remove(self, outcome: OutcomeToken) -> None:
        if outcome == "w":
            self.wins -= 1
        elif outcome == "l":
            self.losses -= 1
        else:
            self.draws -= 1


@dataclass(slots=True)
class OnlineOutcomeTracker:
    """Sliding-window win-rate estimates keyed by opponent snapshot id."""

    window_size: int = 50_000
    draw_value: float = 0.5
    by_opponent: dict[str, _WindowCounts] = field(default_factory=dict)

    def update(self, opponent_id: str, outcome: OutcomeToken) -> None:
        key = _normalize_opponent_id(opponent_id)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        counts = self.by_opponent.get(key)
        if counts is None:
            counts = _WindowCounts(outcomes=deque())
            self.by_opponent[key] = counts
        counts.push(outcome, maxlen=self.window_size)

    def win_rate(self, opponent_id: str) -> float:
        counts = self.by_opponent.get(str(opponent_id).strip())
        if counts is None:
            return 0.5
        return counts.win_rate(draw_value=self.draw_value)

    def win_rates(self, opponent_ids: list[str]) -> list[float]:
        return [self.win_rate(opponent_id) for opponent_id in opponent_ids]

    def counts(self, opponent_id: str) -> tuple[int, int, int]:
        counts = self.by_opponent.get(str(opponent_id).strip())
        if counts is None:
            return (0, 0, 0)
        return counts.counts()


def _normalize_outcome(outcome: OutcomeToken) -> OutcomeToken:
    normalized = str(outcome).strip().lower()
    if normalized not in {"w", "l", "d"}:
        raise ValueError(f"outcome must be one of w/l/d, got {outcome!r}")
    return normalized


def _normalize_opponent_id(opponent_id: str) -> str:
    normalized = str(opponent_id).strip()
    if not normalized:
        raise ValueError("opponent_id must be non-empty")
    return normalized
