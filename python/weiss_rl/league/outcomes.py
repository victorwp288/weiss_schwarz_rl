"""Online outcome tracker for PFSP sliding-window win rates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

OutcomeToken = str


@dataclass(slots=True)
class _WindowCounts:
    outcomes: deque[OutcomeToken]
    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0

    def push(self, outcome: OutcomeToken, *, maxlen: int) -> None:
        normalized = _normalize_outcome(outcome)
        if len(self.outcomes) == maxlen:
            self._remove(self.outcomes.popleft())
        self.outcomes.append(normalized)
        self._add(normalized)

    def total(self) -> int:
        return self.wins + self.losses + self.draws + self.timeouts

    def win_rate(self, *, draw_value: float, timeout_value: float) -> float:
        total = self.total()
        if total == 0:
            return 0.5
        return float(self.wins + draw_value * self.draws + timeout_value * self.timeouts) / float(total)

    def counts(self) -> tuple[int, int, int, int]:
        return self.wins, self.losses, self.draws, self.timeouts

    def _add(self, outcome: OutcomeToken) -> None:
        if outcome == "w":
            self.wins += 1
        elif outcome == "l":
            self.losses += 1
        elif outcome == "d":
            self.draws += 1
        else:
            self.timeouts += 1

    def _remove(self, outcome: OutcomeToken) -> None:
        if outcome == "w":
            self.wins -= 1
        elif outcome == "l":
            self.losses -= 1
        elif outcome == "d":
            self.draws -= 1
        else:
            self.timeouts -= 1


@dataclass(slots=True)
class OnlineOutcomeTracker:
    """Sliding-window win-rate estimates keyed by opponent snapshot id."""

    window_size: int = 50_000
    draw_value: float = 0.5
    timeout_value: float = 0.0
    current_epoch: int = 0
    by_opponent: dict[tuple[int, str], _WindowCounts] = field(default_factory=dict)

    def update(self, opponent_id: str, outcome: OutcomeToken, *, epoch: int | None = None) -> None:
        key = self._scoped_key(opponent_id, epoch=epoch)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        counts = self.by_opponent.get(key)
        if counts is None:
            counts = _WindowCounts(outcomes=deque())
            self.by_opponent[key] = counts
        counts.push(outcome, maxlen=self.window_size)

    def win_rate(self, opponent_id: str, *, epoch: int | None = None) -> float:
        counts = self.by_opponent.get(self._scoped_key(opponent_id, epoch=epoch))
        if counts is None:
            return 0.5
        return counts.win_rate(draw_value=self.draw_value, timeout_value=self.timeout_value)

    def win_rates(self, opponent_ids: list[str], *, epoch: int | None = None) -> list[float]:
        return [self.win_rate(opponent_id, epoch=epoch) for opponent_id in opponent_ids]

    def counts(self, opponent_id: str, *, epoch: int | None = None) -> tuple[int, int, int, int]:
        counts = self.by_opponent.get(self._scoped_key(opponent_id, epoch=epoch))
        if counts is None:
            return (0, 0, 0, 0)
        return counts.counts()

    def clear(self) -> None:
        self.by_opponent.clear()

    def bump_epoch(self, *, drop_previous: bool = True) -> int:
        self.current_epoch = int(self.current_epoch) + 1
        if drop_previous:
            self.by_opponent.clear()
        return int(self.current_epoch)

    def _scoped_key(self, opponent_id: str, *, epoch: int | None = None) -> tuple[int, str]:
        resolved_epoch = int(self.current_epoch if epoch is None else epoch)
        return resolved_epoch, _normalize_opponent_id(opponent_id)


def _normalize_outcome(outcome: OutcomeToken) -> OutcomeToken:
    normalized = str(outcome).strip().lower()
    if normalized not in {"w", "l", "d", "t"}:
        raise ValueError(f"outcome must be one of w/l/d/t, got {outcome!r}")
    return normalized


def _normalize_opponent_id(opponent_id: str) -> str:
    normalized = str(opponent_id).strip()
    if not normalized:
        raise ValueError("opponent_id must be non-empty")
    return normalized
