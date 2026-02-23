"""Deterministic evaluation harness scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MatchupSummary:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncations: int = 0


def summarize_pair_outcomes(outcomes: list[str]) -> MatchupSummary:
    out = MatchupSummary()
    for token in outcomes:
        key = token.strip().lower()
        if key == "w":
            out.wins += 1
        elif key == "l":
            out.losses += 1
        elif key == "d":
            out.draws += 1
        elif key == "t":
            out.truncations += 1
    return out
