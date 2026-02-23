"""Canonical trajectory schema objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrajectoryStep:
    decision_id: int
    action: int
    reward: float
    terminated: bool
    truncated: bool
    engine_status: int
    episode_seed: int
    episode_key: int


@dataclass(slots=True)
class TrajectoryDebug:
    decision_kind: int | None = None
    legal_fingerprint64: int | None = None
    actor: int | None = None
