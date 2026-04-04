"""Opponent-pool selection and PFSP sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .pfsp import pfsp_probabilities
from .registry import SnapshotRegistry

NEUTRAL_WIN_RATE = 0.5


def select_opponent_snapshot_ids(
    registry: SnapshotRegistry,
    *,
    recent_size: int,
    champion_size: int,
) -> tuple[str, ...]:
    recent_ids = registry.latest_ids(recent_size)
    champion_ids = registry.latest_champions(champion_size)
    return tuple(dict.fromkeys([*recent_ids, *champion_ids]))


def resolve_opponent_win_rates(
    snapshot_ids: Sequence[str],
    *,
    win_rates_by_snapshot_id: Mapping[str, float] | None = None,
    neutral_win_rate: float = NEUTRAL_WIN_RATE,
) -> np.ndarray:
    if not 0.0 <= neutral_win_rate <= 1.0:
        raise ValueError("neutral_win_rate must be in [0, 1]")
    win_rates = {} if win_rates_by_snapshot_id is None else win_rates_by_snapshot_id
    return np.asarray([float(win_rates.get(snapshot_id, neutral_win_rate)) for snapshot_id in snapshot_ids])


def sample_opponent_snapshot_ids(
    snapshot_ids: Sequence[str],
    *,
    count: int,
    rng: np.random.Generator,
    win_rates_by_snapshot_id: Mapping[str, float] | None = None,
    power: float = 2.0,
    eps_uniform: float = 0.2,
    neutral_win_rate: float = NEUTRAL_WIN_RATE,
) -> tuple[str, ...]:
    if count <= 0:
        raise ValueError("count must be >= 1")
    if len(snapshot_ids) == 0:
        raise ValueError("snapshot_ids must not be empty")

    win_rates = resolve_opponent_win_rates(
        snapshot_ids,
        win_rates_by_snapshot_id=win_rates_by_snapshot_id,
        neutral_win_rate=neutral_win_rate,
    )
    probabilities = pfsp_probabilities(win_rates, power=power, eps_uniform=eps_uniform)
    sampled_indices = rng.choice(len(snapshot_ids), size=count, replace=True, p=probabilities)
    return tuple(str(snapshot_ids[index]) for index in sampled_indices.tolist())


@dataclass(slots=True)
class OpponentPoolSampler:
    registry: SnapshotRegistry
    recent_size: int
    champion_size: int
    power: float = 2.0
    eps_uniform: float = 0.2
    neutral_win_rate: float = NEUTRAL_WIN_RATE
    win_rates_by_snapshot_id: Mapping[str, float] | None = None

    def snapshot_ids(self) -> tuple[str, ...]:
        return select_opponent_snapshot_ids(
            self.registry,
            recent_size=self.recent_size,
            champion_size=self.champion_size,
        )

    def sample(
        self,
        *,
        count: int,
        rng: np.random.Generator,
        win_rates_by_snapshot_id: Mapping[str, float] | None = None,
    ) -> tuple[str, ...]:
        return sample_opponent_snapshot_ids(
            self.snapshot_ids(),
            count=count,
            rng=rng,
            win_rates_by_snapshot_id=(
                self.win_rates_by_snapshot_id
                if win_rates_by_snapshot_id is None
                else win_rates_by_snapshot_id
            ),
            power=self.power,
            eps_uniform=self.eps_uniform,
            neutral_win_rate=self.neutral_win_rate,
        )
