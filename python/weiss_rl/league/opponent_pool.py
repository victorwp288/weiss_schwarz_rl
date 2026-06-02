"""Opponent-pool selection and PFSP sampling."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .pfsp import pfsp_probabilities
from .registry import SnapshotRegistry

NEUTRAL_WIN_RATE = 0.5


@dataclass(frozen=True, slots=True)
class OpponentSnapshotSelection:
    recent_ids: tuple[str, ...]
    champion_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpponentPoolComposition:
    candidate_ids: tuple[str, ...]
    champion_ids: tuple[str, ...]
    recent_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OpponentSamplingDistribution:
    win_rates: np.ndarray
    probabilities: np.ndarray


def _unique_policy_ids(policy_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(policy_ids))


def select_opponent_snapshot_ids(
    registry: SnapshotRegistry,
    *,
    recent_size: int,
    champion_size: int,
) -> tuple[str, ...]:
    return select_opponent_snapshots(
        registry,
        recent_size=recent_size,
        champion_size=champion_size,
    ).candidate_ids


def select_opponent_snapshots(
    registry: SnapshotRegistry,
    *,
    recent_size: int,
    champion_size: int,
) -> OpponentSnapshotSelection:
    recent_ids = tuple(registry.latest_ids(recent_size))
    champion_ids = tuple(registry.latest_champions(champion_size))
    return OpponentSnapshotSelection(
        recent_ids=recent_ids,
        champion_ids=champion_ids,
        candidate_ids=_unique_policy_ids([*recent_ids, *champion_ids]),
    )


def select_runtime_opponent_snapshots(
    registry: SnapshotRegistry,
    *,
    recent_size: int,
    champion_ids: Sequence[str],
    excluded_policy_ids: Collection[str] = (),
) -> OpponentSnapshotSelection:
    excluded = set(excluded_policy_ids)
    selected_champion_ids = tuple(policy_id for policy_id in champion_ids if policy_id not in excluded)
    selected_recent_ids = tuple(
        policy_id for policy_id in registry.latest_ids(recent_size) if policy_id not in excluded
    )
    return OpponentSnapshotSelection(
        recent_ids=selected_recent_ids,
        champion_ids=selected_champion_ids,
        candidate_ids=_unique_policy_ids([*selected_champion_ids, *selected_recent_ids]),
    )


def compose_runtime_opponent_pool(
    *,
    selection: OpponentSnapshotSelection,
    candidate_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
    hard_negative_overlaps_champions: bool = False,
) -> OpponentPoolComposition:
    candidate_id_set = set(candidate_ids)
    hard_negative_id_set = set(hard_negative_ids)
    champion_ids = tuple(
        policy_id
        for policy_id in selection.champion_ids
        if policy_id in candidate_id_set and (hard_negative_overlaps_champions or policy_id not in hard_negative_id_set)
    )
    champion_id_set = set(champion_ids)
    recent_ids = tuple(
        policy_id
        for policy_id in selection.recent_ids
        if policy_id in candidate_id_set and policy_id not in hard_negative_id_set and policy_id not in champion_id_set
    )
    return OpponentPoolComposition(
        candidate_ids=_unique_policy_ids([*hard_negative_ids, *champion_ids, *recent_ids]),
        champion_ids=champion_ids,
        recent_ids=recent_ids,
        hard_negative_ids=tuple(hard_negative_ids),
    )


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
    weight_multipliers_by_snapshot_id: Mapping[str, float] | None = None,
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
    probabilities = opponent_sampling_distribution(
        snapshot_ids,
        win_rates=win_rates,
        weight_multipliers_by_snapshot_id=weight_multipliers_by_snapshot_id,
        power=power,
        eps_uniform=eps_uniform,
    ).probabilities
    sampled_indices = rng.choice(len(snapshot_ids), size=count, replace=True, p=probabilities)
    return tuple(str(snapshot_ids[index]) for index in sampled_indices.tolist())


def opponent_sampling_distribution(
    snapshot_ids: Sequence[str],
    *,
    win_rates: np.ndarray | None = None,
    win_rates_by_snapshot_id: Mapping[str, float] | None = None,
    weight_multipliers_by_snapshot_id: Mapping[str, float] | None = None,
    power: float = 2.0,
    eps_uniform: float = 0.2,
    neutral_win_rate: float = NEUTRAL_WIN_RATE,
) -> OpponentSamplingDistribution:
    resolved_win_rates = (
        resolve_opponent_win_rates(
            snapshot_ids,
            win_rates_by_snapshot_id=win_rates_by_snapshot_id,
            neutral_win_rate=neutral_win_rate,
        )
        if win_rates is None
        else np.asarray(win_rates, dtype=np.float64)
    )
    probabilities = pfsp_probabilities(resolved_win_rates, power=power, eps_uniform=eps_uniform)
    if weight_multipliers_by_snapshot_id is not None:
        multipliers = np.asarray(
            [float(weight_multipliers_by_snapshot_id.get(snapshot_id, 1.0)) for snapshot_id in snapshot_ids],
            dtype=np.float64,
        )
        if np.any(multipliers <= 0.0):
            raise ValueError("weight multipliers must be positive")
        probabilities = probabilities * multipliers
        probabilities = probabilities / np.sum(probabilities)
    return OpponentSamplingDistribution(win_rates=resolved_win_rates, probabilities=probabilities)


@dataclass(slots=True)
class OpponentPoolSampler:
    registry: SnapshotRegistry
    recent_size: int
    champion_size: int
    power: float = 2.0
    eps_uniform: float = 0.2
    neutral_win_rate: float = NEUTRAL_WIN_RATE
    win_rates_by_snapshot_id: Mapping[str, float] | None = None

    def snapshot_selection(self) -> OpponentSnapshotSelection:
        return select_opponent_snapshots(
            self.registry,
            recent_size=self.recent_size,
            champion_size=self.champion_size,
        )

    def snapshot_ids(self) -> tuple[str, ...]:
        return self.snapshot_selection().candidate_ids

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
                self.win_rates_by_snapshot_id if win_rates_by_snapshot_id is None else win_rates_by_snapshot_id
            ),
            power=self.power,
            eps_uniform=self.eps_uniform,
            neutral_win_rate=self.neutral_win_rate,
        )
