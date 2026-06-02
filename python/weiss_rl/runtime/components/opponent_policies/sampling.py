"""Sampling result types and counters for runtime opponent selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class OpponentSamplingResult:
    policy_ids: tuple[str, ...]
    sampled_envs: int = 0
    mirror_envs: int = 0
    heuristic_public_envs: int = 0
    heuristic_public_variant_envs: int = 0
    noleague_baseline_envs: int = 0
    champion_envs: int = 0
    recent_envs: int = 0
    hard_negative_envs: int = 0
    warmup_snapshot_envs: int = 0
    sampled_policy_envs: tuple[tuple[str, int], ...] = ()
    heuristic_public_policy_envs: tuple[tuple[str, int], ...] = ()
    heuristic_public_variant_policy_envs: tuple[tuple[str, int], ...] = ()
    noleague_baseline_policy_envs: tuple[tuple[str, int], ...] = ()
    champion_policy_envs: tuple[tuple[str, int], ...] = ()
    recent_policy_envs: tuple[tuple[str, int], ...] = ()
    hard_negative_policy_envs: tuple[tuple[str, int], ...] = ()
    warmup_snapshot_policy_envs: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeOpponentGroup:
    name: str
    policy_ids: tuple[str, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class RuntimeOpponentSamplingPlan:
    groups: tuple[RuntimeOpponentGroup, ...]
    probabilities: np.ndarray


@dataclass(slots=True)
class OpponentSamplingAccumulator:
    sample_count: int
    policy_ids: list[str]
    mirror_envs: int = 0
    heuristic_public_envs: int = 0
    heuristic_public_variant_envs: int = 0
    noleague_baseline_envs: int = 0
    champion_envs: int = 0
    recent_envs: int = 0
    hard_negative_envs: int = 0
    warmup_snapshot_envs: int = 0
    sampled_policy_envs: Counter[str] = field(default_factory=Counter)
    heuristic_public_policy_envs: Counter[str] = field(default_factory=Counter)
    heuristic_public_variant_policy_envs: Counter[str] = field(default_factory=Counter)
    noleague_baseline_policy_envs: Counter[str] = field(default_factory=Counter)
    champion_policy_envs: Counter[str] = field(default_factory=Counter)
    recent_policy_envs: Counter[str] = field(default_factory=Counter)
    hard_negative_policy_envs: Counter[str] = field(default_factory=Counter)
    warmup_snapshot_policy_envs: Counter[str] = field(default_factory=Counter)

    @classmethod
    def create(cls, sample_count: int) -> OpponentSamplingAccumulator:
        return cls(
            sample_count=int(sample_count),
            policy_ids=[""] * int(sample_count),
        )

    def record(self, *, group_name: str, positions: np.ndarray, policy_ids: Sequence[str]) -> None:
        env_count = int(positions.size)
        for idx, policy_id in zip(positions.tolist(), policy_ids, strict=True):
            self.policy_ids[int(idx)] = str(policy_id)
            if group_name != "mirror":
                self.sampled_policy_envs[str(policy_id)] += 1
        if group_name == "mirror":
            self.mirror_envs += env_count
        elif group_name == "heuristic_public":
            self.heuristic_public_envs += env_count
            self._policy_counter("heuristic_public").update(str(policy_id) for policy_id in policy_ids)
        elif group_name == "heuristic_public_variant":
            self.heuristic_public_variant_envs += env_count
            self._policy_counter("heuristic_public_variant").update(str(policy_id) for policy_id in policy_ids)
        elif group_name == "noleague_baseline":
            self.noleague_baseline_envs += env_count
            self._policy_counter("noleague_baseline").update(str(policy_id) for policy_id in policy_ids)
        elif group_name == "hard_negative":
            self.hard_negative_envs += env_count
            self._policy_counter("hard_negative").update(str(policy_id) for policy_id in policy_ids)
        elif group_name == "champion":
            self.champion_envs += env_count
            self._policy_counter("champion").update(str(policy_id) for policy_id in policy_ids)
        elif group_name == "warmup_snapshot":
            self.warmup_snapshot_envs += env_count
            self._policy_counter("warmup_snapshot").update(str(policy_id) for policy_id in policy_ids)
        else:
            self.recent_envs += env_count
            self._policy_counter("recent").update(str(policy_id) for policy_id in policy_ids)

    def result(self) -> OpponentSamplingResult:
        return OpponentSamplingResult(
            policy_ids=tuple(str(policy_id) for policy_id in self.policy_ids),
            sampled_envs=self.sample_count - self.mirror_envs,
            mirror_envs=self.mirror_envs,
            heuristic_public_envs=self.heuristic_public_envs,
            heuristic_public_variant_envs=self.heuristic_public_variant_envs,
            noleague_baseline_envs=self.noleague_baseline_envs,
            champion_envs=self.champion_envs,
            recent_envs=self.recent_envs,
            hard_negative_envs=self.hard_negative_envs,
            warmup_snapshot_envs=self.warmup_snapshot_envs,
            sampled_policy_envs=count_items(self._policy_counter("sampled")),
            heuristic_public_policy_envs=count_items(self._policy_counter("heuristic_public")),
            heuristic_public_variant_policy_envs=count_items(self._policy_counter("heuristic_public_variant")),
            noleague_baseline_policy_envs=count_items(self._policy_counter("noleague_baseline")),
            champion_policy_envs=count_items(self._policy_counter("champion")),
            recent_policy_envs=count_items(self._policy_counter("recent")),
            hard_negative_policy_envs=count_items(self._policy_counter("hard_negative")),
            warmup_snapshot_policy_envs=count_items(self._policy_counter("warmup_snapshot")),
        )

    def _policy_counter(self, group_name: str) -> Counter[str]:
        return {
            "sampled": self.sampled_policy_envs,
            "heuristic_public": self.heuristic_public_policy_envs,
            "heuristic_public_variant": self.heuristic_public_variant_policy_envs,
            "noleague_baseline": self.noleague_baseline_policy_envs,
            "champion": self.champion_policy_envs,
            "recent": self.recent_policy_envs,
            "hard_negative": self.hard_negative_policy_envs,
            "warmup_snapshot": self.warmup_snapshot_policy_envs,
        }[group_name]


def empty_opponent_sampling_result() -> OpponentSamplingResult:
    return OpponentSamplingResult(policy_ids=())


def count_items(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple((str(policy_id), int(count)) for policy_id, count in sorted(counts.items()) if int(count) > 0)


__all__ = [
    "OpponentSamplingAccumulator",
    "OpponentSamplingResult",
    "RuntimeOpponentGroup",
    "RuntimeOpponentSamplingPlan",
    "count_items",
    "empty_opponent_sampling_result",
]
