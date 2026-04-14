"""Payoff folding helpers for seat-swapped evaluation records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

from weiss_rl.eval.harness import EvalGameRecord

PayoffFoldScheme = Literal["S0", "S1", "S2"]
PairedSeedGroupKey = tuple[str, str, str, str, str | None, int]

__all__ = [
    "PairedSeedGroupKey",
    "PayoffFoldScheme",
    "fold_game_payoff",
    "paired_seed_group_key",
    "paired_seed_mean_score",
    "validated_paired_seed_groups",
    "paired_seed_score",
    "paired_seed_scores",
]


def fold_game_payoff(outcome: str, *, scheme: PayoffFoldScheme) -> float | None:
    normalized_outcome = _normalize_outcome(outcome)
    normalized_scheme = _normalize_scheme(scheme)

    if normalized_outcome == "W":
        return 1.0
    if normalized_outcome == "L":
        return 0.0
    if normalized_outcome == "D":
        return 0.5
    if normalized_scheme in ("S0", "S1"):
        return 0.0
    return None


def paired_seed_score(records: Sequence[EvalGameRecord], *, scheme: PayoffFoldScheme) -> float | None:
    normalized_scheme = _normalize_scheme(scheme)
    return _paired_seed_score(_validate_pair_records(records), scheme=normalized_scheme)


def validated_paired_seed_groups(
    records: Sequence[EvalGameRecord],
) -> tuple[tuple[EvalGameRecord, ...], ...]:
    if not records:
        raise ValueError("paired seed records require at least one record")

    pair_groups: dict[PairedSeedGroupKey, list[EvalGameRecord]] = defaultdict(list)
    for record in records:
        pair_groups[paired_seed_group_key(record)].append(record)

    return tuple(
        _validate_pair_records(pair_groups[group_key])
        for group_key in sorted(pair_groups, key=_paired_seed_group_sort_key)
    )


def paired_seed_scores(records: Sequence[EvalGameRecord], *, scheme: PayoffFoldScheme) -> tuple[float, ...]:
    normalized_scheme = _normalize_scheme(scheme)
    if not records:
        raise ValueError("paired_seed_scores requires at least one record")

    pair_scores: list[float] = []
    for pair_records in validated_paired_seed_groups(records):
        score = _paired_seed_score(pair_records, scheme=normalized_scheme)
        if score is not None:
            pair_scores.append(score)
    return tuple(pair_scores)


def paired_seed_mean_score(records: Sequence[EvalGameRecord], *, scheme: PayoffFoldScheme) -> float:
    pair_scores = paired_seed_scores(records, scheme=scheme)
    if pair_scores:
        return _mean(pair_scores)
    raise ValueError("S2 excluded all paired seeds")


def paired_seed_group_key(record: EvalGameRecord) -> PairedSeedGroupKey:
    return (
        record.focal_policy_id,
        record.opponent_policy_id,
        record.config_hash256,
        record.spec_hash256,
        record.run_id256,
        int(record.episode_seed),
    )


def _paired_seed_score(
    records: tuple[EvalGameRecord, ...],
    *,
    scheme: PayoffFoldScheme,
) -> float | None:
    scores = [fold_game_payoff(record.outcome, scheme=scheme) for record in records]
    included_scores = [score for score in scores if score is not None]
    if not included_scores:
        return None
    return _mean(included_scores)


def _validate_pair_records(records: Sequence[EvalGameRecord]) -> tuple[EvalGameRecord, ...]:
    if len(records) < 2:
        raise ValueError(f"paired seed group must contain at least 2 records, got {len(records)}")

    _require_shared_value(records, selector=lambda record: record.run_id256, name="run_id256")
    _require_shared_value(records, selector=lambda record: int(record.episode_seed), name="episode_seed")
    _require_shared_value(records, selector=lambda record: record.focal_policy_id, name="focal_policy_id")
    _require_shared_value(records, selector=lambda record: record.opponent_policy_id, name="opponent_policy_id")
    _require_shared_value(records, selector=lambda record: record.config_hash256, name="config_hash256")
    _require_shared_value(records, selector=lambda record: record.spec_hash256, name="spec_hash256")

    records_by_swap: dict[int, list[EvalGameRecord]] = defaultdict(list)
    for record in records:
        swap_index = int(record.swap_index)
        if swap_index not in (0, 1):
            raise ValueError(f"paired seed records must use swap_index 0 or 1, got {swap_index}")
        records_by_swap[swap_index].append(record)

    if set(records_by_swap) != {0, 1} or len(records_by_swap[0]) != len(records_by_swap[1]):
        raise ValueError("paired seed records must contain matching counts for swap_index 0 and 1")

    for record in records_by_swap[0]:
        if int(record.focal_seat) != 0:
            raise ValueError("paired seed records must keep focal_seat=0 for swap_index 0")
        if record.seat0_policy_id != record.focal_policy_id or record.seat1_policy_id != record.opponent_policy_id:
            raise ValueError("paired seed records have inconsistent seat assignment for swap_index 0")

    for record in records_by_swap[1]:
        if int(record.focal_seat) != 1:
            raise ValueError("paired seed records must keep focal_seat=1 for swap_index 1")
        if record.seat0_policy_id != record.opponent_policy_id or record.seat1_policy_id != record.focal_policy_id:
            raise ValueError("paired seed records have inconsistent seat assignment for swap_index 1")

    return tuple(records)


def _require_shared_value(
    records: Sequence[EvalGameRecord],
    *,
    selector: Callable[[EvalGameRecord], object],
    name: str,
) -> None:
    values = {selector(record) for record in records}
    if len(values) != 1:
        raise ValueError(f"paired seed records must share {name}")


def _normalize_outcome(outcome: str) -> str:
    normalized = outcome.strip().upper()
    if normalized in {"W", "L", "D", "T"}:
        return normalized
    raise ValueError(f"unknown outcome token: {outcome!r}")


def _normalize_scheme(scheme: str) -> PayoffFoldScheme:
    normalized = scheme.strip().upper()
    if normalized == "S0":
        return "S0"
    if normalized == "S1":
        return "S1"
    if normalized == "S2":
        return "S2"
    raise ValueError(f"unknown payoff fold scheme: {scheme!r}")


def _paired_seed_group_sort_key(group_key: PairedSeedGroupKey) -> tuple[str, str, str, str, str, int]:
    focal_policy_id, opponent_policy_id, config_hash256, spec_hash256, run_id256, episode_seed = group_key
    return (
        focal_policy_id,
        opponent_policy_id,
        config_hash256,
        spec_hash256,
        "" if run_id256 is None else run_id256,
        episode_seed,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)
