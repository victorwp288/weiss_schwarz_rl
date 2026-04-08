"""Payoff folding helpers for seat-swapped evaluation records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

from weiss_rl.eval.harness import EvalGameRecord

PayoffFoldScheme = Literal["S0", "S1", "S2"]

__all__ = [
    "PayoffFoldScheme",
    "fold_game_payoff",
    "paired_seed_mean_score",
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
        return 0.5
    return None


def paired_seed_score(records: Sequence[EvalGameRecord], *, scheme: PayoffFoldScheme) -> float | None:
    normalized_scheme = _normalize_scheme(scheme)
    pair_records = _validate_pair_records(records)
    scores = [fold_game_payoff(record.outcome, scheme=normalized_scheme) for record in pair_records]
    included_scores = [score for score in scores if score is not None]
    if not included_scores:
        return None
    return _mean(included_scores)


def paired_seed_scores(records: Sequence[EvalGameRecord], *, scheme: PayoffFoldScheme) -> tuple[float, ...]:
    normalized_scheme = _normalize_scheme(scheme)
    if not records:
        raise ValueError("paired_seed_scores requires at least one record")

    pair_groups: dict[tuple[str | None, int], list[EvalGameRecord]] = defaultdict(list)
    for record in records:
        pair_groups[_pair_group_key(record)].append(record)

    pair_scores: list[float] = []
    for pair_key in sorted(pair_groups, key=_pair_group_sort_key):
        score = paired_seed_score(pair_groups[pair_key], scheme=normalized_scheme)
        if score is not None:
            pair_scores.append(score)
    return tuple(pair_scores)


def paired_seed_mean_score(records: Sequence[EvalGameRecord], *, scheme: PayoffFoldScheme) -> float:
    pair_scores = paired_seed_scores(records, scheme=scheme)
    if pair_scores:
        return _mean(pair_scores)
    raise ValueError("S2 excluded all paired seeds")


def _validate_pair_records(records: Sequence[EvalGameRecord]) -> tuple[EvalGameRecord, EvalGameRecord]:
    if len(records) != 2:
        raise ValueError(f"paired seed group must contain exactly 2 records, got {len(records)}")

    if len({_pair_group_key(record) for record in records}) != 1:
        raise ValueError("paired seed records must share pair_index within one run_id256")
    pair_index = int(records[0].pair_index)

    _require_shared_value(records, selector=lambda record: record.run_id256, name="run_id256")
    _require_shared_value(records, selector=lambda record: int(record.episode_seed), name="episode_seed")
    _require_shared_value(records, selector=lambda record: record.focal_policy_id, name="focal_policy_id")
    _require_shared_value(records, selector=lambda record: record.opponent_policy_id, name="opponent_policy_id")

    records_by_swap: dict[int, EvalGameRecord] = {}
    for record in records:
        swap_index = int(record.swap_index)
        if swap_index not in (0, 1):
            raise ValueError(f"pair_index {pair_index} must use swap_index 0 or 1, got {swap_index}")
        if swap_index in records_by_swap:
            raise ValueError(f"pair_index {pair_index} must contain swap_index 0 and 1 exactly once")
        records_by_swap[swap_index] = record

    if set(records_by_swap) != {0, 1}:
        raise ValueError(f"pair_index {pair_index} must contain swap_index 0 and 1 exactly once")

    first = records_by_swap[0]
    second = records_by_swap[1]

    if int(first.focal_seat) != 0 or int(second.focal_seat) != 1:
        raise ValueError(f"pair_index {pair_index} must swap focal seats across the pair")
    if first.seat0_policy_id != first.focal_policy_id or first.seat1_policy_id != first.opponent_policy_id:
        raise ValueError(f"pair_index {pair_index} has inconsistent seat assignment for swap_index 0")
    if second.seat0_policy_id != second.opponent_policy_id or second.seat1_policy_id != second.focal_policy_id:
        raise ValueError(f"pair_index {pair_index} has inconsistent seat assignment for swap_index 1")

    return first, second


def _require_shared_value(
    records: Sequence[EvalGameRecord],
    *,
    selector: Callable[[EvalGameRecord], object],
    name: str,
) -> None:
    values = {selector(record) for record in records}
    if len(values) != 1:
        pair_index = int(records[0].pair_index)
        raise ValueError(f"pair_index {pair_index} must share {name}")


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


def _pair_group_key(record: EvalGameRecord) -> tuple[str | None, int]:
    return (record.run_id256, int(record.pair_index))


def _pair_group_sort_key(pair_key: tuple[str | None, int]) -> tuple[str, int]:
    run_id256, pair_index = pair_key
    return ("" if run_id256 is None else run_id256, pair_index)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)
