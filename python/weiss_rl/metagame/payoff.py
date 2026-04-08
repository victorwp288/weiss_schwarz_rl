"""Payoff matrix helpers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from weiss_rl.eval import EvalGameRecord
from weiss_rl.eval.payoff_folding import PairedSeedGroupKey, PayoffFoldScheme, paired_seed_group_key, paired_seed_score

__all__ = [
    "build_p_mean_and_counts",
    "write_p_mean_csv",
    "write_payoff_counts_json",
    "write_payoff_artifacts",
    "to_antisymmetric",
]


def to_antisymmetric(payoff: np.ndarray) -> np.ndarray:
    arr = np.asarray(payoff, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("payoff must be a square matrix")
    return 0.5 * (arr - arr.T)


def build_p_mean_and_counts(
    records: Sequence[EvalGameRecord],
    *,
    scheme: PayoffFoldScheme = "S0",
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build a payoff mean matrix and paired-seed counts from evaluation records."""
    if not records:
        raise ValueError("records must contain at least one EvalGameRecord")

    _require_single_shared_value(records, selector=lambda record: record.config_hash256, name="config_hash256")
    _require_single_shared_value(records, selector=lambda record: record.spec_hash256, name="spec_hash256")

    policy_ids = sorted(
        {
            *{record.focal_policy_id for record in records},
            *{record.opponent_policy_id for record in records},
        }
    )

    pair_groups: dict[PairedSeedGroupKey, list[EvalGameRecord]] = defaultdict(list)
    for record in records:
        pair_groups[paired_seed_group_key(record)].append(record)

    directed_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    directed_counts: dict[tuple[str, str], int] = defaultdict(int)

    for group in pair_groups.values():
        score = paired_seed_score(group, scheme=scheme)
        if score is None:
            continue
        focal_policy_id = group[0].focal_policy_id
        opponent_policy_id = group[0].opponent_policy_id
        directed_scores[(focal_policy_id, opponent_policy_id)].append(score)
        directed_counts[(focal_policy_id, opponent_policy_id)] += 1

    n = len(policy_ids)
    p_mean = np.full((n, n), np.nan, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        p_mean[i, i] = 0.5

    for i in range(n):
        for j in range(i + 1, n):
            policy_i = policy_ids[i]
            policy_j = policy_ids[j]
            count_ij = directed_counts.get((policy_i, policy_j), 0)
            count_ji = directed_counts.get((policy_j, policy_i), 0)
            scores_ij = directed_scores.get((policy_i, policy_j), [])
            scores_ji = directed_scores.get((policy_j, policy_i), [])

            if count_ij == 0 and count_ji == 0:
                continue

            mean_ij = _mean(scores_ij) if count_ij else None
            mean_ji = _mean(scores_ji) if count_ji else None

            if mean_ij is None:
                if mean_ji is None:
                    raise RuntimeError("expected one directed mean when directed counts are nonzero")
                combined_mean = 1.0 - mean_ji
                combined_count = count_ji
            elif mean_ji is None:
                combined_mean = mean_ij
                combined_count = count_ij
            else:
                combined_mean = (mean_ij * count_ij + (1.0 - mean_ji) * count_ji) / (count_ij + count_ji)
                combined_count = count_ij + count_ji

            p_mean[i, j] = combined_mean
            p_mean[j, i] = 1.0 - combined_mean
            counts[i, j] = combined_count
            counts[j, i] = combined_count

    return p_mean, counts, tuple(policy_ids)


def write_p_mean_csv(path: Path, p_mean: np.ndarray, policy_ids: Sequence[str]) -> None:
    """Write a payoff mean matrix to a CSV file with policy labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(policy_ids))
        for row_index, policy_id in enumerate(policy_ids):
            row = [policy_id]
            for value in p_mean[row_index]:
                if np.isnan(value):
                    row.append("")
                else:
                    row.append(f"{float(value):.6f}")
            writer.writerow(row)


def write_payoff_counts_json(path: Path, counts: np.ndarray, policy_ids: Sequence[str]) -> None:
    """Write payoff counts as a nested JSON object by policy id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, int]] = {}
    for row_index, policy_id in enumerate(policy_ids):
        payload[policy_id] = {
            policy_ids[col_index]: int(counts[row_index, col_index]) for col_index in range(len(policy_ids))
        }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_payoff_artifacts(
    p_mean_csv: Path,
    payoff_counts_json: Path,
    p_mean: np.ndarray,
    counts: np.ndarray,
    policy_ids: Sequence[str],
) -> None:
    write_p_mean_csv(p_mean_csv, p_mean, policy_ids)
    write_payoff_counts_json(payoff_counts_json, counts, policy_ids)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _require_single_shared_value(
    records: Sequence[EvalGameRecord],
    *,
    selector: Callable[[EvalGameRecord], str],
    name: str,
) -> None:
    values = {selector(record) for record in records}
    if len(values) != 1:
        raise ValueError(f"records must share exactly one {name}")
