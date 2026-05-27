"""Shared opponent-context coverage summaries for offline gates."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def summarize_opponent_context_coverage(
    opponent_policy_ids: Sequence[object],
    opponent_context_indices: Sequence[object],
) -> dict[str, Any]:
    """Return JSON-safe per-opponent context coverage for a replay/eval surface."""

    policy_ids = [str(policy_id).strip() for policy_id in opponent_policy_ids]
    context_indices = [int(index) for index in opponent_context_indices]
    if len(policy_ids) != len(context_indices):
        raise ValueError(
            "opponent_policy_ids and opponent_context_indices must have the same length, "
            f"got {len(policy_ids)} and {len(context_indices)}"
        )

    counts = Counter(policy_ids)
    by_policy: dict[str, list[int]] = defaultdict(list)
    for policy_id, context_index in zip(policy_ids, context_indices, strict=True):
        by_policy[policy_id].append(int(context_index))

    opponent_summaries: list[dict[str, Any]] = []
    missing_opponent_ids: list[str] = []
    mapped_opponent_ids: list[str] = []
    for policy_id in sorted(by_policy):
        indices = by_policy[policy_id]
        context_count = sum(1 for index in indices if index != 0)
        missing_count = len(indices) - context_count
        if policy_id and missing_count > 0:
            missing_opponent_ids.append(policy_id)
        if policy_id and context_count > 0:
            mapped_opponent_ids.append(policy_id)
        opponent_summaries.append(
            {
                "opponent_policy_id": policy_id,
                "episode_count": int(counts[policy_id]),
                "context_episode_count": int(context_count),
                "missing_context_episode_count": int(missing_count),
                "context_indices": sorted({int(index) for index in indices}),
            }
        )

    context_episode_count = sum(1 for index in context_indices if index != 0)
    empty_opponent_count = sum(1 for policy_id in policy_ids if not policy_id)
    return {
        "episode_count": len(policy_ids),
        "context_episode_count": int(context_episode_count),
        "missing_context_episode_count": int(len(policy_ids) - context_episode_count),
        "empty_opponent_id_episode_count": int(empty_opponent_count),
        "missing_context_opponent_policy_ids": missing_opponent_ids,
        "mapped_opponent_policy_ids": mapped_opponent_ids,
        "opponents": opponent_summaries,
    }


def context_coverage_failures_from_report(
    report: Mapping[str, Any],
    *,
    coverage_key: str,
    context_count_key: str,
    prefix: str = "",
    require_all_episodes: bool = True,
) -> list[str]:
    """Return gate failure strings for an opponent-context coverage report."""

    failures: list[str] = []
    label = f"{prefix}_" if prefix else ""
    context_count = _safe_int(report.get(context_count_key))
    if context_count <= 0:
        failures.append(
            "missing_opponent_context" if not prefix or prefix == "current" else f"missing_{prefix}_context"
        )

    episode_count = _safe_int(report.get("episode_count"))
    if require_all_episodes and episode_count > 0 and context_count < episode_count:
        failures.append(f"{label}context_episodes_below:{context_count}<{episode_count}")

    coverage = report.get(coverage_key)
    if isinstance(coverage, Mapping):
        coverage_episode_count = _safe_int(coverage.get("episode_count"))
        coverage_context_count = _safe_int(coverage.get("context_episode_count"))
        if require_all_episodes and coverage_episode_count > 0 and coverage_context_count < coverage_episode_count:
            failure = f"{label}coverage_context_episodes_below:{coverage_context_count}<{coverage_episode_count}"
            if failure not in failures:
                failures.append(failure)
        empty_count = _safe_int(coverage.get("empty_opponent_id_episode_count"))
        if empty_count > 0:
            failures.append(f"{label}empty_opponent_id_episodes:{empty_count}")
        missing_ids = coverage.get("missing_context_opponent_policy_ids")
        if isinstance(missing_ids, list) and missing_ids:
            failures.append(f"{label}missing_context_opponents:" + ",".join(str(item) for item in missing_ids))
    return failures


def _safe_int(value: object) -> int:
    if value is None:
        return 0
    return int(value)


__all__ = [
    "context_coverage_failures_from_report",
    "summarize_opponent_context_coverage",
]
