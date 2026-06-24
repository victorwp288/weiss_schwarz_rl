"""Packed legal-candidate metrics emitted by learner forward passes."""

from __future__ import annotations


def build_packed_candidate_metrics(
    *,
    candidate_count: int,
    row_count: int,
    active_candidate_count: int | None = None,
    active_row_count: int | None = None,
) -> dict[str, float]:
    metrics = {
        "packed_candidate_count": float(candidate_count),
        "packed_candidate_rows": float(row_count),
        "avg_legal_actions_per_row": float(candidate_count / max(row_count, 1)),
    }
    if active_candidate_count is not None and active_row_count is not None:
        metrics.update(
            {
                "packed_candidate_train_count": float(active_candidate_count),
                "packed_candidate_train_rows": float(active_row_count),
            }
        )
    return metrics


__all__ = ["build_packed_candidate_metrics"]
