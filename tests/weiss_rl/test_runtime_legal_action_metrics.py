from __future__ import annotations

import pytest

from .runtime_metrics_test_support import _build_runtime_metrics_with_defaults, _runtime_unroll


def test_build_runtime_metrics_uses_total_actions_for_average_legal_rows_when_route_counts_are_absent() -> None:
    metrics, _next_cumulative = _build_runtime_metrics_with_defaults(
        selected=[
            _runtime_unroll(
                t=2,
                n=2,
                behavior_policy_version=1,
                counters={"packed_candidate_count": 28, "total_actions": 4},
            )
        ],
    )

    assert metrics["avg_legal_actions_per_row"] == pytest.approx(7.0)
