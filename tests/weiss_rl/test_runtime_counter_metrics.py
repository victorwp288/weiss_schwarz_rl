from __future__ import annotations

import pytest
from weiss_rl.runtime.components.metrics import runtime_counter_totals, runtime_outcome_metrics

from .runtime_metrics_test_support import _runtime_unroll


def test_runtime_counter_totals_sums_present_unroll_counters() -> None:
    totals = runtime_counter_totals(
        [
            _runtime_unroll(
                t=1,
                n=1,
                behavior_policy_version=1,
                counters={"pass_actions": 2, "max_consecutive_main_moves": 1},
            ),
            _runtime_unroll(t=1, n=1, behavior_policy_version=1, counters=None),
            _runtime_unroll(
                t=1,
                n=1,
                behavior_policy_version=1,
                counters={"pass_actions": 3, "main_move_actions": 4, "max_consecutive_main_moves": 2},
            ),
            _runtime_unroll(
                t=1,
                n=1,
                behavior_policy_version=1,
                counters={"outcome_v1|w|policy_a": 7},
            ),
        ]
    )

    assert totals == {"pass_actions": 5.0, "main_move_actions": 4.0, "max_consecutive_main_moves": 2.0}


def test_runtime_outcome_metrics_export_opponent_win_rates() -> None:
    metrics = runtime_outcome_metrics(
        [
            _runtime_unroll(
                t=1,
                n=1,
                behavior_policy_version=1,
                counters={
                    "outcome_v1|w|b1_noleague_baseline": 3,
                    "outcome_v1|l|b1_noleague_baseline": 1,
                    "outcome_v1|d|b1_noleague_baseline": 1,
                    "outcome_v1|t|b1_noleague_baseline": 1,
                    "outcome_v1|w|B2 HeuristicPublic": 2,
                },
            ),
            _runtime_unroll(
                t=1,
                n=1,
                behavior_policy_version=1,
                counters={"outcome_v1|l|b1_noleague_baseline": 1},
            ),
        ]
    )

    assert metrics["collector_outcome_vs_b1_noleague_baseline_wins"] == pytest.approx(3.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_losses"] == pytest.approx(2.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_draws"] == pytest.approx(1.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_timeouts"] == pytest.approx(1.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_games"] == pytest.approx(7.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_win_rate"] == pytest.approx(3.0 / 7.0)
    assert metrics["collector_outcome_vs_b1_noleague_baseline_decisive_win_rate"] == pytest.approx(3.0 / 5.0)
    assert metrics["collector_outcome_vs_b2_heuristicpublic_wins"] == pytest.approx(2.0)
