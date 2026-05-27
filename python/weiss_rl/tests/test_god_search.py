from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.eval.god_search import GodSearchConfig, GodSearchStats, top_k_legal_actions


def test_top_k_legal_actions_orders_by_logit_then_action_id() -> None:
    logits = np.array([0.0, 2.0, 2.0, -1.0, 3.0], dtype=np.float32)
    legal_ids = np.array([1, 2, 4], dtype=np.uint32)

    assert top_k_legal_actions(logits, legal_ids, top_k=3) == (4, 1, 2)


def test_god_search_config_mapping_preserves_explicit_same_world_label() -> None:
    config = GodSearchConfig.from_mapping(
        {
            "mode": "same_world_prefix_rollout",
            "top_k": 3,
            "rollouts_per_action": 2,
            "max_rollout_decisions": 80,
            "max_search_decisions_per_game": 12,
            "rollout_policy": "sample",
            "fail_on_prefix_mismatch": False,
        }
    )

    assert config.enabled
    assert config.mode == "same_world_prefix_rollout"
    assert config.top_k == 3
    assert config.rollouts_per_action == 2
    assert config.max_rollout_decisions == 80
    assert config.max_search_decisions_per_game == 12
    assert config.rollout_policy == "sample"
    assert not config.fail_on_prefix_mismatch


def test_god_search_stats_reports_changed_fraction() -> None:
    config = GodSearchConfig(mode="same_world_prefix_rollout")
    stats = GodSearchStats(trace_limit=1, search_decisions=4, changed_decisions=1)
    stats.add_trace({"decision_id": 10})
    stats.add_trace({"decision_id": 11})

    payload = stats.to_json_dict(config=config)

    assert payload["changed_fraction"] == pytest.approx(0.25)
    assert payload["counters"]["search_decisions"] == 4
    assert payload["traces"] == [{"decision_id": 10}]
