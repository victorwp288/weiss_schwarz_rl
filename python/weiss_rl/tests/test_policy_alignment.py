from __future__ import annotations

import math

import numpy as np
import pytest

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.eval.policies.alignment import PolicyAlignmentAccumulator


def _catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_space_size": 6,
                "pass_action_id": 0,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 2], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "pass", "base": 0, "count": 1},
                    {"name": "clock_from_hand", "base": 1, "count": 2},
                    {"name": "attack", "base": 3, "count": 3},
                ],
                "attack_type_encoding": [["front", 0], ["side", 1], ["direct", 2]],
            }
        }
    )


def test_policy_alignment_accumulator_reports_rank_family_and_probability_mass() -> None:
    accumulator = PolicyAlignmentAccumulator(action_catalog=_catalog())
    logits = np.asarray([0.0, 2.0, 3.0, -1.0, -9.0, -9.0], dtype=np.float32)

    accumulator.add(
        model_logits=logits,
        legal_ids=np.asarray([0, 1, 2, 3], dtype=np.uint32),
        reference_action_id=1,
    )

    summary = accumulator.summary()
    expected_probability = math.exp(2.0) / sum(math.exp(value) for value in [0.0, 2.0, 3.0, -1.0])
    expected_family_probability = (math.exp(2.0) + math.exp(3.0)) / sum(
        math.exp(value) for value in [0.0, 2.0, 3.0, -1.0]
    )

    assert summary["compared_steps"] == 1
    assert summary["model_matches_reference_top_action_rate"] == 0.0
    assert summary["model_matches_reference_top_action_family_rate"] == 1.0
    assert summary["model_median_rank_of_reference_top_action"] == 2.0
    assert summary["model_mean_probability_on_reference_top_action"] == pytest.approx(expected_probability)
    assert summary["model_mean_probability_on_reference_top_action_family"] == pytest.approx(
        expected_family_probability
    )
    assert summary["model_reference_top_action_same_family_logit_margin_percentiles"]["mean"] == pytest.approx(-1.0)
    assert summary["reference_top_family_summaries"][0]["family"] == "clock_from_hand"
    assert summary["top_action_family_confusions"] == [
        {"reference_family": "clock_from_hand", "model_family": "clock_from_hand", "count": 1}
    ]


def test_policy_alignment_accumulator_tracks_skipped_illegal_reference_actions() -> None:
    accumulator = PolicyAlignmentAccumulator(action_catalog=_catalog())

    accumulator.add(
        model_logits=np.zeros((6,), dtype=np.float32),
        legal_ids=np.asarray([0, 1], dtype=np.uint32),
        reference_action_id=3,
    )

    summary = accumulator.summary()
    assert summary["compared_steps"] == 0
    assert summary["skipped_steps"] == 1
