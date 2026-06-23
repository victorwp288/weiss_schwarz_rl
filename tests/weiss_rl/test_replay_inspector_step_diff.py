from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE
from weiss_rl.replay.bundles import ReplayStep
from weiss_rl.replay.inspection_step_diffs import build_step_diff
from weiss_rl.replay.inspection_summaries import summarize_step_diffs

from .replay_inspector_test_support import _heuristic_spec_bundle


def test_step_diff_reports_top_logit_and_probability_margins() -> None:
    logits_a = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_a[1] = 3.0
    logits_a[2] = 2.25
    logits_a[3] = 2.0
    logits_b[2] = 5.0

    diff = build_step_diff(
        step_index=0,
        expected_step=ReplayStep(
            t=0,
            decision_id=10,
            actor=0,
            action=1,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_fingerprint64=0,
        ),
        raw_legal_ids=np.array([1, 2, 3], dtype=np.uint32),
        legal_ids_a=np.array([1, 2, 3], dtype=np.uint32),
        legal_ids_b=np.array([1, 2, 3], dtype=np.uint32),
        logits_a=logits_a,
        logits_b=logits_b,
        top_actions=2,
        action_catalog=None,
    )
    summarized = summarize_step_diffs([diff], top_k=1)

    assert diff["policy_a_top_logit_margin"] == pytest.approx(0.75)
    assert diff["policy_a_gap_from_top_logit_to_policy_b_top_action"] == pytest.approx(0.75)
    assert diff["policy_a_top_probability_margin"] > 0.0
    assert summarized["policy_a_top_logit_margin_percentiles"]["p50"] == pytest.approx(0.75)
    assert summarized["policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles"]["p50"] == pytest.approx(0.75)
    assert summarized["policy_a_probability_on_policy_b_top_action_percentiles"]["count"] == 1


def test_step_diff_reports_policy_b_top_family_margin_summaries() -> None:
    action_catalog = ActionCatalog.from_spec_bundle(_heuristic_spec_bundle())
    logits_a = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_b = np.zeros((GLOBAL_ACTION_SPACE_SIZE,), dtype=np.float32)
    logits_a[51] = 3.0
    logits_a[472] = 1.0
    logits_a[473] = 2.0
    logits_a[474] = 1.5
    logits_b[473] = 5.0

    diff = build_step_diff(
        step_index=0,
        expected_step=ReplayStep(
            t=0,
            decision_id=10,
            actor=0,
            action=51,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_fingerprint64=0,
        ),
        raw_legal_ids=np.array([51, 472, 473, 474], dtype=np.uint32),
        legal_ids_a=np.array([51, 472, 473, 474], dtype=np.uint32),
        legal_ids_b=np.array([51, 472, 473, 474], dtype=np.uint32),
        logits_a=logits_a,
        logits_b=logits_b,
        top_actions=2,
        action_catalog=action_catalog,
    )
    summarized = summarize_step_diffs([diff], top_k=1)

    assert diff["policy_a_top_action"]["family"] == "pass"
    assert diff["policy_b_top_action"]["family"] == "attack"
    assert diff["policy_a_policy_b_top_action_same_family_logit_margin"] == pytest.approx(0.5)
    assert summarized["policy_a_policy_b_top_action_same_family_logit_margin_percentiles"]["p50"] == pytest.approx(0.5)
    assert summarized["policy_b_top_family_summaries"][0]["family"] == "attack"
    assert summarized["policy_b_top_family_summaries"][0]["count"] == 1
    assert summarized["policy_b_top_family_summaries"][0][
        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles"
    ]["p50"] == pytest.approx(0.5)
    assert summarized["policy_b_top_family_summaries"][0]["policy_a_matches_policy_b_top_action_family_rate"] == 0.0
    assert summarized["policy_b_top_family_summaries"][0]["policy_b_top_action_legal_for_policy_a_rate"] == 1.0
    assert summarized["policy_b_top_family_summaries"][0]["policy_a_legal_surface_filter_rate"] == 0.0
