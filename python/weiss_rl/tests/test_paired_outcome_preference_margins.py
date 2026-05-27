from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.experiments.paired_outcome_preference_margins import preference_margin_rows_from_logps


def test_preference_margin_rows_from_logps_groups_complete_pairs() -> None:
    selected_bundles = [
        {
            "preference_pair_id": 0,
            "preference_role": 1,
            "preference_role_label": "preferred",
            "merge_source_dataset_label": "learned_repair",
            "source_opponent_policy_id": "policy_000003",
            "source_pair_index": 205,
        },
        {
            "preference_pair_id": 0,
            "preference_role": 0,
            "preference_role_label": "rejected",
            "merge_source_dataset_label": "learned_repair",
            "source_opponent_policy_id": "policy_000003",
            "source_pair_index": 205,
        },
    ]
    current = np.asarray([[-1.0, -3.0], [-2.0, -4.0]], dtype=np.float32)
    reference = np.asarray([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float32)
    pair_ids = np.asarray([[0, 0], [0, 0]], dtype=np.int64)
    roles = np.asarray([[1, 0], [1, 0]], dtype=np.int64)
    mask = np.ones((2, 2), dtype=np.bool_)

    rows = preference_margin_rows_from_logps(
        selected_bundles=selected_bundles,
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_roles=roles,
        loss_mask=mask,
        aggregation="mean",
    )

    assert len(rows) == 1
    assert rows[0]["group_label"] == "learned_repair"
    assert rows[0]["preferred_rows"] == 2
    assert rows[0]["rejected_rows"] == 2
    assert rows[0]["current_raw_margin"] == pytest.approx(2.0)
    assert rows[0]["reference_raw_margin"] == pytest.approx(0.0)
    assert rows[0]["dpo_margin"] == pytest.approx(2.0)


def test_preference_margin_rows_from_logps_falls_back_to_source_dataset_label() -> None:
    selected_bundles = [
        {
            "preference_pair_id": 0,
            "preference_role": 1,
            "source_dataset_label": "fixed_protect",
            "source_opponent_policy_id": "B2 HeuristicPublic",
        },
        {
            "preference_pair_id": 0,
            "preference_role": 0,
            "source_dataset_label": "fixed_protect",
            "source_opponent_policy_id": "B2 HeuristicPublic",
        },
    ]
    current = np.asarray([[-1.0, -2.0]], dtype=np.float32)
    reference = np.asarray([[-1.0, -2.0]], dtype=np.float32)
    pair_ids = np.asarray([[0, 0]], dtype=np.int64)
    roles = np.asarray([[1, 0]], dtype=np.int64)

    rows = preference_margin_rows_from_logps(
        selected_bundles=selected_bundles,
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_roles=roles,
        loss_mask=np.ones((1, 2), dtype=np.bool_),
    )

    assert rows[0]["group_label"] == "fixed_protect"


def test_preference_margin_rows_from_logps_skips_incomplete_pairs() -> None:
    current = np.asarray([[-1.0, -3.0]], dtype=np.float32)
    reference = np.asarray([[-1.0, -3.0]], dtype=np.float32)
    pair_ids = np.asarray([[0, 1]], dtype=np.int64)
    roles = np.asarray([[1, 0]], dtype=np.int64)
    mask = np.ones((1, 2), dtype=np.bool_)

    rows = preference_margin_rows_from_logps(
        selected_bundles=[],
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_roles=roles,
        loss_mask=mask,
    )

    assert rows == []
