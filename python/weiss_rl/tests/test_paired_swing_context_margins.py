from __future__ import annotations

import numpy as np
import torch

from weiss_rl.experiments.paired_swing_context_margins import paired_swing_margin_rows_from_packed_scores
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def test_paired_swing_margin_rows_reports_opponent_context_and_action_logps() -> None:
    dataset = ReplayTrajectoryDataset(
        obs=np.zeros((1, 2, 4), dtype=np.float32),
        actor=np.zeros((1, 2), dtype=np.int8),
        to_play_seat=np.zeros((1, 2), dtype=np.int8),
        actions=np.asarray([[124, 104]], dtype=np.int64),
        legal_ids=np.asarray([104, 124, 104, 124], dtype=np.uint32),
        legal_offsets=np.asarray([0, 2, 4], dtype=np.uint32),
        legal_action_meta=np.zeros((4, 4), dtype=np.uint16),
        teacher_family=np.full((1, 2), -1, dtype=np.int32),
        teacher_slot=np.full((1, 2), -1, dtype=np.int32),
        teacher_move_source=np.full((1, 2), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, 2), -1, dtype=np.int32),
        teacher_action=np.asarray([[104, 124]], dtype=np.int32),
        teacher_valid=np.ones((1, 2), dtype=np.bool_),
        policy_train_mask=np.ones((1, 2), dtype=np.bool_),
        reset_before_step=np.zeros((1, 2), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": 2,
            "episode_count": 2,
            "time_steps": 1,
            "row_count": 2,
            "train_rows": 2,
            "selected_bundles": [
                {
                    "source_dataset_label": "b2",
                    "source_opponent_policy_id": "B2 HeuristicPublic",
                    "source_pair_index": 205,
                },
                {
                    "source_dataset_label": "learned",
                    "source_opponent_policy_id": "seed_policy_000003",
                    "source_pair_indices": [205],
                },
            ],
        },
    )

    rows = paired_swing_margin_rows_from_packed_scores(
        dataset,
        packed_scores=torch.tensor([0.0, 1.0, 2.0, 0.0]),
        positive_action_source="actions",
        negative_action_source="teacher_action",
        opponent_context_indices=np.asarray([3, 7], dtype=np.int64),
        report_action_ids=(104, 124),
    )

    assert [row["opponent_context_index"] for row in rows] == [3, 7]
    assert rows[0]["positive_action"] == 124
    assert rows[0]["positive_minus_negative_logp"] > 0.0
    assert rows[1]["positive_action"] == 104
    assert rows[1]["positive_minus_negative_logp"] > 0.0
    assert rows[0]["top_action"] == 124
    assert rows[1]["top_action"] == 104
    assert rows[0]["positive_rank"] == 1
    assert rows[0]["negative_rank"] == 2
    assert rows[1]["positive_rank"] == 1
    assert rows[1]["negative_rank"] == 2
    assert rows[0]["reported_action_logps"]["124"] > rows[0]["reported_action_logps"]["104"]
    assert rows[1]["reported_action_logps"]["104"] > rows[1]["reported_action_logps"]["124"]
