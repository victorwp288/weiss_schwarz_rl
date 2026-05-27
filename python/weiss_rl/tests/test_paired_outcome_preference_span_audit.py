from __future__ import annotations

from pathlib import Path

import numpy as np

from weiss_rl.experiments.paired_outcome_preference_span_audit import (
    PairedOutcomePreferenceSpanAuditConfig,
    build_paired_outcome_preference_span_audit,
)
from weiss_rl.replay.trajectory_bc import BC_DATASET_FORMAT, ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset


def _write_dataset(path: Path, *, repeated_edges: bool) -> Path:
    time_steps = 6
    episodes = 4
    obs = np.zeros((time_steps, episodes, 3), dtype=np.float32)
    actor = np.zeros((time_steps, episodes), dtype=np.int64)
    to_play = np.zeros((time_steps, episodes), dtype=np.int64)
    actions = np.zeros((time_steps, episodes), dtype=np.int64)
    actions[:, 0] = np.asarray([10, 11, 12, 12, 13, 0])
    actions[:, 1] = np.asarray([10, 21, 22, 12, 23, 0])
    actions[:, 2] = np.asarray([10, 11, 12, 12, 13, 0])
    actions[:, 3] = np.asarray([10, 21, 22, 12, 13, 0] if repeated_edges else [10, 31, 32, 12, 13, 0])
    row_count = time_steps * episodes
    legal_offsets = np.arange(0, row_count * 4 + 1, 4, dtype=np.int64)
    legal_ids = np.tile(np.asarray([0, 11, 21, 31], dtype=np.int64), row_count)
    metadata = {
        "format": BC_DATASET_FORMAT,
        "episode_count": episodes,
        "time_steps": time_steps,
        "row_count": row_count,
        "train_rows": row_count,
        "selected_bundles": [
            _bundle(pair_id=0, role=1, source_pair_index=16, episode_seed=101),
            _bundle(pair_id=0, role=0, source_pair_index=16, episode_seed=101),
            _bundle(pair_id=1, role=1, source_pair_index=205, episode_seed=202),
            _bundle(pair_id=1, role=0, source_pair_index=205, episode_seed=202),
        ],
    }
    save_replay_trajectory_bc_dataset(
        path,
        ReplayTrajectoryDataset(
            obs=obs,
            actor=actor,
            to_play_seat=to_play,
            actions=actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=np.zeros((len(legal_ids), 4), dtype=np.int64),
            teacher_family=np.zeros((time_steps, episodes), dtype=np.int64),
            teacher_slot=np.zeros((time_steps, episodes), dtype=np.int64),
            teacher_move_source=np.zeros((time_steps, episodes), dtype=np.int64),
            teacher_attack_type=np.zeros((time_steps, episodes), dtype=np.int64),
            teacher_action=actions.copy(),
            teacher_valid=np.ones((time_steps, episodes), dtype=np.bool_),
            policy_train_mask=np.ones((time_steps, episodes), dtype=np.bool_),
            reset_before_step=np.zeros((time_steps, episodes), dtype=np.bool_),
            metadata=metadata,
        ),
    )
    return path


def _bundle(*, pair_id: int, role: int, source_pair_index: int, episode_seed: int) -> dict:
    return {
        "preference_pair_id": pair_id,
        "preference_role": role,
        "preference_role_label": "preferred" if role == 1 else "rejected",
        "source_opponent_policy_id": "B2 HeuristicPublic",
        "source_pair_index": source_pair_index,
        "episode_seed": episode_seed,
        "swap_index": 0,
    }


def test_span_audit_finds_repeated_compact_action_pattern(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz", repeated_edges=True)

    report = build_paired_outcome_preference_span_audit(
        PairedOutcomePreferenceSpanAuditConfig(dataset_path=dataset, max_compact_span_width=3)
    )

    assert report["passed"] is True
    assert report["span_gate"]["passed"] is True
    assert report["summary"]["passing_opponents"] == ["B2 HeuristicPublic"]
    assert report["complete_pair_count"] == 2
    assert report["pair_summaries"][0]["earliest_span"]["start_step"] == 1
    assert report["pair_summaries"][0]["earliest_span"]["end_step"] == 2
    keys = {row["key"] for row in report["repeated_action_label_edges"]}
    assert "action:11->action:21" in keys
    assert "B2 HeuristicPublic" in report["span_gate"]["passing_opponents"]


def test_span_audit_fails_gate_without_repeated_pattern(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset.npz", repeated_edges=False)

    report = build_paired_outcome_preference_span_audit(
        PairedOutcomePreferenceSpanAuditConfig(dataset_path=dataset, max_compact_span_width=3)
    )

    assert report["passed"] is False
    assert report["failures"] == [{"reason": "no_compact_repeated_span_pattern"}]
    assert report["span_gate"]["passed"] is False
    assert report["span_gate"]["failures"] == [{"reason": "no_compact_repeated_span_pattern"}]
    assert report["repeated_action_label_edges"] == []
