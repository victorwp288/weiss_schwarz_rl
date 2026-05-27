from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weiss_rl.experiments.paired_outcome_contrastive import (
    PairedOutcomeContrastiveSource,
    PairedOutcomeInspectionConfig,
    PairedOutcomeInspectionSource,
    apply_policy_b_top_action_overrides,
    build_paired_outcome_contrastive_source_dataset,
    inspect_paired_outcome_sources,
    sources_from_paired_flip_summary,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset
from weiss_rl.training.paired_swing_replay import paired_swing_distinct_train_row_count


def test_apply_policy_b_top_action_overrides_keeps_only_distinct_trainable_pairs(tmp_path: Path) -> None:
    dataset = _contrastive_source_dataset(tmp_path)

    contrastive, summary = apply_policy_b_top_action_overrides(
        dataset,
        override_rows=[
            {
                "bundle_name": "replay_a_pair000_swap0.zip",
                "step_index": 0,
                "teacher_action": 2,
            },
            {
                "bundle_name": "replay_a_pair000_swap0.zip",
                "step_index": 1,
                "teacher_action": 4,
            },
            {
                "bundle_name": "replay_b_pair001_swap0.zip",
                "step_index": 0,
                "teacher_action": 3,
            },
        ],
        source_label="fixed_B2",
        source_role="fixed_preserve",
        source_dataset_path=tmp_path / "source.npz",
        source_opponent_policy_id="B2 HeuristicPublic",
    )

    assert summary["train_rows"] == 1
    assert summary["distinct_train_rows"] == 1
    assert summary["kept_episode_count"] == 1
    assert summary["counters"]["written_train_rows"] == 1
    assert summary["counters"]["skipped_same_action"] == 1
    assert summary["counters"]["skipped_nontrainable_source_row"] == 1
    assert contrastive.actions.tolist() == [[1], [4]]
    assert contrastive.teacher_action.tolist() == [[2], [-1]]
    assert contrastive.policy_train_mask.tolist() == [[True], [False]]
    assert contrastive.metadata["selected_bundles"][0]["source_dataset_label"] == "fixed_B2"
    assert contrastive.metadata["selected_bundles"][0]["outcome_contrastive_role"] == "fixed_preserve"
    assert (
        paired_swing_distinct_train_row_count(
            contrastive,
            positive_action_source="actions",
            negative_action_source="teacher_action",
        )
        == 1
    )


def test_build_paired_outcome_contrastive_source_dataset_from_inspections(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.npz"
    save_replay_trajectory_bc_dataset(dataset_path, _contrastive_source_dataset(tmp_path))
    inspection_path = tmp_path / "inspection.json"
    inspection_path.write_text(
        json.dumps(
            {
                "bundle_path": (tmp_path / "replay_a_pair000_swap0.zip").as_posix(),
                "top_differences": [
                    {
                        "actor": 0,
                        "policy_a_matches_policy_b_top_action": False,
                        "policy_a_top_action": {"action": 1, "family": "clock_from_hand"},
                        "policy_b_top_action": {"action": 2, "family": "clock_from_hand"},
                        "step_index": 0,
                        "total_variation": 0.25,
                    },
                    {
                        "actor": 0,
                        "policy_a_matches_policy_b_top_action": True,
                        "policy_a_top_action": {"action": 4, "family": "attack"},
                        "policy_b_top_action": {"action": 4, "family": "attack"},
                        "step_index": 1,
                        "total_variation": 0.20,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    contrastive, summary = build_paired_outcome_contrastive_source_dataset(
        PairedOutcomeContrastiveSource(
            source_label="learned_policy_000002",
            source_role="learned_repair",
            source_dataset_path=dataset_path,
            inspection_jsons=(inspection_path,),
            source_opponent_policy_id="seed_policy_000002",
        ),
        min_total_variation=0.1,
    )

    assert contrastive.metadata["intended_auxiliary"] == "paired_swing_replay"
    assert summary["override_summary"]["row_count"] == 1
    assert summary["apply_summary"]["train_rows"] == 1
    assert summary["apply_summary"]["source_role"] == "learned_repair"


def test_sources_from_paired_flip_summary_filters_labels(tmp_path: Path) -> None:
    fallback_dataset = tmp_path / "source_b2" / "datasets" / "trajectory_bc_b2_heuristicpublic.npz"
    fallback_dataset.parent.mkdir(parents=True)
    fallback_dataset.write_bytes(b"placeholder")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generation": {
                    "sources": [
                        {
                            "opponent_policy_id": "B2 HeuristicPublic",
                            "source_label": "fixed_B2",
                            "dataset_path": "runs/fixed_b2.npz",
                            "output_run_dir": (tmp_path / "source_b2").as_posix(),
                        },
                        {
                            "opponent_policy_id": "B3 HeuristicPublicAggro",
                            "source_label": "fixed_B3",
                            "dataset_path": "runs/fixed_b3.npz",
                        },
                    ]
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    sources = sources_from_paired_flip_summary(
        summary_path,
        source_role="fixed_preserve",
        output_dir=tmp_path / "out",
        include_source_labels=("B2 HeuristicPublic",),
    )

    assert len(sources) == 1
    assert sources[0].source_label == "fixed_B2"
    assert sources[0].source_opponent_policy_id == "B2 HeuristicPublic"
    assert sources[0].source_dataset_path == fallback_dataset


def test_inspect_paired_outcome_sources_passes_snapshot_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "source.npz"
    save_replay_trajectory_bc_dataset(dataset_path, _contrastive_source_dataset(tmp_path))
    registry_path = tmp_path / "registry.json"
    seen: dict[str, Path | None] = {}

    def fake_inspect_replay_bundle(**kwargs: object) -> dict[str, object]:
        seen["snapshot_registry_path"] = kwargs.get("snapshot_registry_path")  # type: ignore[assignment]
        return {
            "bundle_path": str(kwargs["bundle_path"]),
            "top_differences": [],
        }

    monkeypatch.setattr(
        "weiss_rl.experiments.paired_outcome_contrastive.inspect_replay_bundle",
        fake_inspect_replay_bundle,
    )

    inspect_paired_outcome_sources(
        PairedOutcomeInspectionConfig(
            sources=(
                PairedOutcomeInspectionSource(
                    source_label="fixed_B2",
                    source_role="fixed_preserve",
                    source_dataset_path=dataset_path,
                    source_opponent_policy_id="B2 HeuristicPublic",
                    output_dir=tmp_path / "out",
                ),
            ),
            stack_config=tmp_path / "stack.yaml",
            run_dir=tmp_path / "run",
            snapshot_registry_json=registry_path,
            policy_a="policy_a",
            policy_b="policy_b",
        )
    )

    assert seen["snapshot_registry_path"] == registry_path


def test_apply_policy_b_top_action_overrides_rejects_zero_signal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="produced no train rows"):
        apply_policy_b_top_action_overrides(
            _contrastive_source_dataset(tmp_path),
            override_rows=[
                {
                    "bundle_name": "replay_a_pair000_swap0.zip",
                    "step_index": 0,
                    "teacher_action": 1,
                }
            ],
            source_label="fixed_B2",
            source_role="fixed_preserve",
            source_dataset_path=tmp_path / "source.npz",
        )


def _contrastive_source_dataset(tmp_path: Path) -> ReplayTrajectoryDataset:
    legal_rows = (
        (1, 2),
        (1, 3),
        (4, 5),
        (4, 6),
    )
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    offsets = [0]
    cursor = 0
    for row in legal_rows:
        ids = np.asarray(row, dtype=np.uint32)
        legal_ids_parts.append(ids)
        legal_meta_parts.append(np.zeros((ids.shape[0], 4), dtype=np.uint16))
        cursor += int(ids.shape[0])
        offsets.append(cursor)
    return ReplayTrajectoryDataset(
        obs=np.zeros((2, 2, 4), dtype=np.float32),
        actor=np.zeros((2, 2), dtype=np.int8),
        to_play_seat=np.zeros((2, 2), dtype=np.int8),
        actions=np.asarray([[1, 1], [4, 4]], dtype=np.int64),
        legal_ids=np.concatenate(legal_ids_parts).astype(np.uint32),
        legal_offsets=np.asarray(offsets, dtype=np.uint32),
        legal_action_meta=np.concatenate(legal_meta_parts, axis=0).astype(np.uint16),
        teacher_family=np.full((2, 2), -1, dtype=np.int32),
        teacher_slot=np.full((2, 2), -1, dtype=np.int32),
        teacher_move_source=np.full((2, 2), -1, dtype=np.int32),
        teacher_attack_type=np.full((2, 2), -1, dtype=np.int32),
        teacher_action=np.full((2, 2), -1, dtype=np.int32),
        teacher_valid=np.zeros((2, 2), dtype=np.bool_),
        policy_train_mask=np.asarray([[True, False], [True, False]], dtype=np.bool_),
        reset_before_step=np.zeros((2, 2), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": 2,
            "episode_count": 2,
            "time_steps": 2,
            "row_count": 4,
            "train_rows": 2,
            "pass_action_id": 51,
            "spec_hash256": "abc",
            "selected_bundles": [
                {
                    "bundle_path": (tmp_path / "replay_a_pair000_swap0.zip").as_posix(),
                    "bundle_name": "replay_a_pair000_swap0.zip",
                    "pair_index": 0,
                    "swap_index": 0,
                    "outcome": "W",
                },
                {
                    "bundle_path": (tmp_path / "replay_b_pair001_swap0.zip").as_posix(),
                    "bundle_name": "replay_b_pair001_swap0.zip",
                    "pair_index": 1,
                    "swap_index": 0,
                    "outcome": "W",
                },
            ],
        },
    )
