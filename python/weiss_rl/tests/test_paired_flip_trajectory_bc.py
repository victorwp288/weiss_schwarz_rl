from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weiss_rl.experiments import paired_flip_trajectory_bc as paired_flip_module
from weiss_rl.experiments.paired_flip_trajectory_bc import (
    PairedFlipTrajectoryBcConfig,
    build_paired_flip_trajectory_bc_dataset,
    paired_flip_opponent_seed_plan,
    paired_flip_target_metadata_by_opponent_seed,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def test_paired_flip_opponent_seed_plan_groups_and_deduplicates_targets() -> None:
    report = {
        "targets": [
            {"opponent_policy_id": "B4 HeuristicPublicControl", "episode_seed": 7},
            {"opponent_policy_id": "B4 HeuristicPublicControl", "episode_seed": "7"},
            {"opponent_policy_id": "B4 HeuristicPublicControl", "episode_seed": 3},
            {"opponent_policy_id": "seed_hard_negative", "episode_seed": 11},
            {"opponent_policy_id": "", "episode_seed": 99},
        ]
    }

    assert paired_flip_opponent_seed_plan(report) == {
        "B4 HeuristicPublicControl": (3, 7),
        "seed_hard_negative": (11,),
    }


def test_paired_flip_opponent_seed_plan_rejects_bad_seed() -> None:
    with pytest.raises(ValueError, match="invalid episode_seed"):
        paired_flip_opponent_seed_plan({"targets": [{"opponent_policy_id": "B4", "episode_seed": "bad"}]})


def test_paired_flip_opponent_seed_plan_requires_targets_list() -> None:
    with pytest.raises(ValueError, match="missing targets list"):
        paired_flip_opponent_seed_plan({})


def test_paired_flip_target_metadata_preserves_source_pair_index_bucket() -> None:
    report = {
        "selection": {
            "flip_kind": "baseline_win_candidate_nonwin",
            "pair_index_min": 128,
            "pair_index_max": None,
        },
        "targets": [
            {
                "target_id": "target-a",
                "opponent_policy_id": "B2 HeuristicPublic",
                "episode_seed": "123",
                "pair_index": 144,
                "swap_index": 1,
                "tags": ["fixed"],
            },
            {
                "target_id": "target-b",
                "opponent_policy_id": "B2 HeuristicPublic",
                "episode_seed": 123,
                "pair_index": 160,
                "swap_index": 0,
                "tags": ["fixed"],
            },
        ],
    }

    metadata = paired_flip_target_metadata_by_opponent_seed(report)

    targets = metadata["B2 HeuristicPublic"][123]
    assert [target["pair_index"] for target in targets] == [144, 160]
    assert {target["pair_index_bucket"] for target in targets} == {"pair_index_gte_128"}
    assert {target["target_id"] for target in targets} == {"target-a", "target-b"}
    assert {target["flip_kind"] for target in targets} == {"baseline_win_candidate_nonwin"}


def test_paired_flip_target_metadata_rejects_missing_pair_index() -> None:
    with pytest.raises(ValueError, match="invalid pair/seed provenance"):
        paired_flip_target_metadata_by_opponent_seed(
            {"targets": [{"opponent_policy_id": "B2 HeuristicPublic", "episode_seed": 123}]}
        )


def test_paired_flip_build_persists_source_pair_metadata_to_source_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "targets.json"
    report_path.write_text(
        json.dumps(
            {
                "selection": {
                    "flip_kind": "baseline_win_candidate_nonwin",
                    "pair_index_min": 128,
                },
                "targets": [
                    {
                        "target_id": "target-b2-ext",
                        "opponent_policy_id": "B2 HeuristicPublic",
                        "episode_seed": 123,
                        "pair_index": 144,
                        "swap_index": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_build_champion_hardneg_trajectory_bc_dataset(**_kwargs: object) -> tuple[ReplayTrajectoryDataset, dict]:
        return _minimal_replay_dataset(episode_seed=123), {"kind": "fake_source"}

    saved: list[tuple[Path, list[dict]]] = []

    def fake_save_replay_trajectory_bc_dataset(path: Path, dataset: ReplayTrajectoryDataset) -> None:
        saved.append((Path(path), [dict(item) for item in dataset.metadata.get("selected_bundles", [])]))

    monkeypatch.setattr(
        paired_flip_module,
        "build_champion_hardneg_trajectory_bc_dataset",
        fake_build_champion_hardneg_trajectory_bc_dataset,
    )
    monkeypatch.setattr(paired_flip_module, "save_replay_trajectory_bc_dataset", fake_save_replay_trajectory_bc_dataset)

    build_paired_flip_trajectory_bc_dataset(
        PairedFlipTrajectoryBcConfig(
            stack=object(),  # type: ignore[arg-type]
            contract=object(),  # type: ignore[arg-type]
            stack_config=tmp_path / "stack.yaml",
            run_dir=tmp_path / "run",
            snapshot_registry_json=tmp_path / "registry.json",
            paired_flip_targets_json=report_path,
            focal_policy_id="main_interp_repair_a015",
            output_run_dir=tmp_path / "out",
            output_dataset=tmp_path / "out" / "merged.npz",
            source_label_prefix="rawext256_extfixed_preserve_",
        )
    )

    source_save = next(
        (bundles for path, bundles in saved if path.name == "paired_flip_bc_b2_heuristicpublic.npz"), None
    )
    assert source_save is not None
    assert source_save[0]["source_pair_index"] == 144
    assert source_save[0]["source_pair_index_bucket"] == "pair_index_gte_128"
    assert source_save[0]["source_target_ids"] == ["target-b2-ext"]


def _minimal_replay_dataset(*, episode_seed: int) -> ReplayTrajectoryDataset:
    return ReplayTrajectoryDataset(
        obs=np.zeros((1, 1, 2), dtype=np.float32),
        actor=np.zeros((1, 1), dtype=np.int8),
        to_play_seat=np.zeros((1, 1), dtype=np.int8),
        actions=np.zeros((1, 1), dtype=np.int64),
        legal_ids=np.zeros((1,), dtype=np.uint32),
        legal_offsets=np.array([0, 1], dtype=np.uint32),
        legal_action_meta=np.zeros((1, 4), dtype=np.uint16),
        teacher_family=np.full((1, 1), -1, dtype=np.int32),
        teacher_slot=np.full((1, 1), -1, dtype=np.int32),
        teacher_move_source=np.full((1, 1), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, 1), -1, dtype=np.int32),
        teacher_action=np.full((1, 1), -1, dtype=np.int32),
        teacher_valid=np.zeros((1, 1), dtype=np.bool_),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
        reset_before_step=np.ones((1, 1), dtype=np.bool_),
        metadata={
            "bundle_count": 1,
            "episode_count": 1,
            "train_rows": 1,
            "selected_bundles": [
                {
                    "bundle_path": "fake_pair000_swap0.zip",
                    "episode_seed": int(episode_seed),
                    "pair_index": 0,
                    "swap_index": 0,
                }
            ],
        },
    )
