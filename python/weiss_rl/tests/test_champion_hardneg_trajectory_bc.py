from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weiss_rl.experiments.champion_hardneg_trajectory_bc import (
    OpponentDatasetResult,
    build_generation_metadata,
    normalize_explicit_paired_seeds,
    normalize_include_outcomes,
    slug_policy_id,
    source_role_for_policy_id,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def _dummy_dataset(train_rows: int = 3) -> ReplayTrajectoryDataset:
    metadata = {"train_rows": train_rows}
    return ReplayTrajectoryDataset(
        obs=np.zeros((1, 1, 2), dtype=np.float32),
        actor=np.zeros((1, 1), dtype=np.int8),
        to_play_seat=np.zeros((1, 1), dtype=np.int8),
        actions=np.zeros((1, 1), dtype=np.int64),
        legal_ids=np.zeros((1,), dtype=np.uint32),
        legal_offsets=np.asarray([0, 1], dtype=np.uint32),
        legal_action_meta=np.zeros((1, 3), dtype=np.uint16),
        teacher_family=np.zeros((1, 1), dtype=np.int32),
        teacher_slot=np.zeros((1, 1), dtype=np.int32),
        teacher_move_source=np.zeros((1, 1), dtype=np.int32),
        teacher_attack_type=np.zeros((1, 1), dtype=np.int32),
        teacher_action=np.zeros((1, 1), dtype=np.int32),
        teacher_valid=np.ones((1, 1), dtype=np.bool_),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
        reset_before_step=np.zeros((1, 1), dtype=np.bool_),
        metadata=metadata,
    )


def test_normalize_include_outcomes_defaults_to_wins_and_all_disables_filter() -> None:
    assert normalize_include_outcomes(None) == ("W",)
    assert normalize_include_outcomes(["w", "D"]) == ("W", "D")
    assert normalize_include_outcomes(["ALL"]) == ()


def test_normalize_include_outcomes_rejects_unknown_tokens() -> None:
    with pytest.raises(ValueError, match="include outcomes"):
        normalize_include_outcomes(["W", "X"])


def test_normalize_explicit_paired_seeds_deduplicates_and_rejects_bad_values() -> None:
    assert normalize_explicit_paired_seeds(["3", 4, "3"]) == (3, 4)
    with pytest.raises(ValueError, match="non-negative"):
        normalize_explicit_paired_seeds(["-1"])
    with pytest.raises(ValueError, match="integer"):
        normalize_explicit_paired_seeds(["not-a-seed"])


def test_source_role_prefers_explicit_hard_negative_over_champion() -> None:
    assert (
        source_role_for_policy_id(
            "policy_hard",
            champion_ids=("policy_hard",),
            hard_negative_ids=("policy_hard",),
        )
        == "hard_negative"
    )
    assert (
        source_role_for_policy_id(
            "policy_champion",
            champion_ids=("policy_champion",),
            hard_negative_ids=(),
        )
        == "imported_champion"
    )
    assert source_role_for_policy_id("policy_other", champion_ids=(), hard_negative_ids=()) == "explicit_opponent"


def test_slug_policy_id_is_stable_for_snapshot_ids() -> None:
    assert slug_policy_id("seed_x/main league selected") == "seed_x_main_league_selected"


def test_build_generation_metadata_records_roles_and_training_rows(tmp_path: Path) -> None:
    result = OpponentDatasetResult(
        opponent_policy_id="policy_hard",
        source_role="hard_negative",
        dataset=_dummy_dataset(train_rows=7),
        dataset_path=tmp_path / "dataset.npz",
        episodes_jsonl=tmp_path / "episodes.jsonl",
        bundle_paths=(tmp_path / "bundle_pair000_swap0.zip",),
        games=8,
        wins=5,
        losses=3,
        draws=0,
        truncations=0,
    )

    metadata = build_generation_metadata(
        focal_policy_id="main",
        stack_config=Path("configs/thesis/example.yaml"),
        run_dir=Path("runs/source"),
        snapshot_registry_json=Path("runs/source/training/snapshots/registry.json"),
        b1_baseline_run_dir=None,
        paired_seeds=(1, 2, 3, 4),
        include_outcomes=("W",),
        champion_ids=("policy_champion",),
        hard_negative_ids=("policy_hard",),
        opponent_results=(result,),
    )

    assert metadata["kind"] == "champion_hardneg_trajectory_bc_dataset_v1"
    assert metadata["paired_seed_count"] == 4
    assert metadata["opponents"][0]["source_role"] == "hard_negative"
    assert metadata["opponents"][0]["train_rows"] == 7
    assert metadata["opponents"][0]["mean"] == pytest.approx(5 / 8)
