from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.experiments.paired_outcome_preference_dataset import (
    PairedOutcomePreferenceDatasetConfig,
    build_paired_outcome_preference_dataset,
)
from weiss_rl.experiments.paired_outcome_preference_dataset_cli import parse_opponent_match_aliases
from weiss_rl.experiments.paired_outcome_preference_dataset_reporting import (
    paired_outcome_preference_dataset_output_line,
    paired_outcome_preference_dataset_output_payload,
)
from weiss_rl.experiments.paired_outcome_preference_dataset_runtime import (
    paired_outcome_preference_dataset_config_from_args,
)
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    replay_trajectory_bc_batch,
    save_replay_trajectory_bc_dataset,
)
from weiss_rl.training.paired_outcome_preference_replay import paired_outcome_preference_complete_pair_count


def test_paired_outcome_preference_dataset_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_preference_dataset,
        paired_outcome_preference_dataset_cli,
        paired_outcome_preference_dataset_entrypoint,
        paired_outcome_preference_dataset_runtime,
    )

    assert paired_outcome_preference_dataset_entrypoint._build_parser is (
        paired_outcome_preference_dataset_cli.build_paired_outcome_preference_dataset_parser
    )
    assert paired_outcome_preference_dataset_entrypoint._parse_opponent_match_aliases is (
        paired_outcome_preference_dataset_cli.parse_opponent_match_aliases
    )
    assert paired_outcome_preference_dataset_entrypoint.run_paired_outcome_preference_dataset is (
        paired_outcome_preference_dataset_runtime.run_paired_outcome_preference_dataset
    )
    assert paired_outcome_preference_dataset_entrypoint.PairedOutcomePreferenceDatasetConfig is (
        paired_outcome_preference_dataset.PairedOutcomePreferenceDatasetConfig
    )
    assert paired_outcome_preference_dataset_entrypoint.build_paired_outcome_preference_dataset is (
        paired_outcome_preference_dataset.build_paired_outcome_preference_dataset
    )


def test_paired_outcome_preference_dataset_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_preference_dataset_cli import (
        build_paired_outcome_preference_dataset_parser,
    )

    args = build_paired_outcome_preference_dataset_parser().parse_args(
        [
            "--preferred-dataset",
            str(tmp_path / "preferred.npz"),
            "--rejected-dataset",
            str(tmp_path / "rejected.npz"),
            "--output",
            str(tmp_path / "preference.npz"),
        ]
    )

    assert args.preferred_dataset == tmp_path / "preferred.npz"
    assert args.rejected_dataset == tmp_path / "rejected.npz"
    assert args.output == tmp_path / "preference.npz"
    assert args.summary_json is None
    assert args.max_pairs is None
    assert args.preferred_label == "preferred"
    assert args.rejected_label == "rejected"
    assert args.opponent_match_alias == []


def test_paired_outcome_preference_dataset_alias_parser_preserves_validation() -> None:
    assert parse_opponent_match_aliases([" source = target ", "a=b"]) == {"source": "target", "a": "b"}
    with pytest.raises(SystemExit, match="must be FROM=TO"):
        parse_opponent_match_aliases(["missing_separator"])
    with pytest.raises(SystemExit, match="non-empty FROM and TO"):
        parse_opponent_match_aliases(["source= "])


def test_paired_outcome_preference_dataset_runtime_maps_args(tmp_path: Path) -> None:
    args = SimpleNamespace(
        preferred_dataset=tmp_path / "preferred.npz",
        rejected_dataset=tmp_path / "rejected.npz",
        output=tmp_path / "preference.npz",
        summary_json=tmp_path / "summary.json",
        max_pairs=7,
        preferred_label="chosen",
        rejected_label="not_chosen",
        opponent_match_alias=["wrapped=plain"],
    )

    config = paired_outcome_preference_dataset_config_from_args(args)

    assert config.preferred_dataset == (tmp_path / "preferred.npz").resolve()
    assert config.rejected_dataset == (tmp_path / "rejected.npz").resolve()
    assert config.output_dataset == tmp_path / "preference.npz"
    assert config.output_summary_json == tmp_path / "summary.json"
    assert config.max_pairs == 7
    assert config.preferred_label == "chosen"
    assert config.rejected_label == "not_chosen"
    assert config.opponent_match_aliases == {"wrapped": "plain"}


def test_paired_outcome_preference_dataset_reporting_preserves_compact_console_json(tmp_path: Path) -> None:
    dataset = _dataset([{"source_opponent_policy_id": "policy_000001", "source_pair_index": 8}])
    summary = {"pair_count": 3}

    assert paired_outcome_preference_dataset_output_payload(
        output_dataset=tmp_path / "preference.npz",
        dataset=dataset,
        summary=summary,
    ) == {
        "output": (tmp_path / "preference.npz").as_posix(),
        "pair_count": 3,
        "episodes": 1,
        "train_rows": 1,
    }
    assert paired_outcome_preference_dataset_output_line(
        output_dataset=tmp_path / "preference.npz",
        dataset=dataset,
        summary=summary,
    ) == (
        f'{{"episodes": 1, "output": "{(tmp_path / "preference.npz").as_posix()}", "pair_count": 3, "train_rows": 1}}'
    )


def test_build_paired_outcome_preference_dataset_matches_pair_metadata(tmp_path: Path) -> None:
    preferred_path = tmp_path / "preferred.npz"
    rejected_path = tmp_path / "rejected.npz"
    output_path = tmp_path / "preference.npz"
    summary_path = tmp_path / "summary.json"
    save_replay_trajectory_bc_dataset(
        preferred_path,
        _dataset(
            [
                {"source_opponent_policy_id": "policy_000001", "source_pair_index": 8, "episode_seed": 100},
                {"source_opponent_policy_id": "policy_000002", "source_pair_index": 9, "episode_seed": 101},
            ]
        ),
    )
    save_replay_trajectory_bc_dataset(
        rejected_path,
        _dataset(
            [
                {"source_opponent_policy_id": "policy_000001", "source_pair_index": 8, "episode_seed": 100},
                {"source_opponent_policy_id": "policy_000099", "source_pair_index": 9, "episode_seed": 101},
            ]
        ),
    )

    dataset, summary = build_paired_outcome_preference_dataset(
        PairedOutcomePreferenceDatasetConfig(
            preferred_dataset=preferred_path,
            rejected_dataset=rejected_path,
            output_dataset=output_path,
            output_summary_json=summary_path,
        )
    )

    assert output_path.is_file()
    assert summary_path.is_file()
    assert summary["pair_count"] == 1
    assert dataset.episode_count == 2
    assert paired_outcome_preference_complete_pair_count(dataset) == 1
    batch = replay_trajectory_bc_batch(dataset, episode_indices=[0, 1])
    assert batch["preference_pair_id"].tolist() == [[0, 0]]
    assert batch["preference_role"].tolist() == [[1, 0]]


def test_build_paired_outcome_preference_dataset_matches_opponent_aliases(tmp_path: Path) -> None:
    preferred_path = tmp_path / "preferred.npz"
    rejected_path = tmp_path / "rejected.npz"
    output_path = tmp_path / "preference.npz"
    preferred_id = "seed_c3aac2f9dc_policy_000004"
    rejected_id = "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004"
    save_replay_trajectory_bc_dataset(
        preferred_path,
        _dataset(
            [
                {"source_opponent_policy_id": preferred_id, "source_pair_index": 205, "episode_seed": 999},
            ]
        ),
    )
    save_replay_trajectory_bc_dataset(
        rejected_path,
        _dataset(
            [
                {"source_opponent_policy_id": rejected_id, "source_pair_index": 205, "episode_seed": 999},
            ]
        ),
    )

    dataset, summary = build_paired_outcome_preference_dataset(
        PairedOutcomePreferenceDatasetConfig(
            preferred_dataset=preferred_path,
            rejected_dataset=rejected_path,
            output_dataset=output_path,
            opponent_match_aliases={rejected_id: preferred_id},
        )
    )

    assert summary["pair_count"] == 1
    assert summary["opponent_match_aliases"] == {rejected_id: preferred_id}
    bundles = dataset.metadata["selected_bundles"]
    assert bundles[0]["source_opponent_policy_id"] == preferred_id
    assert bundles[1]["source_opponent_policy_id"] == rejected_id
    assert bundles[0]["preference_match_opponent_policy_id"] == preferred_id
    assert bundles[1]["preference_match_opponent_policy_id"] == preferred_id
    assert paired_outcome_preference_complete_pair_count(dataset) == 1


def _dataset(selected_bundles: list[dict]) -> ReplayTrajectoryDataset:
    episode_count = len(selected_bundles)
    obs = np.zeros((1, episode_count, 4), dtype=np.float32)
    actions = np.ones((1, episode_count), dtype=np.int64)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    legal_meta = np.stack([legal_ids, legal_ids + 10, legal_ids + 20], axis=1).astype(np.uint16)
    offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    bundles = []
    for index, bundle in enumerate(selected_bundles):
        bundles.append({"source_dataset_label": f"episode_{index}", **bundle})
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=np.zeros((1, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((1, episode_count), dtype=np.int8),
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=offsets,
        legal_action_meta=legal_meta,
        teacher_family=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_slot=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_move_source=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_action=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_valid=np.zeros((1, episode_count), dtype=np.bool_),
        policy_train_mask=np.ones((1, episode_count), dtype=np.bool_),
        reset_before_step=np.zeros((1, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": episode_count,
            "requested_bundle_count": episode_count,
            "include_outcomes": ["ALL"],
            "pass_action_id": 0,
            "spec_hash256": "ab" * 32,
            "train_rows": episode_count,
            "selected_bundles": bundles,
        },
    )
