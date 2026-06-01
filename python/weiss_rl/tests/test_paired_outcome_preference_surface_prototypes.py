from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from weiss_rl.experiments.paired_outcome_preference_surface_prototypes import (
    PairedOutcomePreferenceSurfacePrototypeConfig,
    build_paired_outcome_preference_surface_prototype_report,
)
from weiss_rl.experiments.paired_outcome_preference_surface_prototypes_reporting import (
    paired_outcome_preference_surface_prototype_output_line,
    paired_outcome_preference_surface_prototype_output_payload,
)
from weiss_rl.experiments.paired_outcome_preference_surface_prototypes_runtime import (
    paired_outcome_preference_surface_prototype_config_from_args,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset


def test_surface_prototype_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_preference_surface_prototypes,
        paired_outcome_preference_surface_prototypes_cli,
        paired_outcome_preference_surface_prototypes_entrypoint,
        paired_outcome_preference_surface_prototypes_runtime,
    )

    assert paired_outcome_preference_surface_prototypes_entrypoint._build_parser is (
        paired_outcome_preference_surface_prototypes_cli.build_paired_outcome_preference_surface_prototype_parser
    )
    assert paired_outcome_preference_surface_prototypes_entrypoint.run_paired_outcome_preference_surface_prototype is (
        paired_outcome_preference_surface_prototypes_runtime.run_paired_outcome_preference_surface_prototype
    )
    assert paired_outcome_preference_surface_prototypes_entrypoint.PairedOutcomePreferenceSurfacePrototypeConfig is (
        paired_outcome_preference_surface_prototypes.PairedOutcomePreferenceSurfacePrototypeConfig
    )
    assert (
        paired_outcome_preference_surface_prototypes_entrypoint.build_paired_outcome_preference_surface_prototype_report
        is paired_outcome_preference_surface_prototypes.build_paired_outcome_preference_surface_prototype_report
    )
    assert (
        paired_outcome_preference_surface_prototypes_entrypoint.write_paired_outcome_preference_surface_prototype_report
        is paired_outcome_preference_surface_prototypes.write_paired_outcome_preference_surface_prototype_report
    )


def test_surface_prototype_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_preference_surface_prototypes_cli import (
        build_paired_outcome_preference_surface_prototype_parser,
    )

    args = build_paired_outcome_preference_surface_prototype_parser().parse_args(
        [
            "--prototype-dataset",
            str(tmp_path / "prototype.npz"),
            "--probe-dataset",
            str(tmp_path / "probe.npz"),
            "--output-json",
            str(tmp_path / "surface_prototypes.json"),
        ]
    )

    assert args.prototype_dataset == tmp_path / "prototype.npz"
    assert args.probe_dataset == [tmp_path / "probe.npz"]
    assert args.probe_label == []
    assert args.key_mode == "current_history_opponent"
    assert args.opponent_key_mode == "raw_policy_id"
    assert args.stack_config is None
    assert args.opponent_context_policy_id == []
    assert args.max_examples == 20
    assert args.output_json == tmp_path / "surface_prototypes.json"


def test_surface_prototype_runtime_maps_args_without_resolving_paths(tmp_path: Path) -> None:
    args = SimpleNamespace(
        prototype_dataset=tmp_path / "prototype.npz",
        probe_dataset=[tmp_path / "probe_a.npz", tmp_path / "probe_b.npz"],
        probe_label=["a", "b"],
        stack_config=tmp_path / "stack.yaml",
        opponent_context_policy_id=["B2 HeuristicPublic", "policy_000004"],
        key_mode="current_history",
        opponent_key_mode="context_index",
        max_examples=9,
    )

    config = paired_outcome_preference_surface_prototype_config_from_args(args)

    assert config == PairedOutcomePreferenceSurfacePrototypeConfig(
        prototype_dataset_path=tmp_path / "prototype.npz",
        probe_dataset_paths=(tmp_path / "probe_a.npz", tmp_path / "probe_b.npz"),
        probe_labels=("a", "b"),
        stack_config_path=tmp_path / "stack.yaml",
        opponent_context_policy_ids=("B2 HeuristicPublic", "policy_000004"),
        key_mode="current_history",
        opponent_key_mode="context_index",
        max_examples=9,
    )


def test_surface_prototype_reporting_preserves_compact_console_json(tmp_path: Path) -> None:
    report = {
        "key_mode": "current_history_opponent",
        "opponent_key_mode": "raw_policy_id",
        "prototype": {
            "train_rows": 4,
            "unique_key_count": 3,
            "conflicting_key_count": 1,
        },
        "probes": [
            {
                "label": "probe",
                "train_rows": 5,
                "matched_train_rows": 2,
                "matched_rate": 0.4,
                "unexpected_matched_rows": 1,
                "conflicting_matched_key_count": 1,
            }
        ],
    }

    assert paired_outcome_preference_surface_prototype_output_payload(
        output_json=tmp_path / "surface_prototypes.json",
        report=report,
    ) == {
        "output_json": (tmp_path / "surface_prototypes.json").as_posix(),
        "key_mode": "current_history_opponent",
        "opponent_key_mode": "raw_policy_id",
        "prototype_train_rows": 4,
        "prototype_unique_key_count": 3,
        "prototype_conflicting_key_count": 1,
        "probes": [
            {
                "label": "probe",
                "train_rows": 5,
                "matched_train_rows": 2,
                "matched_rate": 0.4,
                "unexpected_matched_rows": 1,
                "conflicting_matched_key_count": 1,
            }
        ],
    }
    assert paired_outcome_preference_surface_prototype_output_line(
        output_json=tmp_path / "surface_prototypes.json",
        report=report,
    ) == (
        '{"key_mode": "current_history_opponent", "opponent_key_mode": "raw_policy_id", '
        f'"output_json": "{(tmp_path / "surface_prototypes.json").as_posix()}", '
        '"probes": [{"conflicting_matched_key_count": 1, "label": "probe", "matched_rate": 0.4, '
        '"matched_train_rows": 2, "train_rows": 5, "unexpected_matched_rows": 1}], '
        '"prototype_conflicting_key_count": 1, "prototype_train_rows": 4, "prototype_unique_key_count": 3}'
    )


def test_surface_prototype_opponent_key_prevents_cross_opponent_match(tmp_path):
    prototype_path = tmp_path / "prototype.npz"
    probe_path = tmp_path / "probe.npz"
    save_replay_trajectory_bc_dataset(
        prototype_path,
        _dataset(
            [
                _bundle("B2 HeuristicPublic", pair_index=205, action=124),
            ]
        ),
    )
    save_replay_trajectory_bc_dataset(
        probe_path,
        _dataset(
            [
                _bundle("policy_000004", pair_index=205, action=104),
            ]
        ),
    )

    report = build_paired_outcome_preference_surface_prototype_report(
        PairedOutcomePreferenceSurfacePrototypeConfig(
            prototype_dataset_path=prototype_path,
            probe_dataset_paths=(probe_path,),
            probe_labels=("cross_opponent",),
            key_mode="current_history_opponent",
        )
    )

    probe = report["probes"][0]
    assert probe["matched_train_rows"] == 0
    assert probe["unexpected_matched_rows"] == 0
    assert report["diagnostic_only_fields"]
    assert "source_pair_index" in report["diagnostic_only_fields"]


def test_surface_prototype_current_history_reports_cross_opponent_leakage(tmp_path):
    prototype_path = tmp_path / "prototype.npz"
    probe_path = tmp_path / "probe.npz"
    save_replay_trajectory_bc_dataset(
        prototype_path,
        _dataset([_bundle("B2 HeuristicPublic", pair_index=205, action=124)]),
    )
    save_replay_trajectory_bc_dataset(
        probe_path,
        _dataset([_bundle("policy_000004", pair_index=205, action=104)]),
    )

    report = build_paired_outcome_preference_surface_prototype_report(
        PairedOutcomePreferenceSurfacePrototypeConfig(
            prototype_dataset_path=prototype_path,
            probe_dataset_paths=(probe_path,),
            key_mode="current_history",
        )
    )

    probe = report["probes"][0]
    assert probe["matched_train_rows"] == 1
    assert probe["unexpected_matched_rows"] == 1
    assert probe["unexpected_examples"][0]["source_opponent_policy_id"] == "policy_000004"


def test_surface_prototype_does_not_use_pair_index_as_key(tmp_path):
    prototype_path = tmp_path / "prototype.npz"
    probe_path = tmp_path / "probe.npz"
    save_replay_trajectory_bc_dataset(
        prototype_path,
        _dataset([_bundle("B2 HeuristicPublic", pair_index=205, action=124)]),
    )
    save_replay_trajectory_bc_dataset(
        probe_path,
        _dataset([_bundle("B2 HeuristicPublic", pair_index=999, action=124)]),
    )

    report = build_paired_outcome_preference_surface_prototype_report(
        PairedOutcomePreferenceSurfacePrototypeConfig(
            prototype_dataset_path=prototype_path,
            probe_dataset_paths=(probe_path,),
            key_mode="current_history_opponent",
        )
    )

    probe = report["probes"][0]
    assert probe["matched_train_rows"] == 1
    assert probe["unexpected_matched_rows"] == 1
    assert probe["unexpected_examples"][0]["source_pair_index"] == 999


def test_surface_prototype_context_index_key_uses_model_suffix_mapping(tmp_path):
    prototype_path = tmp_path / "prototype.npz"
    probe_path = tmp_path / "probe.npz"
    save_replay_trajectory_bc_dataset(
        prototype_path,
        _dataset([_bundle("seed_b8c698d26a_seed_c3aac2f9dc_policy_000004", pair_index=205, action=104)]),
    )
    save_replay_trajectory_bc_dataset(
        probe_path,
        _dataset([_bundle("seed_c3aac2f9dc_policy_000004", pair_index=205, action=104)]),
    )

    report = build_paired_outcome_preference_surface_prototype_report(
        PairedOutcomePreferenceSurfacePrototypeConfig(
            prototype_dataset_path=prototype_path,
            probe_dataset_paths=(probe_path,),
            opponent_context_policy_ids=("seed_c3aac2f9dc_policy_000004",),
            key_mode="current_history_opponent",
            opponent_key_mode="context_index",
        )
    )

    probe = report["probes"][0]
    assert probe["matched_train_rows"] == 1
    assert probe["unexpected_matched_rows"] == 1
    assert report["prototype"]["summary"]["opponent_context_indices"] == [{"value": "1", "count": 1}]


def test_surface_prototype_legal_candidate_meta_is_part_of_key(tmp_path):
    prototype_path = tmp_path / "prototype.npz"
    probe_path = tmp_path / "probe.npz"
    save_replay_trajectory_bc_dataset(
        prototype_path,
        _dataset([_bundle("B2 HeuristicPublic", pair_index=205, action=124)], legal_meta_value=0),
    )
    save_replay_trajectory_bc_dataset(
        probe_path,
        _dataset([_bundle("B2 HeuristicPublic", pair_index=205, action=124)], legal_meta_value=7),
    )

    report = build_paired_outcome_preference_surface_prototype_report(
        PairedOutcomePreferenceSurfacePrototypeConfig(
            prototype_dataset_path=prototype_path,
            probe_dataset_paths=(probe_path,),
            key_mode="current_history_opponent",
        )
    )

    assert report["probes"][0]["matched_train_rows"] == 0


def test_surface_prototype_reports_ambiguous_prototype_keys(tmp_path):
    prototype_path = tmp_path / "prototype.npz"
    save_replay_trajectory_bc_dataset(
        prototype_path,
        _dataset(
            [
                _bundle("B2 HeuristicPublic", pair_index=205, action=124),
                _bundle("B2 HeuristicPublic", pair_index=205, action=104),
            ]
        ),
    )

    report = build_paired_outcome_preference_surface_prototype_report(
        PairedOutcomePreferenceSurfacePrototypeConfig(
            prototype_dataset_path=prototype_path,
            probe_dataset_paths=(prototype_path,),
            key_mode="current_history_opponent",
        )
    )

    assert report["prototype"]["conflicting_key_count"] == 1
    assert report["probes"][0]["conflicting_matched_key_count"] == 1
    assert report["prototype"]["ambiguous_keys"][0]["action_counts"] == [
        {"value": "124", "count": 1},
        {"value": "104", "count": 1},
    ]


def _bundle(opponent_id: str, *, pair_index: int, action: int) -> dict[str, object]:
    return {
        "source_opponent_policy_id": opponent_id,
        "source_pair_index": pair_index,
        "preference_pair_id": pair_index,
        "preference_role": 1,
        "preference_role_label": "repair",
        "source_dataset_label": "unit_surface",
        "action": action,
    }


def _dataset(bundles: list[dict[str, object]], *, legal_meta_value: int = 0) -> ReplayTrajectoryDataset:
    episode_count = len(bundles)
    actions = np.asarray([[int(bundle["action"]) for bundle in bundles]], dtype=np.int64)
    legal_ids = np.tile(np.asarray([104, 124], dtype=np.uint32), episode_count)
    legal_offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    legal_meta = np.full((legal_ids.shape[0], 4), legal_meta_value, dtype=np.uint16)
    selected_bundles = [{key: value for key, value in bundle.items() if key != "action"} for bundle in bundles]
    return ReplayTrajectoryDataset(
        obs=np.ones((1, episode_count, 3), dtype=np.float32),
        actor=np.zeros((1, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((1, episode_count), dtype=np.int8),
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        teacher_family=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_slot=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_move_source=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_attack_type=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_action=np.full((1, episode_count), -1, dtype=np.int32),
        teacher_valid=np.zeros((1, episode_count), dtype=np.bool_),
        policy_train_mask=np.ones((1, episode_count), dtype=np.bool_),
        reset_before_step=np.ones((1, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": episode_count,
            "requested_bundle_count": episode_count,
            "include_outcomes": ["ALL"],
            "pass_action_id": 0,
            "spec_hash256": "ab" * 32,
            "train_rows": episode_count,
            "selected_bundles": selected_bundles,
        },
    )
