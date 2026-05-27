from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    write_replay_bundle,
)
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    build_replay_trajectory_bc_dataset,
    load_replay_trajectory_bc_dataset,
    load_teacher_action_overrides_jsonl,
    merge_replay_trajectory_bc_datasets,
    replay_trajectory_bc_batch,
    save_replay_trajectory_bc_dataset,
)
from weiss_rl.tests._config_paths import canonical_stack_config_path


class _FakeReplayEnv:
    def __init__(self, initial_batch: DecisionBoundaryBatch, transitions: list[tuple[int, DecisionBoundaryBatch]]):
        self._initial_batch = initial_batch
        self._transitions = list(transitions)
        self.closed = False

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        return self._initial_batch

    def step(self, actions: np.ndarray) -> DecisionBoundaryBatch:
        expected_action, next_batch = self._transitions.pop(0)
        assert int(np.asarray(actions, dtype=np.int64)[0]) == expected_action
        return next_batch

    def close(self) -> None:
        self.closed = True


def test_replay_trajectory_bc_dataset_extracts_focal_supported_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    run_dir.mkdir()
    (run_dir / "spec_bundle.json").write_text(json.dumps(_spec_bundle(), indent=2) + "\n", encoding="utf-8")
    contract = ReplayRerunContract(version=2, observation_visibility="public", max_decisions=50, max_ticks=1000)
    bundle = _write_bundle(
        tmp_path,
        contract=contract,
        steps=[
            ReplayStep(
                t=0,
                decision_id=10,
                actor=0,
                action=1,
                reward=0.25,
                terminated=False,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=10, legal_ids=np.asarray([0, 1], dtype=np.uint32)),
            ),
            ReplayStep(
                t=1,
                decision_id=11,
                actor=1,
                action=2,
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=11, legal_ids=np.asarray([0, 2], dtype=np.uint32)),
            ),
        ],
    )
    named_bundle = tmp_path / "replay_feedface_pair000_swap0.zip"
    shutil.copy2(bundle, named_bundle)
    episodes_jsonl = tmp_path / "episodes.jsonl"
    episodes_jsonl.write_text(
        json.dumps(
            {
                "pair_index": 0,
                "swap_index": 0,
                "focal_seat": 0,
                "outcome": "W",
                "episode_seed": 44,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    env = _FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            legal_ids=np.asarray([0, 1], dtype=np.uint32),
            obs=np.asarray([0, 3, -1, 0], dtype=np.float32),
        ),
        transitions=[
            (
                1,
                _ids_batch(
                    decision_id=11,
                    actor=1,
                    reward=0.25,
                    terminated=False,
                    truncated=False,
                    legal_ids=np.asarray([0, 2], dtype=np.uint32),
                    obs=np.asarray([1, 3, -1, 0], dtype=np.float32),
                ),
            ),
            (
                2,
                _ids_batch(
                    decision_id=11,
                    actor=1,
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    legal_ids=np.asarray([], dtype=np.uint32),
                    obs=np.asarray([1, 3, -1, 0], dtype=np.float32),
                ),
            ),
        ],
    )

    dataset = build_replay_trajectory_bc_dataset(
        bundle_paths=[named_bundle],
        run_dir=run_dir,
        stack=canonical_stack_config_path(),
        episodes_jsonl=episodes_jsonl,
        env_factory=lambda observed_contract: _return_env(observed_contract, contract, env),
    )

    assert dataset.metadata["train_rows"] == 1
    assert dataset.metadata["opponent_rows"] == 1
    assert dataset.obs.shape == (2, 1, 4)
    assert dataset.policy_train_mask.tolist() == [[True], [False]]
    assert dataset.teacher_valid.tolist() == [[True], [False]]
    assert dataset.teacher_action.tolist() == [[1], [-1]]
    assert dataset.legal_offsets.tolist() == [0, 2, 4]
    assert env.closed is True

    output_path = tmp_path / "dataset.npz"
    save_replay_trajectory_bc_dataset(output_path, dataset)
    loaded = load_replay_trajectory_bc_dataset(output_path)
    batch = replay_trajectory_bc_batch(
        loaded,
        episode_indices=[0],
        initial_hidden_state=np.zeros((1, 2, 8), dtype=np.float32),
        opponent_context_indices=[3],
    )
    assert batch["obs"].shape == (2, 1, 4)
    assert batch["legal_offsets"].tolist() == [0, 2, 4]
    assert batch["initial_hidden_state"].shape == (1, 2, 8)
    assert batch["opponent_context_index"].tolist() == [[3], [3]]


def test_replay_trajectory_bc_dataset_uses_teacher_action_overrides(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    run_dir.mkdir()
    (run_dir / "spec_bundle.json").write_text(json.dumps(_spec_bundle(), indent=2) + "\n", encoding="utf-8")
    contract = ReplayRerunContract(version=2, observation_visibility="public", max_decisions=50, max_ticks=1000)
    bundle = _write_bundle(
        tmp_path,
        contract=contract,
        steps=[
            ReplayStep(
                t=0,
                decision_id=10,
                actor=0,
                action=1,
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=10, legal_ids=np.asarray([0, 1, 2], dtype=np.uint32)),
            )
        ],
    )
    named_bundle = tmp_path / "replay_feedface_pair000_swap0.zip"
    shutil.copy2(bundle, named_bundle)
    episodes_jsonl = tmp_path / "episodes.jsonl"
    episodes_jsonl.write_text(
        json.dumps(
            {
                "pair_index": 0,
                "swap_index": 0,
                "focal_seat": 0,
                "outcome": "L",
                "episode_seed": 44,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    overrides_jsonl = tmp_path / "overrides.jsonl"
    overrides_jsonl.write_text(
        json.dumps(
            {
                "bundle_name": named_bundle.name,
                "step_index": 0,
                "teacher_action": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    env = _FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            legal_ids=np.asarray([0, 1, 2], dtype=np.uint32),
            obs=np.asarray([0, 3, -1, 0], dtype=np.float32),
        ),
        transitions=[
            (
                1,
                _ids_batch(
                    decision_id=10,
                    actor=0,
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    legal_ids=np.asarray([], dtype=np.uint32),
                    obs=np.asarray([0, 3, -1, 0], dtype=np.float32),
                ),
            )
        ],
    )

    dataset = build_replay_trajectory_bc_dataset(
        bundle_paths=[named_bundle],
        run_dir=run_dir,
        stack=canonical_stack_config_path(),
        episodes_jsonl=episodes_jsonl,
        include_outcomes=(),
        teacher_action_overrides=load_teacher_action_overrides_jsonl(overrides_jsonl),
        env_factory=lambda observed_contract: _return_env(observed_contract, contract, env),
    )

    assert dataset.metadata["train_rows"] == 1
    assert dataset.metadata["teacher_action_override_rows"] == 1
    assert dataset.actions.tolist() == [[1]]
    assert dataset.teacher_valid.tolist() == [[True]]
    assert dataset.teacher_action.tolist() == [[2]]
    assert dataset.policy_train_mask.tolist() == [[True]]


def test_replay_trajectory_bc_batch_broadcasts_preference_metadata() -> None:
    dataset = _synthetic_dataset(
        time_steps=2,
        episode_count=2,
        legal_rows=[
            [0, 1],
            [0, 1],
            [0],
            [0],
        ],
        train_mask=[[True, True], [False, False]],
        label="preference",
    )
    dataset.metadata["selected_bundles"] = [
        {"source_dataset_label": "preferred", "preference_pair_id": 42, "preference_role": 1},
        {"source_dataset_label": "rejected", "preference_pair_id": 42, "preference_role": 0},
    ]

    batch = replay_trajectory_bc_batch(dataset, episode_indices=[1, 0])

    assert batch["preference_pair_id"].tolist() == [[42, 42], [42, 42]]
    assert batch["preference_role"].tolist() == [[0, 1], [0, 1]]
    assert batch["source_label_id"].tolist() == [[1, 0], [1, 0]]


def test_merge_replay_trajectory_bc_datasets_offsets_preference_pair_ids() -> None:
    first = _synthetic_dataset(
        time_steps=1,
        episode_count=2,
        legal_rows=[
            [0, 1],
            [0, 1],
        ],
        train_mask=[[True, True]],
        label="first",
    )
    first.metadata["selected_bundles"] = [
        {"source_dataset_label": "first_preferred", "preference_pair_id": 0, "preference_role": 1},
        {"source_dataset_label": "first_rejected", "preference_pair_id": 0, "preference_role": 0},
    ]
    second = _synthetic_dataset(
        time_steps=1,
        episode_count=2,
        legal_rows=[
            [0, 1],
            [0, 1],
        ],
        train_mask=[[True, True]],
        label="second",
    )
    second.metadata["selected_bundles"] = [
        {"source_dataset_label": "second_preferred", "preference_pair_id": 0, "preference_role": 1},
        {"source_dataset_label": "second_rejected", "preference_pair_id": 0, "preference_role": 0},
    ]

    merged = merge_replay_trajectory_bc_datasets(
        [first, second],
        source_labels=("first", "second"),
        preserve_source_bundle_labels=True,
    )
    batch = replay_trajectory_bc_batch(merged, episode_indices=[0, 1, 2, 3])

    assert batch["preference_pair_id"].tolist() == [[0, 0, 1, 1]]
    assert [bundle["merge_source_preference_pair_id"] for bundle in merged.metadata["selected_bundles"]] == [
        0,
        0,
        0,
        0,
    ]


def test_merge_replay_trajectory_bc_datasets_pads_shorter_sources() -> None:
    first = _synthetic_dataset(
        time_steps=2,
        episode_count=1,
        legal_rows=[
            [0, 1],
            [0, 2],
        ],
        train_mask=[[True], [False]],
        label="b4_win",
    )
    second = _synthetic_dataset(
        time_steps=1,
        episode_count=2,
        legal_rows=[
            [0, 3],
            [0, 4],
        ],
        train_mask=[[True, True]],
        label="b2_win",
    )

    merged = merge_replay_trajectory_bc_datasets(
        [first, second],
        source_labels=["b4_win", "b2_win"],
    )

    assert merged.obs.shape == (2, 3, 4)
    assert merged.metadata["source_dataset_count"] == 2
    assert merged.metadata["bundle_count"] == 3
    assert merged.metadata["train_rows"] == 3
    assert merged.metadata["source_datasets"][1]["label"] == "b2_win"
    assert merged.policy_train_mask.tolist() == [[True, True, True], [False, False, False]]
    assert merged.actions[:, 1:].tolist() == [[1, 1], [0, 0]]
    assert merged.legal_offsets.shape == (7,)
    padded_row_1_start = int(merged.legal_offsets[4])
    padded_row_1_stop = int(merged.legal_offsets[5])
    assert merged.legal_ids[padded_row_1_start:padded_row_1_stop].tolist() == [0]
    padded_row_2_start = int(merged.legal_offsets[5])
    padded_row_2_stop = int(merged.legal_offsets[6])
    assert merged.legal_ids[padded_row_2_start:padded_row_2_stop].tolist() == [0]

    batch = replay_trajectory_bc_batch(merged, episode_indices=[0, 2])
    assert batch["obs"].shape == (2, 2, 4)
    assert batch["legal_offsets"].tolist() == [0, 2, 4, 6, 7]


def test_merge_replay_trajectory_bc_datasets_preserves_all_outcome_sentinel() -> None:
    all_outcomes = _synthetic_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 1]],
        train_mask=[[True]],
        label="all_outcomes",
        include_outcomes=[],
    )
    wins_only = _synthetic_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 2]],
        train_mask=[[True]],
        label="wins_only",
        include_outcomes=["W"],
    )

    merged = merge_replay_trajectory_bc_datasets([all_outcomes, wins_only])

    assert merged.metadata["include_outcomes"] == []


def test_merge_replay_trajectory_bc_datasets_can_preserve_nested_source_labels() -> None:
    repair_a = _synthetic_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 1]],
        train_mask=[[True]],
        label="repair_a",
    )
    repair_b = _synthetic_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 2]],
        train_mask=[[True]],
        label="repair_b",
    )
    premerged = merge_replay_trajectory_bc_datasets(
        [repair_a, repair_b],
        source_labels=["repair_a", "repair_b"],
    )
    loss_state = _synthetic_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 3]],
        train_mask=[[True]],
        label="b1_lossstate",
    )

    merged = merge_replay_trajectory_bc_datasets(
        [premerged, loss_state],
        source_labels=["winnerrepair_mix", "b1_lossstate"],
        preserve_source_bundle_labels=True,
    )

    labels = [bundle["source_dataset_label"] for bundle in merged.metadata["selected_bundles"]]
    assert labels == ["repair_a", "repair_b", "b1_lossstate"]
    assert merged.metadata["selected_bundles"][0]["merge_source_dataset_label"] == "winnerrepair_mix"
    assert "nested_source_datasets" in merged.metadata["source_datasets"][0]


def test_merge_replay_trajectory_bc_datasets_flattens_stale_merge_labels_by_default() -> None:
    premerged = _synthetic_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 1]],
        train_mask=[[True]],
        label="old_mix",
    )
    premerged.metadata["selected_bundles"][0]["source_dataset_label"] = "old_source"
    premerged.metadata["selected_bundles"][0]["merge_source_dataset_label"] = "old_nested_mix"
    fresh = _synthetic_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 2]],
        train_mask=[[True]],
        label="fresh",
    )

    merged = merge_replay_trajectory_bc_datasets(
        [premerged, fresh],
        source_labels=["fixed_protect", "learned_repair"],
    )

    bundles = merged.metadata["selected_bundles"]
    assert bundles[0]["source_dataset_label"] == "fixed_protect"
    assert bundles[1]["source_dataset_label"] == "learned_repair"
    assert "merge_source_dataset_label" not in bundles[0]


def _return_env(
    observed_contract: ReplayRerunContract,
    expected_contract: ReplayRerunContract,
    env: _FakeReplayEnv,
) -> _FakeReplayEnv:
    assert observed_contract == expected_contract
    return env


def _synthetic_dataset(
    *,
    time_steps: int,
    episode_count: int,
    legal_rows: list[list[int]],
    train_mask: list[list[bool]],
    label: str,
    include_outcomes: list[str] | None = None,
) -> ReplayTrajectoryDataset:
    obs = np.zeros((time_steps, episode_count, 4), dtype=np.float32)
    actions = np.ones((time_steps, episode_count), dtype=np.int64)
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    offsets = [0]
    cursor = 0
    for row_ids in legal_rows:
        ids = np.asarray(row_ids, dtype=np.uint32)
        legal_ids_parts.append(ids)
        legal_meta_parts.append(np.stack([ids, ids + 10, ids + 20], axis=1).astype(np.uint16))
        cursor += int(ids.shape[0])
        offsets.append(cursor)
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=np.zeros((time_steps, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((time_steps, episode_count), dtype=np.int8),
        actions=actions,
        legal_ids=np.concatenate(legal_ids_parts).astype(np.uint32),
        legal_offsets=np.asarray(offsets, dtype=np.uint32),
        legal_action_meta=np.concatenate(legal_meta_parts, axis=0).astype(np.uint16),
        teacher_family=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_slot=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_move_source=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_attack_type=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_action=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_valid=np.zeros((time_steps, episode_count), dtype=np.bool_),
        policy_train_mask=np.asarray(train_mask, dtype=np.bool_),
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": episode_count,
            "requested_bundle_count": episode_count,
            "include_outcomes": ["W"] if include_outcomes is None else include_outcomes,
            "pass_action_id": 0,
            "spec_hash256": "ab" * 32,
            "train_rows": int(np.count_nonzero(train_mask)),
            "selected_bundles": [{"source": label}],
        },
    )


def _spec_bundle() -> dict[str, object]:
    return {
        "policy_version": 2,
        "spec_hash": 123,
        "observation": {
            "obs_encoding_version": 2,
            "obs_len": 4,
            "dtype": "f32",
            "self_first": True,
            "header_fields": [
                {"name": "active_player", "index": 0},
                {"name": "decision_kind", "index": 1},
                {"name": "last_action_arg0", "index": 2},
            ],
            "player_blocks": [],
            "tail_slices": [],
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 10,
            "pass_action_id": 0,
            "attack_type_encoding": [["direct", 0]],
            "constants": [["MAX_HAND", 2], ["MAX_STAGE", 2], ["ATTACK_SLOT_COUNT", 1]],
            "families": [
                {"name": "pass", "base": 0, "count": 1},
                {"name": "clock_from_hand", "base": 1, "count": 2},
                {"name": "main_play_character", "base": 3, "count": 4},
                {"name": "main_move", "base": 7, "count": 2},
                {"name": "attack", "base": 9, "count": 1},
            ],
        },
    }


def _write_bundle(tmp_path: Path, *, contract: ReplayRerunContract, steps: list[ReplayStep]) -> Path:
    meta = make_replay_bundle_meta(
        simulator_episode_key=555,
        run_id256=b"r" * 32,
        spec_hash256=bytes.fromhex("ab" * 32),
        actor_id=1,
        env_id=2,
        episode_index=3,
        episode_seed64=44,
        rerun_contract=contract,
    )
    return write_replay_bundle(out_dir=tmp_path, meta=meta, steps=steps)


def _fingerprint(*, decision_id: int, legal_ids: np.ndarray) -> int:
    return compute_legal_fingerprint64(
        spec_hash256=bytes.fromhex("ab" * 32),
        decision_id=decision_id,
        legal_ids=legal_ids,
    )


def _ids_batch(
    *,
    decision_id: int,
    actor: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    legal_ids: np.ndarray,
    obs: np.ndarray,
) -> DecisionBoundaryBatch:
    ids = np.asarray(legal_ids, dtype=np.uint32)
    return DecisionBoundaryBatch(
        obs=np.asarray(obs, dtype=np.float32).reshape(1, -1),
        reward=np.asarray([reward], dtype=np.float32),
        terminated=np.asarray([terminated], dtype=np.bool_),
        truncated=np.asarray([truncated], dtype=np.bool_),
        to_play=np.asarray([actor], dtype=np.int32),
        actor=np.asarray([actor], dtype=np.int32),
        decision_kind=np.asarray([3], dtype=np.int32),
        decision_id=np.asarray([decision_id], dtype=np.int64),
        engine_status=np.asarray([0], dtype=np.uint8),
        decision_count=np.asarray([0], dtype=np.uint32),
        tick_count=np.asarray([0], dtype=np.uint32),
        episode_seed=np.asarray([44], dtype=np.uint64),
        episode_key=np.asarray([555], dtype=np.uint64),
        ids_offsets=(ids, np.asarray([0, int(ids.size)], dtype=np.uint32)),
    )
