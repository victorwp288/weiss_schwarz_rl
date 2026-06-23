from __future__ import annotations

from pathlib import Path

import numpy as np
from weiss_rl.replay.trajectory_bc import (
    build_replay_trajectory_bc_dataset,
    load_replay_trajectory_bc_dataset,
    replay_trajectory_bc_batch,
    save_replay_trajectory_bc_dataset,
)

from ._config_paths import canonical_stack_config_path
from .replay_trajectory_bc_workflow_test_support import (
    FakeReplayEnv,
    ids_batch,
    replay_contract,
    replay_step,
    return_env,
    write_episode_manifest,
    write_named_bundle,
    write_source_run,
)


def test_replay_trajectory_bc_dataset_extracts_focal_supported_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    write_source_run(run_dir)
    contract = replay_contract()
    named_bundle = write_named_bundle(
        tmp_path,
        contract=contract,
        steps=[
            replay_step(
                t=0,
                decision_id=10,
                actor=0,
                action=1,
                reward=0.25,
                terminated=False,
                legal_ids=np.asarray([0, 1], dtype=np.uint32),
            ),
            replay_step(
                t=1,
                decision_id=11,
                actor=1,
                action=2,
                reward=1.0,
                terminated=True,
                legal_ids=np.asarray([0, 2], dtype=np.uint32),
            ),
        ],
    )
    episodes_jsonl = tmp_path / "episodes.jsonl"
    write_episode_manifest(episodes_jsonl, outcome="W")
    env = FakeReplayEnv(
        ids_batch(
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
                ids_batch(
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
                ids_batch(
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
        env_factory=lambda observed_contract: return_env(observed_contract, contract, env),
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
