from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from weiss_rl.replay.trajectory_bc import (
    build_replay_trajectory_bc_dataset,
    load_teacher_action_overrides_jsonl,
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


def test_replay_trajectory_bc_dataset_uses_teacher_action_overrides(tmp_path: Path) -> None:
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
                reward=1.0,
                terminated=True,
                legal_ids=np.asarray([0, 1, 2], dtype=np.uint32),
            )
        ],
    )
    episodes_jsonl = tmp_path / "episodes.jsonl"
    write_episode_manifest(episodes_jsonl, outcome="L")
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
    env = FakeReplayEnv(
        ids_batch(
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
                ids_batch(
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
        env_factory=lambda observed_contract: return_env(observed_contract, contract, env),
    )

    assert dataset.metadata["train_rows"] == 1
    assert dataset.metadata["teacher_action_override_rows"] == 1
    assert dataset.actions.tolist() == [[1]]
    assert dataset.teacher_valid.tolist() == [[True]]
    assert dataset.teacher_action.tolist() == [[2]]
    assert dataset.policy_train_mask.tolist() == [[True]]
