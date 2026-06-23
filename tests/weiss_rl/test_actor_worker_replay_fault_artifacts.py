from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.replay.bundles import load_replay_bundle

from .actor_worker_test_support import (
    ACTION_SPACE,
    EngineFaultIdsEnv,
    ReplayIdsAutoResetEnv,
    ReplayIdsMissingIdentityAutoResetEnv,
    StaticMaskEnv,
    _policy_logits,
    _uniform_policy_logits,
)


def test_actor_worker_clears_replay_buffer_on_episode_boundary(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    worker = ActorWorker(
        actor_id=12,
        unroll_length=2,
        num_envs=1,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=101,
        replay_dir=replay_dir,
        run_id256=b"r" * 32,
        spec_hash256=bytes.fromhex("ab" * 32),
    )

    worker.run_once(env=ReplayIdsAutoResetEnv(), policy_logits_fn=_uniform_policy_logits)
    worker._flush_replay_for_env(env_index=0)

    [bundle_path] = sorted(replay_dir.glob("replay_*.zip"))
    meta, steps, fault = load_replay_bundle(bundle_path)

    assert fault is None
    assert meta.episode_identity_source == "simulator"
    assert meta.simulator_episode_key_u64 == 222
    assert meta.episode_seed64 == 22
    assert [step.decision_id for step in steps] == [5]
    assert [step.t for step in steps] == [1]


def test_actor_worker_keeps_derived_replay_identity_distinct_without_simulator_episode_ids(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    worker = ActorWorker(
        actor_id=14,
        unroll_length=2,
        num_envs=1,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=303,
        replay_dir=replay_dir,
        run_id256=b"s" * 32,
        spec_hash256=bytes.fromhex("ef" * 32),
        capture_replays_on_done=True,
    )

    worker.run_once(env=ReplayIdsMissingIdentityAutoResetEnv(), policy_logits_fn=_uniform_policy_logits)

    bundle_paths = sorted(replay_dir.glob("replay_*.zip"))
    assert len(bundle_paths) == 2
    assert bundle_paths[0].name != bundle_paths[1].name

    bundle_payloads = [load_replay_bundle(path) for path in bundle_paths]
    metas = sorted((meta for meta, _, _ in bundle_payloads), key=lambda meta: meta.episode_index)
    steps_by_episode = {meta.episode_index: steps for meta, steps, _ in bundle_payloads}
    faults = [fault for _, _, fault in bundle_payloads]

    assert faults == [None, None]
    assert [meta.episode_index for meta in metas] == [0, 1]
    assert all(meta.episode_identity_source == "derived" for meta in metas)
    assert all(meta.simulator_episode_key_kind is None for meta in metas)
    assert all(meta.simulator_episode_key_u64 is None for meta in metas)
    assert metas[0].episode_seed64 != metas[1].episode_seed64
    assert [step.decision_id for step in steps_by_episode[0]] == [0]
    assert [step.decision_id for step in steps_by_episode[1]] == [5]


def test_actor_worker_captures_engine_error_replay_with_actual_episode_identity(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    worker = ActorWorker(
        actor_id=13,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=202,
        replay_dir=replay_dir,
        run_id256=b"r" * 32,
        spec_hash256=bytes.fromhex("cd" * 32),
        capture_replays_on_done=False,
    )

    worker.run_once(env=EngineFaultIdsEnv(), policy_logits_fn=_uniform_policy_logits)

    [bundle_path] = sorted(replay_dir.glob("replay_*.zip"))
    meta, steps, fault = load_replay_bundle(bundle_path)

    assert meta.episode_identity_source == "simulator"
    assert meta.simulator_episode_key_u64 == 777
    assert meta.episode_seed64 == 77
    assert len(steps) == 1
    assert steps[0].engine_status == 17
    assert fault is not None
    assert fault["engine_status"] == 17
    assert fault["simulator_episode_key"] == 777


def test_actor_worker_writes_fault_bundle_on_nonfinite_logits(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    worker = ActorWorker(
        actor_id=8,
        unroll_length=2,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=41,
        fault_dir=fault_dir,
    )

    def nan_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
        logits = _policy_logits(obs, to_play)
        logits[0, 0] = np.nan
        return logits

    with pytest.raises(RuntimeError, match="non-finite actor policy logits; wrote fault bundle to ") as excinfo:
        worker.run_once(env=StaticMaskEnv(2), policy_logits_fn=nan_policy_logits)

    [fault_path] = sorted(fault_dir.glob("actor_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "actor_worker"
    assert payload["reason"] == "non-finite actor policy logits"
    assert payload["step"] == 0
    assert payload["logits_nonfinite_indices"]["data"] == [[0, 0]]
