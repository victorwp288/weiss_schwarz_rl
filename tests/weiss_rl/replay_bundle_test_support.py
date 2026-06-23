from __future__ import annotations

from pathlib import Path

import numpy as np
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.replay.bundles import ReplayRerunContract, ReplayStep, make_replay_bundle_meta, write_replay_bundle


class FakeReplayEnv:
    def __init__(
        self, initial_batch: DecisionBoundaryBatch, transitions: list[tuple[int, DecisionBoundaryBatch]]
    ) -> None:
        self._initial_batch = initial_batch
        self._transitions = list(transitions)
        self.closed = False
        self.reset_seed: int | None = None
        self.actions: list[int] = []

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        self.reset_seed = seed
        return self._initial_batch

    def step(self, actions: np.ndarray) -> DecisionBoundaryBatch:
        action = int(np.asarray(actions, dtype=np.int64)[0])
        self.actions.append(action)
        expected_action, next_batch = self._transitions.pop(0)
        assert action == expected_action
        return next_batch

    def close(self) -> None:
        self.closed = True


def rerun_contract(
    *,
    version: int = 2,
    reward_json: str | None = '{"objective":"terminal_pm1"}',
    curriculum_json: str | None = '{"version":"curriculum_v1"}',
    deck: str | None = "preset:main_deck_5hy_yotsuba_v1",
    opponent_deck: str | None = "preset:main_deck_5hy_yotsuba_v1",
) -> ReplayRerunContract:
    return ReplayRerunContract(
        version=version,
        observation_visibility="public",
        max_decisions=200,
        max_ticks=10_000,
        reward_json=reward_json,
        curriculum_json=curriculum_json,
        deck=deck,
        opponent_deck=opponent_deck,
    )


def write_test_bundle(
    tmp_path: Path,
    *,
    contract: ReplayRerunContract | None,
    steps: list[ReplayStep],
) -> Path:
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


def return_fake_env(
    observed_contract: ReplayRerunContract,
    expected_contract: ReplayRerunContract,
    env: FakeReplayEnv,
) -> FakeReplayEnv:
    assert observed_contract == expected_contract
    return env


def ids_batch(
    *,
    decision_id: int,
    actor: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    engine_status: int,
    legal_ids: np.ndarray,
    episode_seed: int,
    episode_key: int,
) -> DecisionBoundaryBatch:
    ids = np.asarray(legal_ids, dtype=np.uint32)
    return DecisionBoundaryBatch(
        obs=np.zeros((1, 4), dtype=np.int16),
        reward=np.array([reward], dtype=np.float32),
        terminated=np.array([terminated], dtype=np.bool_),
        truncated=np.array([truncated], dtype=np.bool_),
        to_play=np.array([actor], dtype=np.int32),
        actor=np.array([actor], dtype=np.int32),
        decision_id=np.array([decision_id], dtype=np.int64),
        engine_status=np.array([engine_status], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([episode_seed], dtype=np.uint64),
        episode_key=np.array([episode_key], dtype=np.uint64),
        ids_offsets=(ids, np.array([0, int(ids.size)], dtype=np.int32)),
    )
