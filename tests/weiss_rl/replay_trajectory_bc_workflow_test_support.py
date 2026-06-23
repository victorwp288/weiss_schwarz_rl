"""Shared replay workflow fixtures for trajectory-BC extraction tests."""

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


class FakeReplayEnv:
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


def return_env(
    observed_contract: ReplayRerunContract,
    expected_contract: ReplayRerunContract,
    env: FakeReplayEnv,
) -> FakeReplayEnv:
    assert observed_contract == expected_contract
    return env


def write_source_run(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / "spec_bundle.json").write_text(json.dumps(spec_bundle(), indent=2) + "\n", encoding="utf-8")


def write_episode_manifest(
    path: Path,
    *,
    outcome: str,
    focal_seat: int = 0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "pair_index": 0,
                "swap_index": 0,
                "focal_seat": focal_seat,
                "outcome": outcome,
                "episode_seed": 44,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_named_bundle(tmp_path: Path, *, contract: ReplayRerunContract, steps: list[ReplayStep]) -> Path:
    bundle = write_bundle(tmp_path, contract=contract, steps=steps)
    named_bundle = tmp_path / "replay_feedface_pair000_swap0.zip"
    shutil.copy2(bundle, named_bundle)
    return named_bundle


def replay_contract() -> ReplayRerunContract:
    return ReplayRerunContract(version=2, observation_visibility="public", max_decisions=50, max_ticks=1000)


def write_bundle(tmp_path: Path, *, contract: ReplayRerunContract, steps: list[ReplayStep]) -> Path:
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


def replay_step(
    *,
    t: int,
    decision_id: int,
    actor: int,
    action: int,
    reward: float,
    terminated: bool,
    legal_ids: np.ndarray,
) -> ReplayStep:
    return ReplayStep(
        t=t,
        decision_id=decision_id,
        actor=actor,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=False,
        engine_status=0,
        legal_fingerprint64=fingerprint(decision_id=decision_id, legal_ids=legal_ids),
    )


def fingerprint(*, decision_id: int, legal_ids: np.ndarray) -> int:
    return compute_legal_fingerprint64(
        spec_hash256=bytes.fromhex("ab" * 32),
        decision_id=decision_id,
        legal_ids=legal_ids,
    )


def ids_batch(
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


def spec_bundle() -> dict[str, object]:
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
