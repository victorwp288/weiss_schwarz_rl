from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.config import StackConfig, compute_config_hash256, load_stack_config
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, PolicyValueModel
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    write_replay_bundle,
)
from weiss_rl.replay.inspector import inspect_replay_bundle

from ._config_paths import canonical_stack_config_path

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeReplayEnv:
    def __init__(
        self,
        initial_batch: DecisionBoundaryBatch,
        transitions: list[tuple[int, DecisionBoundaryBatch]],
    ) -> None:
        self._initial_batch = initial_batch
        self._transitions = list(transitions)
        self.actions: list[int] = []
        self.closed = False

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        return self._initial_batch

    def step(self, actions: np.ndarray) -> DecisionBoundaryBatch:
        action = int(np.asarray(actions, dtype=np.int64)[0])
        self.actions.append(action)
        expected_action, next_batch = self._transitions.pop(0)
        assert action == expected_action
        return next_batch

    def close(self) -> None:
        self.closed = True


def _write_policy_weights(
    *,
    run_dir: Path,
    stack: StackConfig,
    policy_id: str,
    observation_dim: int,
    logits: dict[int, float],
    observation_spec: dict[str, object] | None = None,
    config_hash256: str | None = None,
) -> Path:
    weights_dir = run_dir / "training" / "snapshots" / policy_id
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_config = stack.config.model
    assert model_config is not None
    model = PolicyValueModel(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=GLOBAL_ACTION_SPACE_SIZE,
        observation_spec=_typed_observation_spec(obs_len=observation_dim)
        if observation_spec is None
        else observation_spec,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for action_id, value in logits.items():
            cast(Any, model.policy_head).bias[action_id] = value
    weights_path = weights_dir / "weights.pt"
    torch.save(
        {
            "format": "minimal_train_snapshot_weights_v1",
            "policy_id": policy_id,
            "update": 1,
            "config_hash256": compute_config_hash256(stack) if config_hash256 is None else config_hash256,
            "model_state_dict": model.state_dict(),
        },
        weights_path,
    )
    return weights_path


def _typed_observation_spec(*, obs_len: int) -> dict[str, object]:
    return {
        "obs_encoding_version": 2,
        "dtype": "f32",
        "obs_len": obs_len,
        "self_first": True,
        "header_fields": [{"name": f"feature_{index}", "index": index} for index in range(obs_len)],
        "player_blocks": [],
        "tail_slices": [],
    }


def _heuristic_spec_bundle() -> dict[str, object]:
    return {
        "policy_version": 2,
        "spec_hash": 123,
        "observation": {
            "obs_encoding_version": 2,
            "obs_len": 512,
            "dtype": "i32",
            "self_first": True,
            "header_fields": [
                {"name": "active_player", "index": 0},
                {"name": "phase", "index": 1},
                {"name": "decision_kind", "index": 2},
                {"name": "decision_player", "index": 3},
                {"name": "terminal", "index": 4},
                {"name": "last_action_kind", "index": 5},
                {"name": "last_action_arg0", "index": 6},
                {"name": "last_action_arg1", "index": 7},
                {"name": "attack_slot", "index": 8},
                {"name": "defender_slot", "index": 9},
                {"name": "attack_type", "index": 10},
                {"name": "attack_damage", "index": 11},
                {"name": "attack_counter_power", "index": 12},
                {"name": "focus_slot", "index": 13},
                {"name": "choice_page_start", "index": 14},
                {"name": "choice_total", "index": 15},
            ],
            "player_blocks": [
                {
                    "player_index": 0,
                    "base": 16,
                    "len": 42,
                    "slices": [
                        {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                        {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                        {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                        {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                        {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                    ],
                },
                {
                    "player_index": 1,
                    "base": 58,
                    "len": 42,
                    "slices": [
                        {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                        {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                        {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                        {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                        {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                    ],
                },
            ],
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 527,
            "pass_action_id": 51,
            "attack_type_encoding": [["frontal", 0], ["side", 1], ["direct", 2]],
            "constants": [["MAX_HAND", 50], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
            "families": [
                {"name": "mulligan_confirm", "base": 0, "count": 1},
                {"name": "mulligan_select", "base": 1, "count": 50},
                {"name": "pass", "base": 51, "count": 1},
                {"name": "clock_from_hand", "base": 52, "count": 50},
                {"name": "main_play_character", "base": 102, "count": 250},
                {"name": "main_play_event", "base": 352, "count": 50},
                {"name": "main_move", "base": 402, "count": 20},
                {"name": "climax_play", "base": 422, "count": 50},
                {"name": "attack", "base": 472, "count": 9},
                {"name": "level_up", "base": 481, "count": 7},
                {"name": "encore_pay", "base": 488, "count": 5},
                {"name": "encore_decline", "base": 493, "count": 5},
                {"name": "trigger_order", "base": 498, "count": 10},
                {"name": "choice_select", "base": 508, "count": 16},
                {"name": "choice_prev_page", "base": 524, "count": 1},
                {"name": "choice_next_page", "base": 525, "count": 1},
                {"name": "concede", "base": 526, "count": 1},
            ],
        },
    }


def _heuristic_obs() -> np.ndarray:
    return np.zeros((512,), dtype=np.int32)


def _set_stage(
    obs: np.ndarray,
    *,
    player_index: int,
    slot: int,
    occupied: bool,
    attacked: bool = False,
    power: int = 0,
    effective_soul: int = 0,
    side_attack_allowed: bool = True,
) -> None:
    player_base = 16 if player_index == 0 else 58
    stage_base = player_base + 3 + slot * 7
    obs[stage_base] = 100 + slot if occupied else 0
    obs[stage_base + 2] = int(attacked)
    obs[stage_base + 3] = int(power)
    obs[stage_base + 5] = int(effective_soul)
    obs[stage_base + 6] = int(side_attack_allowed)


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


def _return_fake_env(
    observed_contract: ReplayRerunContract,
    expected_contract: ReplayRerunContract,
    env: FakeReplayEnv,
) -> FakeReplayEnv:
    assert observed_contract == expected_contract
    return env


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
    engine_status: int,
    legal_ids: np.ndarray,
    episode_seed: int,
    episode_key: int,
    legal_action_meta: np.ndarray | None = None,
    obs: np.ndarray | None = None,
) -> DecisionBoundaryBatch:
    ids = np.asarray(legal_ids, dtype=np.uint32)
    return DecisionBoundaryBatch(
        obs=np.asarray(np.zeros((4,), dtype=np.int16) if obs is None else obs).reshape(1, -1),
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
        legal_action_meta=None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16),
    )


def _inspect_with_heuristic_public_policy(
    tmp_path: Path,
    *,
    policy_b: str,
    policy_a_logits: dict[int, float],
    top_actions: int,
    obs: np.ndarray | None = None,
) -> dict[str, Any]:
    stack = load_stack_config(canonical_stack_config_path())
    case_dir = tmp_path / _policy_slug(policy_b)
    case_dir.mkdir(parents=True, exist_ok=True)
    run_dir = case_dir / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "spec_bundle.json").write_text(
        json.dumps(_heuristic_spec_bundle(), indent=2) + "\n",
        encoding="utf-8",
    )
    registry_path = run_dir / "training" / "snapshots" / "registry.json"

    policy_a_path = _write_policy_weights(
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_a",
        observation_dim=512,
        logits=policy_a_logits,
        observation_spec=_heuristic_spec_bundle()["observation"],  # type: ignore[arg-type]
    )
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_a",
        update=1,
        weights_sha256="sha-a",
        path=policy_a_path.relative_to(run_dir).as_posix(),
    )
    registry.save(registry_path)

    contract = ReplayRerunContract(version=2, observation_visibility="public", max_decisions=200, max_ticks=10_000)
    legal_ids = np.array([51, 472, 473, 474], dtype=np.uint16)
    bundle_path = _write_bundle(
        case_dir,
        contract=contract,
        steps=[
            ReplayStep(
                t=0,
                decision_id=10,
                actor=0,
                action=474,
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                legal_fingerprint64=_fingerprint(decision_id=10, legal_ids=legal_ids),
            ),
        ],
    )

    replay_obs = _heuristic_obs_with_stage() if obs is None else obs
    env = FakeReplayEnv(
        _ids_batch(
            decision_id=10,
            actor=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            engine_status=0,
            legal_ids=legal_ids,
            episode_seed=44,
            episode_key=555,
            obs=replay_obs,
        ),
        transitions=[
            (
                474,
                _ids_batch(
                    decision_id=10,
                    actor=0,
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    engine_status=0,
                    legal_ids=np.array([], dtype=np.uint16),
                    episode_seed=44,
                    episode_key=555,
                    obs=replay_obs,
                ),
            ),
        ],
    )

    return inspect_replay_bundle(
        bundle_path=bundle_path,
        stack=stack,
        run_dir=run_dir,
        snapshot_registry_path=registry_path,
        policy_a="policy_a",
        policy_b=policy_b,
        top_k=1,
        top_actions=top_actions,
        env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
    )


def _heuristic_obs_with_stage(*, include_counts: bool = False) -> np.ndarray:
    obs = _heuristic_obs()
    if include_counts:
        obs[16] = 1
        obs[17] = 6
        obs[18] = 7
        obs[58] = 0
        obs[59] = 4
        obs[60] = 6
    _set_stage(obs, player_index=0, slot=0, occupied=True, power=5000, effective_soul=1)
    return obs


def _policy_slug(policy_id: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in policy_id).strip("_")
