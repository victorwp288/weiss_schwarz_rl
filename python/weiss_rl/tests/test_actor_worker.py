from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask, resolve_pass_action_id

OBS_LEN = 8
ACTION_SPACE = 52


@dataclass(slots=True)
class FakeBatch:
    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play: np.ndarray
    actor: np.ndarray
    decision_id: np.ndarray
    engine_status: np.ndarray
    ids_offsets: tuple[np.ndarray, np.ndarray] | None = None
    mask: np.ndarray | None = None
    episode_seed: np.ndarray | None = None
    episode_key: np.ndarray | None = None


class FakeDecisionBoundaryEnv:
    def __init__(self, num_envs: int, *, layout_name: str, seed: int = 0):
        self.num_envs = num_envs
        self.layout_name = layout_name
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self) -> FakeBatch:
        self.t = 0
        return self._make_batch()

    def step(self, actions: np.ndarray) -> FakeBatch:
        _ = actions
        self.t += 1
        return self._make_batch()

    def _make_batch(self) -> FakeBatch:
        obs = np.full((self.num_envs, OBS_LEN), self.t, dtype=np.int16)
        reward = np.full((self.num_envs,), self.t, dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=np.bool_)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        to_play = ((np.arange(self.num_envs) + self.t) % 2).astype(np.int8)
        decision_id = np.full((self.num_envs,), self.t, dtype=np.int32)
        engine_status = np.zeros((self.num_envs,), dtype=np.int32)
        episode_seed = np.full((self.num_envs,), 100 + self.t, dtype=np.uint64)
        episode_key = np.full((self.num_envs,), 200 + self.t, dtype=np.uint64)

        if self.layout_name == "i16_legal_ids":
            slices = []
            offsets = [0]
            for env_index in range(self.num_envs):
                if env_index == 0 and self.t % 2 == 0:
                    legal_ids = np.array([], dtype=np.int32)
                else:
                    count = 2 + (env_index % 3)
                    legal_ids = np.array(
                        sorted(self.rng.choice(ACTION_SPACE, size=count, replace=False)),
                        dtype=np.int32,
                    )
                slices.append(legal_ids)
                offsets.append(offsets[-1] + legal_ids.size)
            return FakeBatch(
                obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                to_play=to_play,
                actor=to_play,
                decision_id=decision_id,
                engine_status=engine_status,
                ids_offsets=(
                    np.concatenate(slices, axis=0) if offsets[-1] > 0 else np.array([], dtype=np.int32),
                    np.array(offsets, dtype=np.uint32),
                ),
                episode_seed=episode_seed,
                episode_key=episode_key,
            )

        mask = np.zeros((self.num_envs, ACTION_SPACE), dtype=np.uint8)
        for env_index in range(self.num_envs):
            if env_index == 0 and self.t % 2 == 0:
                continue
            legal_count = 2 + (env_index % 3)
            legal_ids = np.sort(self.rng.choice(ACTION_SPACE, size=legal_count, replace=False))
            mask[env_index, legal_ids] = 1

        return FakeBatch(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            to_play=to_play,
            actor=to_play,
            decision_id=decision_id,
            engine_status=engine_status,
            mask=mask,
            episode_seed=episode_seed,
            episode_key=episode_key,
        )


def _fake_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    num_envs = obs.shape[0]
    logits = np.zeros((num_envs, ACTION_SPACE), dtype=np.float32)
    base = obs[:, 0].astype(np.float32) + to_play.astype(np.float32)
    logits[:] = base[:, None] * 0.01 + np.arange(ACTION_SPACE, dtype=np.float32)[None, :] * 0.001
    return logits


def test_actor_worker_ids_offsets_produces_flattened_packed_legality() -> None:
    T = 4
    N = 3
    env = FakeDecisionBoundaryEnv(N, layout_name="i16_legal_ids", seed=123)
    worker = ActorWorker(
        actor_id=0,
        unroll_length=T,
        num_envs=N,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=999,
    )

    batch = worker.run_once(env=env, policy_logits_fn=_fake_policy_logits)

    assert batch.reward.shape == (T, N)
    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None
    assert batch.legal_offsets.shape == (T * N + 1,)
    assert int(batch.legal_offsets[0]) == 0
    assert np.all(batch.legal_offsets[1:] >= batch.legal_offsets[:-1])

    pass_action_id = resolve_pass_action_id()
    flat_actions = batch.action.reshape(T * N).astype(np.int64, copy=False)
    flat_logits = _fake_policy_logits(batch.obs.reshape(T * N, OBS_LEN), batch.to_play_seat.reshape(T * N))
    recomputed = masked_logp_from_legal_ids(
        flat_logits,
        batch.legal_ids,
        batch.legal_offsets,
        flat_actions,
        pass_action_id=pass_action_id,
    )

    assert np.allclose(recomputed.reshape(T, N), batch.behavior_logp, atol=0.0, rtol=0.0)


def test_actor_worker_mask_uses_decision_boundary_mask_contract() -> None:
    T = 4
    N = 3
    env = FakeDecisionBoundaryEnv(N, layout_name="mask", seed=123)
    worker = ActorWorker(
        actor_id=1,
        unroll_length=T,
        num_envs=N,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=111,
    )

    batch = worker.run_once(env=env, policy_logits_fn=_fake_policy_logits)

    assert batch.reward.shape == (T, N)
    assert batch.legal_mask is not None
    assert batch.legal_mask.shape == (T, N, ACTION_SPACE)
    assert batch.legal_ids is None
    assert batch.legal_offsets is None

    pass_action_id = resolve_pass_action_id()
    recomputed = masked_logp_from_mask(
        _fake_policy_logits(batch.obs.reshape(T * N, OBS_LEN), batch.to_play_seat.reshape(T * N)),
        batch.legal_mask.reshape(T * N, ACTION_SPACE),
        batch.action.reshape(T * N).astype(np.int64, copy=False),
        pass_action_id=pass_action_id,
    )

    assert np.allclose(recomputed.reshape(T, N), batch.behavior_logp, atol=0.0, rtol=0.0)


def test_actor_worker_reports_checkpoint_lag_in_checkpoint_update_units(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_100.pt").write_text("stub\n", encoding="utf-8")
    (checkpoint_dir / "checkpoint_250.pt").write_text("stub\n", encoding="utf-8")

    worker = ActorWorker(
        actor_id=7,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        checkpoint_dir=checkpoint_dir,
        reload_interval_updates=2,
    )

    first = worker.poll_checkpoint_sync()
    assert first == {"loaded_checkpoint_update": 0, "checkpoint_lag_updates": 250}

    second = worker.poll_checkpoint_sync()
    assert second == {"loaded_checkpoint_update": 250, "checkpoint_lag_updates": 0}

    (checkpoint_dir / "checkpoint_400.pt").write_text("stub\n", encoding="utf-8")
    third = worker.poll_checkpoint_sync()
    assert third == {"loaded_checkpoint_update": 250, "checkpoint_lag_updates": 150}

    fourth = worker.poll_checkpoint_sync()
    assert fourth == {"loaded_checkpoint_update": 400, "checkpoint_lag_updates": 0}


def test_actor_worker_ignores_malformed_checkpoint_names(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_latest.pt").write_text("bad\n", encoding="utf-8")
    (checkpoint_dir / "checkpoint_50.pt").write_text("stub\n", encoding="utf-8")

    worker = ActorWorker(
        actor_id=1,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        checkpoint_dir=checkpoint_dir,
        reload_interval_updates=1,
    )

    result = worker.poll_checkpoint_sync()
    assert result == {"loaded_checkpoint_update": 50, "checkpoint_lag_updates": 0}
