from __future__ import annotations

from pathlib import Path

import numpy as np

from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask, resolve_pass_action_id

OBS_LEN = 6
ACTION_SPACE = 64


class IdsBatch:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, OBS_LEN), dtype=np.int16)
        self.reward = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=np.bool_)
        self.truncated = np.zeros((num_envs,), dtype=np.bool_)
        self.engine_status = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int32)
        self.to_play = np.zeros((num_envs,), dtype=np.int8)
        self.actor = self.to_play
        self.episode_seed = np.zeros((num_envs,), dtype=np.uint64)
        self.episode_key = np.zeros((num_envs,), dtype=np.uint64)
        self.ids_offsets: tuple[np.ndarray, np.ndarray] | None = None


class FakeIdsEnv:
    def __init__(self, num_envs: int, *, seed: int = 0) -> None:
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.step_index = 0

    def reset(self) -> IdsBatch:
        self.step_index = 0
        return self._batch()

    def step(self, actions: np.ndarray) -> IdsBatch:
        assert actions.shape == (self.num_envs,)
        self.step_index += 1
        return self._batch()

    def _batch(self) -> IdsBatch:
        batch = IdsBatch(self.num_envs)
        batch.obs[:] = (self.step_index + np.arange(self.num_envs)[:, None]).astype(np.int16)
        batch.to_play[:] = (np.arange(self.num_envs) + self.step_index) % 2
        batch.decision_id[:] = self.step_index

        legal_slices = []
        offsets = [0]
        for env_index in range(self.num_envs):
            if env_index == 0 and self.step_index % 2 == 0:
                legal_ids = np.array([], dtype=np.int32)
            else:
                size = 2 + env_index
                legal_ids = np.sort(self.rng.choice(ACTION_SPACE, size=size, replace=False)).astype(np.int32)
            legal_slices.append(legal_ids)
            offsets.append(offsets[-1] + int(legal_ids.size))

        batch.ids_offsets = (
            np.concatenate(legal_slices, axis=0) if offsets[-1] else np.array([], dtype=np.int32),
            np.array(offsets, dtype=np.uint32),
        )
        return batch


class MaskBatch:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, OBS_LEN), dtype=np.int16)
        self.rewards = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=np.bool_)
        self.truncated = np.zeros((num_envs,), dtype=np.bool_)
        self.engine_status = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int32)
        self.actor = np.zeros((num_envs,), dtype=np.int8)
        self.episode_seed = np.zeros((num_envs,), dtype=np.uint64)
        self.episode_key = np.zeros((num_envs,), dtype=np.uint64)
        self.masks = np.zeros((num_envs, ACTION_SPACE), dtype=np.uint8)


class FakeMaskEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.step_index = 0

    def reset(self) -> MaskBatch:
        self.step_index = 0
        return self._batch()

    def step(self, actions: np.ndarray) -> MaskBatch:
        assert actions.shape == (self.num_envs,)
        self.step_index += 1
        return self._batch()

    def _batch(self) -> MaskBatch:
        batch = MaskBatch(self.num_envs)
        batch.obs[:] = (self.step_index + np.arange(self.num_envs)[:, None]).astype(np.int16)
        batch.actor[:] = (np.arange(self.num_envs) + self.step_index) % 2
        batch.decision_id[:] = self.step_index
        batch.masks[0, [1, 3, 5]] = 1
        batch.masks[1, :] = 0
        return batch


def _policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    logits = np.zeros((obs.shape[0], ACTION_SPACE), dtype=np.float32)
    base = obs[:, 0].astype(np.float32) + to_play.astype(np.float32)
    logits[:] = base[:, None] * 0.01 + np.arange(ACTION_SPACE, dtype=np.float32)[None, :] * 0.001
    return logits


def _step_legal_slice(batch, t: int) -> tuple[np.ndarray, np.ndarray]:
    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None

    row_start = t * batch.N
    row_stop = row_start + batch.N
    offset_start = int(batch.legal_offsets[row_start])
    offset_stop = int(batch.legal_offsets[row_stop])
    legal_ids = batch.legal_ids[offset_start:offset_stop]
    legal_offsets = batch.legal_offsets[row_start : row_stop + 1] - offset_start
    return legal_ids, legal_offsets


def test_actor_worker_ids_offsets_preserves_behavior_logp_contract() -> None:
    worker = ActorWorker(
        actor_id=0,
        unroll_length=4,
        num_envs=3,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=7,
    )

    batch = worker.run_once(env=FakeIdsEnv(3, seed=11), policy_logits_fn=_policy_logits)

    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None
    assert batch.legal_offsets.shape == (batch.T * batch.N + 1,)

    pass_action_id = resolve_pass_action_id()
    for t in range(batch.T):
        logits = _policy_logits(batch.obs[t], batch.to_play_seat[t])
        legal_ids, legal_offsets = _step_legal_slice(batch, t)
        recomputed = masked_logp_from_legal_ids(
            logits,
            legal_ids,
            legal_offsets,
            batch.action[t].astype(np.int64, copy=False),
            pass_action_id=pass_action_id,
        )
        assert np.array_equal(recomputed, batch.behavior_logp[t])


def test_actor_worker_mask_layout_returns_entropy_and_pass_fallback() -> None:
    T = 3
    N = 2
    worker = ActorWorker(
        actor_id=1,
        unroll_length=T,
        num_envs=N,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=5,
    )

    batch = worker.run_once(env=FakeMaskEnv(2), policy_logits_fn=_policy_logits)

    assert batch.legal_mask is not None
    assert batch.legal_mask.shape == (T, N, ACTION_SPACE)
    assert batch.legal_ids is None
    assert batch.legal_offsets is None
    assert batch.entropy is not None
    assert batch.entropy.shape == (T, N)

    pass_action_id = resolve_pass_action_id()
    recomputed = masked_logp_from_mask(
        _policy_logits(batch.obs.reshape(T * N, OBS_LEN), batch.to_play_seat.reshape(T * N)),
        batch.legal_mask.reshape(T * N, ACTION_SPACE),
        batch.action.reshape(T * N).astype(np.int64, copy=False),
        pass_action_id=pass_action_id,
    )

    assert np.allclose(recomputed.reshape(T, N), batch.behavior_logp, atol=0.0, rtol=0.0)
    assert np.all(batch.action[:, 1] == pass_action_id)
    assert np.all(batch.behavior_logp[:, 1] == 0.0)
    assert batch.counters == {"empty_legal": batch.T}


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
