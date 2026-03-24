"""Smoke test for ActorWorker ids/offsets unroll collection."""

from __future__ import annotations

import numpy as np

from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.masking import masked_logp_from_legal_ids, resolve_pass_action_id

OBS_LEN = 8
ACTION_SPACE = 52


class FakeBatch:
    def __init__(self, num_envs: int):
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


class FakeEnvIdsOffsets:
    def __init__(self, num_envs: int, *, seed: int = 0):
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        return self._make_batch()

    def step(self, actions: np.ndarray):
        _ = actions
        self.t += 1
        return self._make_batch()

    def _make_batch(self):
        batch = FakeBatch(self.num_envs)
        batch.obs[:] = (self.t + np.arange(self.num_envs)[:, None]).astype(np.int16)
        batch.to_play[:] = (np.arange(self.num_envs) + self.t) % 2
        batch.decision_id[:] = self.t

        legal_slices = []
        offsets = [0]
        for env_index in range(self.num_envs):
            if env_index == 0 and (self.t % 3 == 0):
                ids = np.array([], dtype=np.int32)
            else:
                count = 3 + (env_index % 3)
                ids = np.array(sorted(self.rng.choice(ACTION_SPACE, size=count, replace=False)), dtype=np.int32)
            legal_slices.append(ids)
            offsets.append(offsets[-1] + ids.size)

        legal_ids = np.concatenate(legal_slices, axis=0) if offsets[-1] > 0 else np.array([], dtype=np.int32)
        legal_offsets = np.array(offsets, dtype=np.uint32)
        batch.ids_offsets = (legal_ids, legal_offsets)
        return batch


def fake_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    num_envs = obs.shape[0]
    logits = np.zeros((num_envs, ACTION_SPACE), dtype=np.float32)
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
    step_legal_ids = batch.legal_ids[offset_start:offset_stop]
    step_legal_offsets = batch.legal_offsets[row_start : row_stop + 1] - offset_start
    return step_legal_ids, step_legal_offsets


def main():
    T = 5
    N = 4
    env = FakeEnvIdsOffsets(N, seed=123)

    worker = ActorWorker(
        actor_id=0,
        unroll_length=T,
        num_envs=N,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=999,
    )

    batch = worker.run_once(env=env, policy_logits_fn=fake_policy_logits)

    assert batch.obs.shape == (T, N, OBS_LEN)
    assert batch.action.shape == (T, N)
    assert batch.reward.shape == (T, N)
    assert batch.behavior_logp.shape == (T, N)
    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None
    assert batch.legal_offsets.shape == (T * N + 1,)

    pass_id = resolve_pass_action_id()

    for t in range(T):
        obs_t = batch.obs[t]
        to_play_t = batch.to_play_seat[t]
        logits_t = fake_policy_logits(obs_t, to_play_t)

        legal_ids_t, legal_offsets_t = _step_legal_slice(batch, t)
        actions_t = batch.action[t].astype(np.int64, copy=False)
        recomputed = masked_logp_from_legal_ids(
            logits_t,
            legal_ids_t,
            legal_offsets_t,
            actions_t,
            pass_action_id=pass_id,
        )
        if not np.allclose(recomputed, batch.behavior_logp[t], atol=0.0, rtol=0.0):
            raise AssertionError(f"logp mismatch at t={t}")

    print("ok")


if __name__ == "__main__":
    main()
