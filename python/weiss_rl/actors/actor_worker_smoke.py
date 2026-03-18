# Test for ids_offsets layout (Validation of M3-03)
from __future__ import annotations

import numpy as np

from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.masking import masked_logp_from_legal_ids, resolve_pass_action_id


OBS_LEN = 8
ACTION_SPACE = 16

class FakeBatch:
    def __init__(self, N: int):
        self.obs = np.zeros((N, OBS_LEN), dtype=np.int16)
        self.legal_ids = None
        self.legal_offsets = None
        self.rewards = np.zeros((N,), dtype=np.float32)
        self.terminated = np.zeros((N,), dtype=np.bool_)
        self.truncated = np.zeros((N,), dtype=np.bool_)
        self.engine_status = np.zeros((N,), dtype=np.int32)
        self.decision_id = np.zeros((N,), dtype=np.int32)
        self.actor = np.zeros((N,), dtype=np.int8)  # to_play_seat
        self.episode_seed = np.zeros((N,), dtype=np.uint64)
        self.episode_key = np.zeros((N,), dtype=np.uint64)

class FakeEnvIdsOffsets:
    def __init__(self, N: int, *, seed: int = 0):
        self.N = N
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        return self._make_batch()

    def step(self, actions: np.ndarray):
        # We ignore actions; we just advance
        self.t += 1
        return self._make_batch()

    def _make_batch(self):
        b = FakeBatch(self.N)

        # obs: deterministic but changes with t
        b.obs[:] = (self.t + np.arange(self.N)[:, None]).astype(np.int16)

        # alternate to_play_seat
        b.actor[:] = (np.arange(self.N) + self.t) % 2

        # decision_id: monotonic per env
        b.decision_id[:] = self.t

        # packed legal ids: vary per env; include an empty-legal row occasionally
        # env 0: empty legal every 3 steps to exercise pass fallback
        legal_slices = []
        offsets = [0]
        for n in range(self.N):
            if n == 0 and (self.t % 3 == 0):
                ids = np.array([], dtype=np.int32)
            else:
                k = 3 + (n % 3)
                ids = np.array(sorted(self.rng.choice(ACTION_SPACE, size=k, replace=False)), dtype=np.int32)
            legal_slices.append(ids)
            offsets.append(offsets[-1] + ids.size)

        b.legal_ids = np.concatenate(legal_slices, axis=0) if offsets[-1] > 0 else np.array([], dtype=np.int32)
        b.legal_offsets = np.array(offsets, dtype=np.uint32)

        return b


def fake_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    # deterministic logits from obs to make recomputation stable
    N = obs.shape[0]
    logits = np.zeros((N, ACTION_SPACE), dtype=np.float32)
    base = obs[:, 0].astype(np.float32) + to_play.astype(np.float32)
    logits[:] = base[:, None] * 0.01 + np.arange(ACTION_SPACE, dtype=np.float32)[None, :] * 0.001
    return logits


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
    assert batch.behavior_logp.shape == (T, N)

    pass_id = resolve_pass_action_id()

    # Recompute behavior logp step-by-step using stored legality.
    # This assumes your UnrollBatch stores per-step legality in lists:
    # batch.legal_ids_steps[t], batch.legal_offsets_steps[t]
    # If you stored it differently, adjust this loop accordingly.
    for t in range(T):
        obs_t = batch.obs[t]
        to_play_t = batch.to_play_seat[t]
        logits_t = fake_policy_logits(obs_t, to_play_t)

        legal_ids_t = batch.legal_ids_steps[t]
        legal_offsets_t = batch.legal_offsets_steps[t]

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