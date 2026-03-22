# python/weiss_rl/actors/actor_worker_smoke_torch.py
from __future__ import annotations
import numpy as np

from weiss_rl.actors.actor_worker import ActorWorker

try:
    import torch
except Exception:
    torch = None

from weiss_rl.actors.actor_worker_smoke import ACTION_SPACE, FakeEnvIdsOffsets

def torch_policy_logits(obs: np.ndarray, to_play: np.ndarray):
    assert torch is not None
    x = torch.from_numpy(obs.astype(np.float32, copy=False))
    # dummy linear-ish logits, requires_grad should be False inside inference_mode
    logits = x[:, :1] * 0.01 + torch.arange(ACTION_SPACE, dtype=torch.float32)[None, :] * 0.001
    assert logits.requires_grad is False
    return logits

def main():
    if torch is None:
        print("skip (torch not installed)")
        return

    T, N = 5, 4
    env = FakeEnvIdsOffsets(N, seed=123)

    worker = ActorWorker(
        actor_id=0,
        unroll_length=T,
        num_envs=N,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=999,
        actor_torch_threads=1,
    )

    batch = worker.run_once(env=env, policy_logits_fn=torch_policy_logits)

    assert batch.behavior_logp.shape == (T, N)
    assert batch.action.shape == (T, N)

    # Thread pinning check
    assert torch.get_num_threads() == 1

    print("ok")

if __name__ == "__main__":
    main()