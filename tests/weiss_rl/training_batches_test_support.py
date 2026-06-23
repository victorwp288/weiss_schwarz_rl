from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from weiss_rl.training.batches import MinimalRollout


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def minimal_rollout(*, truncated_bootstrap_actor: np.ndarray | None = None) -> MinimalRollout:
    bootstrap_actor = np.array([0], dtype=np.int64) if truncated_bootstrap_actor is None else truncated_bootstrap_actor
    return MinimalRollout(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_mask=np.ones((2, 1, 2), dtype=np.bool_),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        logits=np.zeros((2, 1, 2), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        bootstrap_obs=np.arange(len(bootstrap_actor), dtype=np.float32).reshape(len(bootstrap_actor), 1),
        bootstrap_actor=bootstrap_actor,
    )


class RuntimeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def collect_update_batch(self, **kwargs: object) -> str:
        self.calls.append(("impala", kwargs))
        return "impala_batch"

    def collect_policy_batch(self, **kwargs: object) -> str:
        self.calls.append(("ppo", kwargs))
        return "ppo_batch"


def training_config() -> SimpleNamespace:
    return SimpleNamespace(
        vtrace_rho_bar=1.25,
        vtrace_c_bar=0.75,
        ppo_gae_lambda=0.92,
    )


def rewards_config() -> SimpleNamespace:
    return SimpleNamespace(
        gamma=0.99,
        truncation=SimpleNamespace(
            reward=-0.5,
            bootstrap_value=True,
        ),
    )
