"""Actor worker scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from weiss_rl.masking import (
    MaskingAnomalyCounters,
    masked_logp_from_legal_ids,
    resolve_pass_action_id,
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
)

LayoutName = Literal["i16_legal_ids", "mask"]


@dataclass(slots=True)
class UnrollBatch:
    """
    Fixed-shape unroll: T x N, ready for learner consumption.

    This is the minimal M3-03 contract: fixed shape + behavior_logp present.

    Notes:
    - obs dtype/layout depends on simulator layout; keep as np.ndarray.
    - legal surfaces: either (legal_ids, legal_offsets) OR (legal_mask).
    - ids/offsets batches use one coherent flattened packed representation over
      the row-major decision order of ``obs.reshape(T * N, ...)``.
    """

    # dimensions
    T: int
    N: int
    layout_name: LayoutName

    # per-step fields (T, N, ...)
    obs: np.ndarray  # (T, N, OBS_LEN) int16 or int32
    to_play_seat: np.ndarray  # (T, N) int8
    decision_id: np.ndarray  # (T, N) int32
    action: np.ndarray  # (T, N) uint32 or int64
    reward: np.ndarray  # (T, N) float32
    terminated: np.ndarray  # (T, N) bool
    truncated: np.ndarray  # (T, N) bool
    engine_status: np.ndarray  # (T, N) int32
    episode_seed: np.ndarray  # (T, N) uint64
    episode_key: np.ndarray  # (T, N) uint64 or bytes-like object array
    behavior_logp: np.ndarray  # (T, N) float32

    # legality surfaces (layout-dependent)
    legal_ids: np.ndarray | None = None  # (sum_k,) int, flattened over T*N rows
    legal_offsets: np.ndarray | None = None  # (T*N + 1,) flattened packed offsets
    legal_mask: np.ndarray | None = None  # (T, N, A) uint8/bool

    # optional debug
    entropy: np.ndarray | None = None  # (T, N) float32
    counters: dict[str, int] | None = None  # anomaly counters snapshot


@dataclass(slots=True)
class ActorWorker:
    actor_id: int
    unroll_length: int
    num_envs: int
    action_space: int
    layout_name: LayoutName = "i16_legal_ids"
    seed: int = 0

    def run_once(
        self,
        *,
        env: Any,
        policy_logits_fn: Any,
    ) -> UnrollBatch:
        """
        Collect one fixed-length unroll of shape (T, N).

        Parameters
        - env: ideally a DecisionBoundary-style env exposing reset()/step() that
          returns batches with reward + mask/ids_offsets.
        - policy_logits_fn: callable that returns logits given obs (+ maybe seat).
            Signature expected (minimal): logits = policy_logits_fn(obs_batch, to_play_seat_batch)
            where logits is (N, A) float32

        Returns
        - UnrollBatch with behavior_logp filled.
        """
        T = int(self.unroll_length)
        N = int(self.num_envs)
        A = int(self.action_space)
        if T <= 0 or N <= 0 or A <= 0:
            raise ValueError("unroll_length, num_envs, action_space must be > 0")

        rng = np.random.default_rng(self.seed + self.actor_id)
        pass_action_id = resolve_pass_action_id()
        anomaly = MaskingAnomalyCounters()

        obs_buf: np.ndarray | None = None
        to_play_buf = np.empty((T, N), dtype=np.int8)
        decision_id_buf = np.empty((T, N), dtype=np.int32)
        action_buf = np.empty((T, N), dtype=np.uint32)
        reward_buf = np.empty((T, N), dtype=np.float32)
        terminated_buf = np.empty((T, N), dtype=np.bool_)
        truncated_buf = np.empty((T, N), dtype=np.bool_)
        engine_status_buf = np.empty((T, N), dtype=np.int32)
        episode_seed_buf = np.empty((T, N), dtype=np.uint64)
        episode_key_buf = np.empty((T, N), dtype=np.uint64)
        behavior_logp_buf = np.empty((T, N), dtype=np.float32)
        entropy_buf = np.empty((T, N), dtype=np.float32)

        packed_legal_ids: list[np.ndarray] = []
        packed_legal_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
        legal_mask_buf: np.ndarray | None = None

        batch = env.reset()

        obs0 = np.asarray(batch.obs)
        if obs0.ndim != 2 or obs0.shape[0] != N:
            raise ValueError("expected batch.obs shape (N, OBS_LEN)")
        obs_buf = np.empty((T, N, obs0.shape[1]), dtype=obs0.dtype)

        for t in range(T):
            obs = np.asarray(batch.obs)
            to_play = _batch_to_play(batch)
            decision_id = np.asarray(batch.decision_id)
            reward = _batch_reward(batch)
            terminated = np.asarray(batch.terminated)
            truncated = np.asarray(batch.truncated)
            engine_status = np.asarray(batch.engine_status)
            episode_seed = np.asarray(getattr(batch, "episode_seed", np.zeros((N,), dtype=np.uint64)), dtype=np.uint64)
            episode_key = np.asarray(getattr(batch, "episode_key", np.zeros((N,), dtype=np.uint64)), dtype=np.uint64)

            if obs.shape != (N, obs_buf.shape[2]):
                raise ValueError("batch.obs shape changed within unroll")

            logits = np.asarray(policy_logits_fn(obs, to_play), dtype=np.float32)
            if logits.shape != (N, A):
                raise ValueError(f"policy_logits_fn must return shape (N, A)=({N}, {A})")

            if self.layout_name == "i16_legal_ids":
                legal_ids, legal_offsets = _batch_legal_ids_offsets(batch)
                actions, logp, ent = sample_actions_from_legal_ids(
                    logits,
                    legal_ids,
                    legal_offsets,
                    rng=rng,
                    counters=anomaly,
                    pass_action_id=pass_action_id,
                )

                base = int(packed_legal_offsets[-1][-1])
                packed_legal_ids.append(legal_ids.astype(np.int32, copy=False))
                packed_legal_offsets.append((legal_offsets[1:] + base).astype(np.uint32, copy=False))
            else:
                legal_mask = _batch_legal_mask(batch)
                if legal_mask.shape != (N, A):
                    raise ValueError(f"expected legal_mask shape (N, A)=({N}, {A})")
                if legal_mask_buf is None:
                    legal_mask_buf = np.empty((T, N, A), dtype=legal_mask.dtype)
                actions, logp, ent = sample_actions_from_mask(
                    logits,
                    legal_mask,
                    rng=rng,
                    counters=anomaly,
                    pass_action_id=pass_action_id,
                )
                legal_mask_buf[t] = legal_mask

            obs_buf[t] = obs
            to_play_buf[t] = to_play.astype(np.int8, copy=False)
            decision_id_buf[t] = decision_id.astype(np.int32, copy=False)
            action_buf[t] = actions.astype(np.uint32, copy=False)
            reward_buf[t] = reward.astype(np.float32, copy=False)
            terminated_buf[t] = terminated.astype(np.bool_, copy=False)
            truncated_buf[t] = truncated.astype(np.bool_, copy=False)
            engine_status_buf[t] = engine_status.astype(np.int32, copy=False)
            episode_seed_buf[t] = episode_seed
            episode_key_buf[t] = episode_key
            behavior_logp_buf[t] = logp.astype(np.float32, copy=False)
            entropy_buf[t] = ent.astype(np.float32, copy=False)

            batch = env.step(action_buf[t])

        if self.layout_name == "i16_legal_ids":
            legal_ids_final = (
                np.concatenate(packed_legal_ids, axis=0).astype(np.int32, copy=False)
                if packed_legal_ids
                else np.zeros((0,), dtype=np.int32)
            )
            legal_offsets_final = np.concatenate(packed_legal_offsets, axis=0).astype(np.uint32, copy=False)
            legal_mask_final = None
        else:
            legal_ids_final = None
            legal_offsets_final = None
            legal_mask_final = legal_mask_buf

        return UnrollBatch(
            T=T,
            N=N,
            layout_name=self.layout_name,
            obs=obs_buf,
            to_play_seat=to_play_buf,
            decision_id=decision_id_buf,
            action=action_buf,
            reward=reward_buf,
            terminated=terminated_buf,
            truncated=truncated_buf,
            engine_status=engine_status_buf,
            episode_seed=episode_seed_buf,
            episode_key=episode_key_buf,
            behavior_logp=behavior_logp_buf,
            legal_ids=legal_ids_final,
            legal_offsets=legal_offsets_final,
            legal_mask=legal_mask_final,
            entropy=entropy_buf,
            counters={"empty_legal": anomaly.empty_legal},
        )


def actor_behavior_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    """Compatibility wrapper kept for older masking tests and actor code."""
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def _batch_to_play(batch: Any) -> np.ndarray:
    if hasattr(batch, "to_play"):
        return np.asarray(batch.to_play)
    if hasattr(batch, "to_play_seat"):
        return np.asarray(batch.to_play_seat)
    if hasattr(batch, "actor"):
        return np.asarray(batch.actor)
    raise AttributeError("batch must expose .to_play, .to_play_seat, or .actor")


def _batch_reward(batch: Any) -> np.ndarray:
    if hasattr(batch, "reward"):
        return np.asarray(batch.reward)
    if hasattr(batch, "rewards"):
        return np.asarray(batch.rewards)
    raise AttributeError("batch must expose .reward or .rewards")


def _batch_legal_mask(batch: Any) -> np.ndarray:
    if hasattr(batch, "mask"):
        return np.asarray(batch.mask)
    if hasattr(batch, "masks"):
        return np.asarray(batch.masks)
    raise AttributeError("mask layout batch must expose .mask or .masks")


def _batch_legal_ids_offsets(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    ids_offsets = getattr(batch, "ids_offsets", None)
    if ids_offsets is not None:
        legal_ids, legal_offsets = ids_offsets
        return np.asarray(legal_ids), np.asarray(legal_offsets)

    if hasattr(batch, "legal_ids") and hasattr(batch, "legal_offsets"):
        return np.asarray(batch.legal_ids), np.asarray(batch.legal_offsets)

    raise AttributeError("ids_offsets layout batch must expose .ids_offsets or (.legal_ids, .legal_offsets)")
