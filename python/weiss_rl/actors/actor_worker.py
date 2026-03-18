"""Actor worker scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from weiss_rl.masking import (
    MaskingAnomalyCounters,
    apply_empty_legal_action_fallback,
    masked_log_softmax,
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
    resolve_pass_action_id,
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
    legal_ids: np.ndarray | None = None  # (sum_k,) int
    legal_offsets: np.ndarray | None = None  # (T, N+1) or flattened offsets
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
        - env: a simulator wrapper you already have (WeissEnv or something exposing reset/step batches)
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

        # Allocate fixed-shape arrays.
        # These dtypes are chosen to align with master-plan intent.
        obs_buf = None  # allocated after first reset when obs dtype known
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

        # For ids_offsets layout, store packed legal_ids and per-step offsets.
        # We store offsets per t as (N+1,) pointing into one concatenated legal_ids vector for that t.
        legal_ids_steps: list[np.ndarray] = []
        legal_offsets_steps: list[np.ndarray] = []

        # For mask layout.
        legal_mask_buf = None

        # Reset once at the start of the unroll.
        batch = env.reset()

        # Allocate obs buffer once we see obs shape.
        obs0 = np.asarray(batch.obs)
        if obs0.ndim != 2 or obs0.shape[0] != N:
            raise ValueError("expected batch.obs shape (N, OBS_LEN)")
        obs_buf = np.empty((T, N, obs0.shape[1]), dtype=obs0.dtype)

        # Main rollout loop.
        for t in range(T):
            obs = np.asarray(batch.obs)
            to_play = np.asarray(batch.actor) if hasattr(batch, "actor") else np.asarray(batch.to_play_seat)
            decision_id = np.asarray(batch.decision_id)
            rewards = np.asarray(batch.rewards)
            terminated = np.asarray(batch.terminated)
            truncated = np.asarray(batch.truncated)
            engine_status = np.asarray(batch.engine_status)

            # These two may not exist depending on simulator layout; guard conservatively.
            episode_seed = np.asarray(getattr(batch, "episode_seed", np.zeros((N,), dtype=np.uint64)), dtype=np.uint64)
            episode_key = np.asarray(getattr(batch, "episode_key", np.zeros((N,), dtype=np.uint64)), dtype=np.uint64)

            if obs.shape[0] != N:
                raise ValueError("batch dimension mismatch")

            logits = np.asarray(policy_logits_fn(obs, to_play), dtype=np.float32)
            if logits.shape != (N, A):
                raise ValueError(f"policy_logits_fn must return shape (N, A)=({N}, {A})")

            if self.layout_name == "i16_legal_ids":
                # Expect packed legal ids surfaces in the batch.
                legal_ids = np.asarray(batch.legal_ids)
                legal_offsets = np.asarray(batch.legal_offsets)
                # sample actions + behavior logp + entropy
                actions, logp, ent = sample_actions_from_legal_ids(
                    logits,
                    legal_ids,
                    legal_offsets,
                    rng=rng,
                    counters=anomaly,
                    pass_action_id=pass_action_id,
                )
                # record legal surfaces per step (keep per-step packed representation)
                legal_ids_steps.append(np.asarray(legal_ids))
                legal_offsets_steps.append(np.asarray(legal_offsets))
            else:
                # mask layout
                legal_mask = np.asarray(batch.masks)
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

            # Write step fields into buffers.
            obs_buf[t] = obs
            to_play_buf[t] = to_play.astype(np.int8, copy=False)
            decision_id_buf[t] = decision_id.astype(np.int32, copy=False)
            action_buf[t] = actions.astype(np.uint32, copy=False)
            reward_buf[t] = rewards.astype(np.float32, copy=False)
            terminated_buf[t] = terminated.astype(np.bool_, copy=False)
            truncated_buf[t] = truncated.astype(np.bool_, copy=False)
            engine_status_buf[t] = engine_status.astype(np.int32, copy=False)
            episode_seed_buf[t] = episode_seed
            episode_key_buf[t] = episode_key
            behavior_logp_buf[t] = logp.astype(np.float32, copy=False)
            entropy_buf[t] = ent.astype(np.float32, copy=False)

            # Step environment.
            batch = env.step(action_buf[t])

        # Package legality surfaces.
        if self.layout_name == "i16_legal_ids":
            # For M3-03 we can keep a simple per-step list and concatenate.
            # M3-04 will likely switch this into a single packed buffer with offsets.
            legal_ids_packed = np.concatenate(legal_ids_steps, axis=0) if legal_ids_steps else np.zeros((0,), dtype=np.int32)

            # Offsets are per-step, so keep (T, N+1) for now (fixed-shape).
            legal_offsets_mat = np.stack(legal_offsets_steps, axis=0).astype(np.uint32, copy=False)
            legal_mask_final = None
            legal_ids_final = legal_ids_packed
            legal_offsets_final = legal_offsets_mat
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

