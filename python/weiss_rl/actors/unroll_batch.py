"""Fixed-shape actor unroll buffers and learner batch records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

LayoutName = Literal["i16_legal_ids", "mask"]


@dataclass(slots=True)
class UnrollBatch:
    """Fixed-shape unroll: T x N, ready for learner consumption."""

    T: int
    N: int
    layout_name: LayoutName
    obs: np.ndarray
    to_play_seat: np.ndarray
    decision_id: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    engine_status: np.ndarray
    episode_seed: np.ndarray
    episode_key: np.ndarray
    behavior_logp: np.ndarray
    legal_ids: np.ndarray | None = None
    legal_offsets: np.ndarray | None = None
    legal_mask: np.ndarray | None = None
    entropy: np.ndarray | None = None
    counters: dict[str, int] | None = None


@dataclass(slots=True)
class ActorUnrollBuffers:
    """Mutable collection buffers for one actor unroll."""

    T: int
    N: int
    layout_name: LayoutName
    obs: np.ndarray
    to_play_seat: np.ndarray
    decision_id: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    engine_status: np.ndarray
    episode_seed: np.ndarray
    episode_key: np.ndarray
    behavior_logp: np.ndarray
    entropy: np.ndarray
    packed_legal_ids: list[np.ndarray]
    packed_legal_offsets: list[np.ndarray]
    legal_mask: np.ndarray | None

    @classmethod
    def allocate(
        cls,
        *,
        T: int,
        N: int,
        obs_width: int,
        obs_dtype: np.dtype,
        action_space: int,
        layout_name: LayoutName,
    ) -> ActorUnrollBuffers:
        legal_mask = None
        if layout_name == "mask":
            legal_mask = np.empty((T, N, action_space), dtype=np.bool_)
        return cls(
            T=T,
            N=N,
            layout_name=layout_name,
            obs=np.empty((T, N, obs_width), dtype=obs_dtype),
            to_play_seat=np.empty((T, N), dtype=np.int8),
            decision_id=np.empty((T, N), dtype=np.int32),
            action=np.empty((T, N), dtype=np.uint32),
            reward=np.empty((T, N), dtype=np.float32),
            terminated=np.empty((T, N), dtype=np.bool_),
            truncated=np.empty((T, N), dtype=np.bool_),
            engine_status=np.empty((T, N), dtype=np.int32),
            episode_seed=np.empty((T, N), dtype=np.uint64),
            episode_key=np.empty((T, N), dtype=np.uint64),
            behavior_logp=np.empty((T, N), dtype=np.float32),
            entropy=np.empty((T, N), dtype=np.float32),
            packed_legal_ids=[],
            packed_legal_offsets=[np.array([0], dtype=np.uint32)],
            legal_mask=legal_mask,
        )

    @property
    def next_legal_offset(self) -> int:
        return int(self.packed_legal_offsets[-1][-1])

    def append_legal_ids(self, legal_ids: np.ndarray, legal_offsets: np.ndarray) -> None:
        self.packed_legal_ids.append(legal_ids)
        self.packed_legal_offsets.append(legal_offsets)

    def record_legal_mask(self, t: int, legal_mask: np.ndarray) -> None:
        if self.legal_mask is None:
            self.legal_mask = np.empty((self.T, self.N, legal_mask.shape[1]), dtype=legal_mask.dtype)
        self.legal_mask[t] = legal_mask

    def record_step(
        self,
        *,
        t: int,
        obs: np.ndarray,
        to_play: np.ndarray,
        decision_id: np.ndarray,
        actions: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        engine_status: np.ndarray,
        episode_seed: np.ndarray,
        episode_key: np.ndarray,
        behavior_logp: np.ndarray,
        entropy: np.ndarray,
    ) -> None:
        self.obs[t] = obs
        self.to_play_seat[t] = to_play.astype(np.int8, copy=False)
        self.decision_id[t] = decision_id.astype(np.int32, copy=False)
        self.action[t] = actions.astype(np.uint32, copy=False)
        self.reward[t] = reward.astype(np.float32, copy=False)
        self.terminated[t] = terminated.astype(np.bool_, copy=False)
        self.truncated[t] = truncated.astype(np.bool_, copy=False)
        self.engine_status[t] = engine_status.astype(np.int32, copy=False)
        self.episode_seed[t] = episode_seed
        self.episode_key[t] = episode_key
        self.behavior_logp[t] = behavior_logp.astype(np.float32, copy=False)
        self.entropy[t] = entropy.astype(np.float32, copy=False)

    def to_batch(self, *, counters: dict[str, int]) -> UnrollBatch:
        legal_ids: np.ndarray | None
        legal_offsets: np.ndarray | None
        legal_mask: np.ndarray | None
        if self.layout_name == "i16_legal_ids":
            legal_ids = (
                np.concatenate(self.packed_legal_ids, axis=0).astype(np.int32, copy=False)
                if self.packed_legal_ids
                else np.zeros((0,), dtype=np.int32)
            )
            legal_offsets = np.concatenate(self.packed_legal_offsets, axis=0).astype(np.uint32, copy=False)
            legal_mask = None
        else:
            legal_ids = None
            legal_offsets = None
            legal_mask = self.legal_mask

        return UnrollBatch(
            T=self.T,
            N=self.N,
            layout_name=self.layout_name,
            obs=self.obs,
            to_play_seat=self.to_play_seat,
            decision_id=self.decision_id,
            action=self.action,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            engine_status=self.engine_status,
            episode_seed=self.episode_seed,
            episode_key=self.episode_key,
            behavior_logp=self.behavior_logp,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_mask=legal_mask,
            entropy=self.entropy,
            counters=counters,
        )


__all__ = ["ActorUnrollBuffers", "LayoutName", "UnrollBatch"]
