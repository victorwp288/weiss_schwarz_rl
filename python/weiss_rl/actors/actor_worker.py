"""Actor worker scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np

from weiss_rl.masking import (
    MaskingAnomalyCounters,
    masked_logp_from_legal_ids,
    resolve_pass_action_id,
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
)

torch: ModuleType | None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


LayoutName = Literal["i16_legal_ids", "mask"]

_CHECKPOINT_METADATA_STEM = re.compile(r"(?:checkpoint_metadata|checkpoint)_(\d+)")


def _configure_actor_torch_threads(actor_torch_threads: int) -> None:
    if torch is None:
        return
    threads = int(actor_torch_threads)
    if threads < 1:
        raise ValueError("actor_torch_threads must be >= 1")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


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
class ActorWorker:
    actor_id: int
    unroll_length: int
    num_envs: int
    action_space: int
    layout_name: LayoutName = "i16_legal_ids"
    seed: int = 0
    actor_torch_threads: int = 1
    checkpoint_dir: Path | None = None
    reload_interval_updates: int = 1000  # Deprecated alias for checkpoint metadata polling cadence.

    update_count: int = field(default=0, init=False)
    observed_checkpoint_update: int = field(default=0, init=False)
    last_observed_checkpoint_update: int = field(default=-1, init=False)
    checkpoint_metadata_lag_updates: int = field(default=0, init=False)
    _torch_threads_configured: bool = field(default=False, init=False)
    _rng: np.random.Generator | None = field(default=None, init=False)

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
        self.poll_checkpoint_metadata()
        T = int(self.unroll_length)
        N = int(self.num_envs)
        A = int(self.action_space)
        if T <= 0 or N <= 0 or A <= 0:
            raise ValueError("unroll_length, num_envs, action_space must be > 0")

        if not self._torch_threads_configured:
            _configure_actor_torch_threads(self.actor_torch_threads)
            self._torch_threads_configured = True
            if torch is not None and int(torch.get_num_threads()) != int(self.actor_torch_threads):
                raise RuntimeError(
                    f"torch threads mismatch: got {torch.get_num_threads()}, want {self.actor_torch_threads}"
                )

        if self._rng is None:
            self._rng = np.random.default_rng(self.seed + self.actor_id)

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
            episode_seed, episode_key = _batch_episode_identity(batch, num_envs=N)

            if obs.shape != (N, obs_buf.shape[2]):
                raise ValueError("batch.obs shape changed within unroll")

            logits = _policy_logits(policy_logits_fn, obs, to_play)
            if logits.shape != (N, A):
                raise ValueError(f"policy_logits_fn must return shape (N, A)=({N}, {A})")

            if self.layout_name == "i16_legal_ids":
                legal_ids, legal_offsets = _batch_legal_ids_offsets(batch)
                actions, logp, ent = sample_actions_from_legal_ids(
                    logits,
                    legal_ids,
                    legal_offsets,
                    rng=self._rng,
                    counters=anomaly,
                    pass_action_id=pass_action_id,
                )
                legal_ids_prefix = _packed_legal_ids_prefix(legal_ids, legal_offsets)
                base = int(packed_legal_offsets[-1][-1])
                packed_legal_ids.append(legal_ids_prefix.astype(np.int32, copy=False))
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
                    rng=self._rng,
                    counters=anomaly,
                    pass_action_id=pass_action_id,
                )
                legal_mask_buf[t] = legal_mask

            next_batch = env.step(actions.astype(np.uint32, copy=False))
            reward = _batch_reward(next_batch)
            terminated = np.asarray(next_batch.terminated)
            truncated = np.asarray(next_batch.truncated)
            engine_status = np.asarray(next_batch.engine_status)

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

            done = np.logical_or(terminated, truncated)
            if np.any(done):
                reset_done = getattr(env, "reset_done", None)
                if callable(reset_done):
                    batch = reset_done(done.astype(np.bool_, copy=False))
                else:
                    batch = next_batch
            else:
                batch = next_batch

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

    @property
    def checkpoint_metadata_poll_interval_updates(self) -> int:
        """Compatibility-preserving name for checkpoint metadata polling cadence."""
        return self.reload_interval_updates

    @property
    def loaded_checkpoint_update(self) -> int:
        """Deprecated alias for observed_checkpoint_update.

        This worker only observes learner-emitted checkpoint metadata markers.
        It does not reload model parameters.
        """
        return self.observed_checkpoint_update

    @property
    def last_reload_checkpoint_update(self) -> int:
        """Deprecated alias for last_observed_checkpoint_update."""
        return self.last_observed_checkpoint_update

    @property
    def checkpoint_lag_updates(self) -> int:
        """Deprecated alias for checkpoint_metadata_lag_updates."""
        return self.checkpoint_metadata_lag_updates

    def poll_checkpoint_metadata(self) -> dict[str, int]:
        """Observe learner-emitted checkpoint metadata markers.

        This is a metadata-only surface used for lag tracking. The actor worker
        does not reload model parameters in this scaffold.
        """
        self.update_count += 1
        if self.checkpoint_dir and self.update_count % self.checkpoint_metadata_poll_interval_updates == 0:
            self._observe_checkpoint_metadata_if_available()

        latest_checkpoint_update = self._get_latest_checkpoint_metadata_update()
        self.checkpoint_metadata_lag_updates = max(0, latest_checkpoint_update - self.observed_checkpoint_update)
        return {
            "observed_checkpoint_update": self.observed_checkpoint_update,
            "checkpoint_metadata_lag_updates": self.checkpoint_metadata_lag_updates,
        }

    def poll_checkpoint_sync(self) -> dict[str, int]:
        """Deprecated metadata-only alias.

        The actor does not implement parameter reload. This method only tracks
        learner-emitted checkpoint metadata markers and preserves the legacy
        surface for callers that have not yet migrated.
        """
        metadata_status = self.poll_checkpoint_metadata()
        return {
            "loaded_checkpoint_update": metadata_status["observed_checkpoint_update"],
            "checkpoint_lag_updates": metadata_status["checkpoint_metadata_lag_updates"],
        }

    def _observe_checkpoint_metadata_if_available(self) -> None:
        if not self.checkpoint_dir:
            return

        latest_checkpoint_update = self._get_latest_checkpoint_metadata_update()
        if latest_checkpoint_update <= self.last_observed_checkpoint_update:
            return

        checkpoint_metadata_path = self._checkpoint_metadata_path_for_update(latest_checkpoint_update)
        if checkpoint_metadata_path is None:
            return

        print(f"Actor {self.actor_id} observed checkpoint metadata: {checkpoint_metadata_path}")
        self.observed_checkpoint_update = latest_checkpoint_update
        self.last_observed_checkpoint_update = latest_checkpoint_update

    def _checkpoint_metadata_path_for_update(self, update_count: int) -> Path | None:
        if not self.checkpoint_dir:
            return None

        for path in (
            self.checkpoint_dir / f"checkpoint_metadata_{update_count}.json",
            self.checkpoint_dir / f"checkpoint_{update_count}.pt",
        ):
            if path.exists():
                return path
        return None

    def _get_latest_checkpoint_metadata_update(self) -> int:
        if not self.checkpoint_dir:
            return 0

        latest_checkpoint_update = 0
        for pattern in ("checkpoint_metadata_*.json", "checkpoint_*.pt"):
            for checkpoint_path in self.checkpoint_dir.glob(pattern):
                checkpoint_update = _checkpoint_update_from_path(checkpoint_path)
                if checkpoint_update is None:
                    continue
                latest_checkpoint_update = max(latest_checkpoint_update, checkpoint_update)
        return latest_checkpoint_update


def _checkpoint_update_from_path(checkpoint_path: Path) -> int | None:
    match = _CHECKPOINT_METADATA_STEM.fullmatch(checkpoint_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def actor_behavior_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def _policy_logits(policy_logits_fn: Any, obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    if torch is not None:
        with torch.inference_mode():
            out = policy_logits_fn(obs, to_play)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        out = policy_logits_fn(obs, to_play)
    return np.asarray(out, dtype=np.float32)


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


def _batch_episode_identity(batch: Any, *, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    episode_seed = getattr(batch, "episode_seed", None)
    episode_key = getattr(batch, "episode_key", None)
    if episode_seed is None or episode_key is None:
        zeros = np.zeros((num_envs,), dtype=np.uint64)
        return zeros, zeros.copy()
    return np.asarray(episode_seed, dtype=np.uint64), np.asarray(episode_key, dtype=np.uint64)


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


def _packed_legal_ids_prefix(legal_ids: np.ndarray, legal_offsets: np.ndarray) -> np.ndarray:
    used = 0 if legal_offsets.size == 0 else int(legal_offsets[-1])
    if used < 0 or used > legal_ids.shape[0]:
        raise ValueError(f"legal_ids prefix out of bounds: used={used}, capacity={legal_ids.shape[0]}")
    return legal_ids[:used]
