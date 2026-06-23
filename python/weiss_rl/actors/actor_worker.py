"""Actor worker scaffold."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from weiss_rl.actors.action_accounting import (
    make_actor_action_accounting,
    record_actor_actions,
)
from weiss_rl.actors.action_decision import choose_actor_actions
from weiss_rl.actors.actor_opponents import current_opponent_policy_ids as _current_opponent_policy_ids
from weiss_rl.actors.actor_opponents import resample_actor_opponents
from weiss_rl.actors.actor_worker_helpers import (
    actor_behavior_logp_from_legal_ids as _actor_behavior_logp_from_legal_ids,
)
from weiss_rl.actors.actor_worker_helpers import (
    batch_counter as _batch_counter,
)
from weiss_rl.actors.actor_worker_helpers import (
    batch_episode_identity as _batch_episode_identity,
)
from weiss_rl.actors.actor_worker_helpers import (
    batch_reward as _batch_reward,
)
from weiss_rl.actors.actor_worker_helpers import (
    batch_to_play as _batch_to_play,
)
from weiss_rl.actors.actor_worker_helpers import (
    env_timeout_limits as _env_timeout_limits,
)
from weiss_rl.actors.actor_worker_helpers import (
    episode_identity_or_zeros as _episode_identity_or_zeros,
)
from weiss_rl.actors.actor_worker_helpers import (
    refresh_opponent_ids as _refresh_opponent_ids,
)
from weiss_rl.actors.actor_worker_replay import (
    actor_fault_dir_path,
    append_actor_replay_step,
    clear_actor_replay_for_env,
    ensure_actor_episode_buffers,
    flush_actor_replay_for_env,
    raise_actor_numeric_fault,
    resolve_actor_replay_episode_seed64,
    sync_actor_replay_episode_buffers,
)
from weiss_rl.actors.checkpoint_metadata import (
    latest_checkpoint_metadata_update,
    observe_new_checkpoint_metadata,
)
from weiss_rl.actors.episode_boundary import ActorEpisodeCounters, handle_actor_episode_boundaries
from weiss_rl.actors.replay_capture import (
    ReplayEpisodeBuffer,
)
from weiss_rl.actors.unroll_batch import ActorUnrollBuffers, LayoutName, UnrollBatch
from weiss_rl.core.masking import (
    MaskingAnomalyCounters,
    resolve_pass_action_id,
)
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
)

torch: ModuleType | None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _configure_actor_torch_threads(actor_torch_threads: int) -> None:
    if torch is None:
        return
    threads = int(actor_torch_threads)
    if threads < 1:
        raise ValueError("actor_torch_threads must be >= 1")
    torch.set_num_threads(threads)
    with suppress(Exception):
        torch.set_num_interop_threads(1)


def actor_behavior_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    """Compatibility wrapper for the public actor-worker masking helper."""
    return _actor_behavior_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


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
    fault_dir: Path | None = None
    reload_interval_updates: int = 1000  # Deprecated alias for checkpoint metadata polling cadence.
    opponent_sampler: Any | None = None
    opponent_assignment_fn: Any | None = None
    pass_with_nonpass_penalty: float = 0.0

    update_count: int = field(default=0, init=False)
    observed_checkpoint_update: int = field(default=0, init=False)
    last_observed_checkpoint_update: int = field(default=-1, init=False)
    checkpoint_metadata_lag_updates: int = field(default=0, init=False)
    _torch_threads_configured: bool = field(default=False, init=False)
    _rng: np.random.Generator | None = field(default=None, init=False)
    _opponent_rng: np.random.Generator | None = field(default=None, init=False)
    _current_opponent_policy_ids: np.ndarray | None = field(default=None, init=False)
    outcomes: OnlineOutcomeTracker = field(default_factory=OnlineOutcomeTracker)
    opponent_id_by_env: np.ndarray | None = field(default=None, init=False)

    # Replay capture (M5-07)
    # Note: replay bundles require run_id256 and spec_hash256 to be set by the caller.
    # If they are None, replay capture stays disabled (flush is a no-op).
    episode_index_by_env: np.ndarray | None = field(default=None, init=False)
    episode_seed64_by_env: np.ndarray | None = field(default=None, init=False)
    run_id256: bytes | None = None
    spec_hash256: bytes | None = None
    replay_dir: Path | None = None  # defaults to checkpoint_dir/../replays if None
    replay_rerun_contract: ReplayRerunContract | None = None
    env_id_base: int = 0  # offset if you shard env ids across actors
    capture_replays_on_done: bool = False  # keep False by default to avoid huge output

    _episode_buffers_by_env: list[ReplayEpisodeBuffer | None] = field(default_factory=list, init=False)

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

        if self.opponent_id_by_env is None or int(self.opponent_id_by_env.shape[0]) != N:
            self.opponent_id_by_env = np.full((N,), "unknown", dtype=object)

        if not self._torch_threads_configured:
            _configure_actor_torch_threads(self.actor_torch_threads)
            self._torch_threads_configured = True
            if torch is not None and int(torch.get_num_threads()) != int(self.actor_torch_threads):
                raise RuntimeError(
                    f"torch threads mismatch: got {torch.get_num_threads()}, want {self.actor_torch_threads}"
                )

        if self._rng is None:
            self._rng = np.random.default_rng(self.seed + self.actor_id)

        if self.episode_index_by_env is None:
            self.episode_index_by_env = np.zeros((N,), dtype=np.int64)
        if self.episode_seed64_by_env is None:
            base_seed64 = np.uint64(self.seed) ^ (np.uint64(self.actor_id) << np.uint64(32))
            self.episode_seed64_by_env = (base_seed64 + np.arange(N, dtype=np.uint64)).astype(np.uint64, copy=False)

        self._ensure_episode_buffers()

        pass_action_id = resolve_pass_action_id()
        anomaly = MaskingAnomalyCounters()

        self._resample_opponents(np.ones((N,), dtype=np.bool_))
        batch = env.reset()
        obs0 = np.asarray(batch.obs)
        if obs0.ndim != 2 or obs0.shape[0] != N:
            raise ValueError("expected batch.obs shape (N, OBS_LEN)")
        buffers = ActorUnrollBuffers.allocate(
            T=T,
            N=N,
            obs_width=int(obs0.shape[1]),
            obs_dtype=obs0.dtype,
            action_space=A,
            layout_name=self.layout_name,
        )
        timeout_limits = _env_timeout_limits(env)
        episode_counters = ActorEpisodeCounters()
        action_accounting = make_actor_action_accounting(N)

        for t in range(T):
            _refresh_opponent_ids(self.opponent_id_by_env, batch=batch, env=env, num_envs=N)
            obs = np.array(batch.obs, copy=True)
            to_play = np.array(_batch_to_play(batch), copy=True)
            decision_id = np.array(batch.decision_id, copy=True)
            batch_episode_seed, batch_episode_key = _batch_episode_identity(batch)
            if batch_episode_seed is not None:
                batch_episode_seed = np.array(batch_episode_seed, dtype=np.uint64, copy=True)
            if batch_episode_key is not None:
                batch_episode_key = np.array(batch_episode_key, dtype=np.uint64, copy=True)
            episode_seed = _episode_identity_or_zeros(batch_episode_seed, num_envs=N)
            episode_key = _episode_identity_or_zeros(batch_episode_key, num_envs=N)
            replay_episode_seed64 = np.array(
                self._resolve_replay_episode_seed64(batch_episode_seed, num_envs=N),
                dtype=np.uint64,
                copy=True,
            )
            self._sync_replay_episode_buffers(
                episode_seed64=replay_episode_seed64,
                simulator_episode_key=batch_episode_key,
            )

            if obs.shape != (N, buffers.obs.shape[2]):
                raise ValueError("batch.obs shape changed within unroll")

            decision = choose_actor_actions(
                self,
                batch=batch,
                policy_logits_fn=policy_logits_fn,
                obs=obs,
                to_play=to_play,
                decision_id=decision_id,
                episode_seed=episode_seed,
                episode_key=episode_key,
                buffers=buffers,
                t=t,
                num_envs=N,
                action_space=A,
                layout_name=self.layout_name,
                rng=self._rng,
                anomaly=anomaly,
                pass_action_id=pass_action_id,
            )
            selection = decision.selection
            actions = decision.actions
            next_batch = env.step(actions.astype(np.uint32, copy=False))
            reward = _batch_reward(next_batch)
            reward_shaped = record_actor_actions(
                accounting=action_accounting,
                layout_name=self.layout_name,
                selection=selection,
                rewards=reward,
                pass_action_id=pass_action_id,
                pass_with_nonpass_penalty=float(self.pass_with_nonpass_penalty),
                main_move_action=getattr(next_batch, "main_move_action", None),
            )
            terminated = np.asarray(next_batch.terminated)
            truncated = np.asarray(next_batch.truncated)
            engine_status = np.asarray(next_batch.engine_status)
            decision_count = _batch_counter(next_batch, "decision_count", num_envs=N)
            tick_count = _batch_counter(next_batch, "tick_count", num_envs=N)
            no_progress_count = _batch_counter(next_batch, "no_progress_count", num_envs=N)

            # Replay capture: append post-step signals using pre-step legality and identity.
            if self.layout_name == "i16_legal_ids":
                for i in range(N):
                    self._append_replay_step(
                        env_index=int(i),
                        t=int(t),
                        decision_id=int(decision_id[i]),
                        actor=int(to_play[i]),
                        action=int(actions[i]),
                        reward=float(reward[i]),
                        terminated=bool(terminated[i]),
                        truncated=bool(truncated[i]),
                        engine_status=int(engine_status[i]),
                        legal_ids=selection.replay_legal_slices[i],
                    )

            buffers.record_step(
                t=t,
                obs=obs,
                to_play=to_play,
                decision_id=decision_id,
                actions=actions,
                reward=reward_shaped,
                terminated=terminated,
                truncated=truncated,
                engine_status=engine_status,
                episode_seed=episode_seed,
                episode_key=episode_key,
                behavior_logp=selection.logp,
                entropy=selection.entropy,
            )

            done = np.logical_or(terminated, truncated)
            if np.any(done):
                done_mask = done.astype(np.bool_, copy=False)
                batch = handle_actor_episode_boundaries(
                    self,
                    env=env,
                    next_batch=next_batch,
                    done_mask=done_mask,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    engine_status=engine_status,
                    decision_count=decision_count,
                    tick_count=tick_count,
                    no_progress_count=no_progress_count,
                    timeout_limits=timeout_limits,
                    action_accounting=action_accounting,
                    episode_counters=episode_counters,
                    t=t,
                    decision_id=decision_id,
                    replay_episode_seed64=replay_episode_seed64,
                    simulator_episode_key=batch_episode_key,
                )
            else:
                batch = next_batch

        return buffers.to_batch(
            counters={
                "empty_legal": anomaly.empty_legal,
                **episode_counters.as_dict(),
                **action_accounting.counters,
            },
        )

    @property
    def current_opponent_policy_ids(self) -> tuple[str, ...]:
        return _current_opponent_policy_ids(self._current_opponent_policy_ids)

    def _resample_opponents(self, done: np.ndarray) -> None:
        state = resample_actor_opponents(
            opponent_sampler=self.opponent_sampler,
            opponent_rng=self._opponent_rng,
            seed=self.seed,
            actor_id=self.actor_id,
            num_envs=self.num_envs,
            done=done,
            current_opponent_policy_ids=self._current_opponent_policy_ids,
            opponent_id_by_env=self.opponent_id_by_env,
            opponent_assignment_fn=self.opponent_assignment_fn,
        )
        self._opponent_rng = state.opponent_rng
        self._current_opponent_policy_ids = state.current_opponent_policy_ids
        self.opponent_id_by_env = state.opponent_id_by_env

    def _fault_dir_path(self) -> Path:
        return actor_fault_dir_path(self)

    def _ensure_episode_buffers(self) -> None:
        ensure_actor_episode_buffers(self)

    def _resolve_replay_episode_seed64(self, episode_seed: np.ndarray | None, *, num_envs: int) -> np.ndarray:
        return resolve_actor_replay_episode_seed64(self, episode_seed, num_envs=num_envs)

    def _sync_replay_episode_buffers(
        self,
        *,
        episode_seed64: np.ndarray,
        simulator_episode_key: np.ndarray | None,
    ) -> None:
        sync_actor_replay_episode_buffers(
            self,
            episode_seed64=episode_seed64,
            simulator_episode_key=simulator_episode_key,
        )

    def _clear_replay_for_env(self, *, env_index: int) -> None:
        clear_actor_replay_for_env(self, env_index=env_index)

    def _append_replay_step(
        self,
        *,
        env_index: int,
        t: int,
        decision_id: int,
        actor: int,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        engine_status: int,
        legal_ids: np.ndarray,
    ) -> None:
        append_actor_replay_step(
            self,
            env_index=env_index,
            t=t,
            decision_id=decision_id,
            actor=actor,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            engine_status=engine_status,
            legal_ids=legal_ids,
        )

    def _flush_replay_for_env(self, *, env_index: int, fault_payload: dict[str, Any] | None = None) -> None:
        flush_actor_replay_for_env(self, env_index=env_index, fault_payload=fault_payload)

    def _raise_numeric_fault(
        self,
        reason: str,
        *,
        step: int,
        obs: np.ndarray,
        to_play: np.ndarray,
        decision_id: np.ndarray,
        episode_seed: np.ndarray,
        episode_key: np.ndarray,
        logits: np.ndarray,
        actions: np.ndarray | None = None,
        logp: np.ndarray | None = None,
        entropy: np.ndarray | None = None,
        legal_ids: np.ndarray | None = None,
        legal_offsets: np.ndarray | None = None,
        legal_mask: np.ndarray | None = None,
    ) -> None:
        raise_actor_numeric_fault(
            self,
            reason=reason,
            step=step,
            obs=obs,
            to_play=to_play,
            decision_id=decision_id,
            episode_seed=episode_seed,
            episode_key=episode_key,
            logits=logits,
            actions=actions,
            logp=logp,
            entropy=entropy,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_mask=legal_mask,
        )

    def poll_checkpoint_metadata(self) -> dict[str, int]:
        """Observe learner-emitted checkpoint metadata markers.

        This is a metadata-only surface used for lag tracking. The actor worker
        does not reload model parameters in this scaffold.
        """
        self.update_count += 1
        if self.checkpoint_dir and self.update_count % self.reload_interval_updates == 0:
            checkpoint_observation = observe_new_checkpoint_metadata(
                self.checkpoint_dir,
                last_observed_update=self.last_observed_checkpoint_update,
            )
            if checkpoint_observation is not None:
                print(f"Actor {self.actor_id} observed checkpoint metadata: {checkpoint_observation.path}")
                self.observed_checkpoint_update = checkpoint_observation.update_count
                self.last_observed_checkpoint_update = checkpoint_observation.update_count

        latest_checkpoint_update = latest_checkpoint_metadata_update(self.checkpoint_dir)
        self.checkpoint_metadata_lag_updates = max(0, latest_checkpoint_update - self.observed_checkpoint_update)
        return {
            "observed_checkpoint_update": self.observed_checkpoint_update,
            "checkpoint_metadata_lag_updates": self.checkpoint_metadata_lag_updates,
        }
