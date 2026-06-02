"""Actor worker scaffold."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np

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
    batch_legal_ids_offsets as _batch_legal_ids_offsets,
)
from weiss_rl.actors.actor_worker_helpers import (
    batch_legal_mask as _batch_legal_mask,
)
from weiss_rl.actors.actor_worker_helpers import (
    batch_reward as _batch_reward,
)
from weiss_rl.actors.actor_worker_helpers import (
    batch_to_play as _batch_to_play,
)
from weiss_rl.actors.actor_worker_helpers import (
    checkpoint_update_from_path as _checkpoint_update_from_path,
)
from weiss_rl.actors.actor_worker_helpers import (
    env_timeout_limits as _env_timeout_limits,
)
from weiss_rl.actors.actor_worker_helpers import (
    episode_identity_or_zeros as _episode_identity_or_zeros,
)
from weiss_rl.actors.actor_worker_helpers import (
    packed_legal_ids_prefix as _packed_legal_ids_prefix,
)
from weiss_rl.actors.actor_worker_helpers import (
    policy_logits as _policy_logits,
)
from weiss_rl.actors.actor_worker_helpers import (
    refresh_opponent_ids as _refresh_opponent_ids,
)
from weiss_rl.actors.actor_worker_helpers import (
    sampled_opponent_policy_ids as _sampled_opponent_policy_ids,
)
from weiss_rl.actors.actor_worker_helpers import (
    update_outcomes as _update_outcomes,
)
from weiss_rl.core.masking import (
    MaskingAnomalyCounters,
    resolve_pass_action_id,
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
)
from weiss_rl.core.termination_reason import classify_episode_end_reason
from weiss_rl.diagnostics.action_diagnostics import (
    make_action_sequence_state,
    reset_action_sequence_state,
    update_action_summary_from_ids,
    update_action_summary_from_mask,
)
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    write_fault_bundle,
    write_replay_bundle,
)
from weiss_rl.runtime.components.reward_shaping import apply_pass_with_nonpass_penalty as _apply_pass_penalty

torch: ModuleType | None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


LayoutName = Literal["i16_legal_ids", "mask"]


def _nonfinite_indices(values: np.ndarray) -> np.ndarray:
    return np.argwhere(~np.isfinite(values)).astype(np.int64, copy=False)


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
class ReplayEpisodeBuffer:
    actor_episode_index: int
    episode_seed64: int
    simulator_episode_key: int | bytes | None
    steps: list[ReplayStep] = field(default_factory=list)


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

        self._resample_opponents(np.ones((N,), dtype=np.bool_))
        batch = env.reset()
        obs0 = np.asarray(batch.obs)
        if obs0.ndim != 2 or obs0.shape[0] != N:
            raise ValueError("expected batch.obs shape (N, OBS_LEN)")
        obs_buf = np.empty((T, N, obs0.shape[1]), dtype=obs0.dtype)
        timeout_limits = _env_timeout_limits(env)
        no_progress_timeout_rows = 0
        natural_timeout_rows = 0
        decision_limit_timeout_rows = 0
        tick_limit_timeout_rows = 0
        timeout_unknown_rows = 0
        engine_fault_done_rows = 0
        action_counters = {
            "total_actions": 0,
            "pass_actions": 0,
            "main_move_actions": 0,
            "pass_with_nonpass_available": 0,
            "pass_with_nonpass_penalty_count": 0,
            "pass_with_nonpass_penalty_total_micros": 0,
            "max_consecutive_main_moves": 0,
        }
        action_sequence_state = make_action_sequence_state(N)

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

            if obs.shape != (N, obs_buf.shape[2]):
                raise ValueError("batch.obs shape changed within unroll")

            logits = _policy_logits(policy_logits_fn, obs, to_play)
            if logits.shape != (N, A):
                raise ValueError(f"policy_logits_fn must return shape (N, A)=({N}, {A})")
            if not np.all(np.isfinite(logits)):
                self._raise_numeric_fault(
                    "non-finite actor policy logits",
                    step=t,
                    obs=obs,
                    to_play=to_play,
                    decision_id=decision_id,
                    episode_seed=episode_seed,
                    episode_key=episode_key,
                    logits=logits,
                )

            if self.layout_name == "i16_legal_ids":
                legal_ids, legal_offsets = _batch_legal_ids_offsets(batch)
                legal_ids_array = np.array(legal_ids, copy=True)
                legal_offsets_array = np.array(legal_offsets, copy=True)
                packed_legal_ids_array = np.array(legal_ids_array, dtype=np.int64, copy=True)
                packed_legal_offsets_array = np.array(legal_offsets_array, dtype=np.int64, copy=True)
                actions, logp, ent = sample_actions_from_legal_ids(
                    logits,
                    legal_ids_array,
                    legal_offsets_array,
                    rng=self._rng,
                    counters=anomaly,
                    pass_action_id=pass_action_id,
                )
                # Keep per-env legal slices for replay fingerprinting (pre-step legality).
                legal_slices: list[np.ndarray] = []
                for i in range(N):
                    start = int(legal_offsets_array[i])
                    end = int(legal_offsets_array[i + 1])
                    legal_slices.append(np.array(legal_ids_array[start:end], dtype=np.uint16, copy=True))

                legal_ids_prefix = _packed_legal_ids_prefix(legal_ids_array, legal_offsets_array)
                offset_base = int(packed_legal_offsets[-1][-1])
                packed_legal_ids.append(np.array(legal_ids_prefix, dtype=np.int32, copy=True))
                packed_legal_offsets.append(np.array(legal_offsets_array[1:] + offset_base, dtype=np.uint32, copy=True))

                if not np.all(np.isfinite(logp)):
                    self._raise_numeric_fault(
                        "non-finite actor sampled logp",
                        step=t,
                        obs=obs,
                        to_play=to_play,
                        decision_id=decision_id,
                        episode_seed=episode_seed,
                        episode_key=episode_key,
                        logits=logits,
                        actions=actions,
                        logp=logp,
                        entropy=ent,
                        legal_ids=legal_ids_array,
                        legal_offsets=legal_offsets_array,
                    )
                if not np.all(np.isfinite(ent)):
                    self._raise_numeric_fault(
                        "non-finite actor sampled entropy",
                        step=t,
                        obs=obs,
                        to_play=to_play,
                        decision_id=decision_id,
                        episode_seed=episode_seed,
                        episode_key=episode_key,
                        logits=logits,
                        actions=actions,
                        logp=logp,
                        entropy=ent,
                        legal_ids=legal_ids_array,
                        legal_offsets=legal_offsets_array,
                    )
            else:
                legal_mask = _batch_legal_mask(batch)
                if legal_mask.shape != (N, A):
                    raise ValueError(f"expected legal_mask shape (N, A)=({N}, {A})")
                legal_mask_array = np.array(legal_mask, copy=True)
                if legal_mask_buf is None:
                    legal_mask_buf = np.empty((T, N, A), dtype=legal_mask_array.dtype)
                actions, logp, ent = sample_actions_from_mask(
                    logits,
                    legal_mask_array,
                    rng=self._rng,
                    counters=anomaly,
                    pass_action_id=pass_action_id,
                )
                legal_mask_buf[t] = legal_mask_array

                if not np.all(np.isfinite(logp)):
                    self._raise_numeric_fault(
                        "non-finite actor sampled logp",
                        step=t,
                        obs=obs,
                        to_play=to_play,
                        decision_id=decision_id,
                        episode_seed=episode_seed,
                        episode_key=episode_key,
                        logits=logits,
                        actions=actions,
                        logp=logp,
                        entropy=ent,
                        legal_mask=legal_mask_array,
                    )
                if not np.all(np.isfinite(ent)):
                    self._raise_numeric_fault(
                        "non-finite actor sampled entropy",
                        step=t,
                        obs=obs,
                        to_play=to_play,
                        decision_id=decision_id,
                        episode_seed=episode_seed,
                        episode_key=episode_key,
                        logits=logits,
                        actions=actions,
                        logp=logp,
                        entropy=ent,
                        legal_mask=legal_mask_array,
                    )

            next_batch = env.step(actions.astype(np.uint32, copy=False))
            if self.layout_name == "i16_legal_ids":
                update_action_summary_from_ids(
                    counters=action_counters,
                    state=action_sequence_state,
                    actions=actions,
                    legal_ids=packed_legal_ids_array,
                    legal_offsets=packed_legal_offsets_array,
                    pass_action_id=pass_action_id,
                    main_move_action=getattr(next_batch, "main_move_action", None),
                )
            else:
                update_action_summary_from_mask(
                    counters=action_counters,
                    state=action_sequence_state,
                    actions=actions,
                    legal_mask=legal_mask_array,
                    pass_action_id=pass_action_id,
                    main_move_action=getattr(next_batch, "main_move_action", None),
                )
            reward = _batch_reward(next_batch)
            if self.layout_name == "i16_legal_ids":
                reward_shaped, penalty_count, penalty_total_micros = _apply_pass_penalty(
                    reward,
                    actions,
                    pass_action_id=pass_action_id,
                    penalty=float(self.pass_with_nonpass_penalty),
                    legal_ids=packed_legal_ids_array,
                    legal_offsets=packed_legal_offsets_array,
                )
            else:
                reward_shaped, penalty_count, penalty_total_micros = _apply_pass_penalty(
                    reward,
                    actions,
                    pass_action_id=pass_action_id,
                    penalty=float(self.pass_with_nonpass_penalty),
                    legal_mask=legal_mask_array,
                )
            action_counters["pass_with_nonpass_penalty_count"] += penalty_count
            action_counters["pass_with_nonpass_penalty_total_micros"] += penalty_total_micros
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
                        legal_ids=legal_slices[i],
                    )

            obs_buf[t] = obs
            to_play_buf[t] = to_play.astype(np.int8, copy=False)
            decision_id_buf[t] = decision_id.astype(np.int32, copy=False)
            action_buf[t] = actions.astype(np.uint32, copy=False)
            reward_buf[t] = reward_shaped.astype(np.float32, copy=False)
            terminated_buf[t] = terminated.astype(np.bool_, copy=False)
            truncated_buf[t] = truncated.astype(np.bool_, copy=False)
            engine_status_buf[t] = engine_status.astype(np.int32, copy=False)
            episode_seed_buf[t] = episode_seed
            episode_key_buf[t] = episode_key
            behavior_logp_buf[t] = logp.astype(np.float32, copy=False)
            entropy_buf[t] = ent.astype(np.float32, copy=False)

            done = np.logical_or(terminated, truncated)
            if np.any(done):
                done_mask = done.astype(np.bool_, copy=False)

                _update_outcomes(
                    self.outcomes,
                    opponent_ids=self.opponent_id_by_env,
                    reward=reward,
                    engine_status=engine_status,
                    done=done_mask,
                )

                for env_index in np.flatnonzero(done_mask):
                    env_index_int = int(env_index)
                    termination_reason = classify_episode_end_reason(
                        terminated=bool(terminated[env_index_int]),
                        truncated=bool(truncated[env_index_int]),
                        engine_status=int(engine_status[env_index_int]),
                        decision_count=int(decision_count[env_index_int]),
                        tick_count=int(tick_count[env_index_int]),
                        no_progress_count=int(no_progress_count[env_index_int]),
                        max_decisions=timeout_limits["max_decisions"],
                        max_ticks=timeout_limits["max_ticks"],
                        max_no_progress_decisions=timeout_limits["max_no_progress_decisions"],
                    )
                    if termination_reason == "engine_fault":
                        engine_fault_done_rows += 1
                    elif termination_reason == "no_progress_timeout":
                        no_progress_timeout_rows += 1
                    elif termination_reason == "decision_limit_timeout":
                        natural_timeout_rows += 1
                        decision_limit_timeout_rows += 1
                    elif termination_reason == "tick_limit_timeout":
                        natural_timeout_rows += 1
                        tick_limit_timeout_rows += 1
                    elif termination_reason == "timeout_unknown":
                        natural_timeout_rows += 1
                        timeout_unknown_rows += 1
                    env_fault_payload = None
                    if int(engine_status[env_index_int]) != 0:
                        env_fault_payload = {
                            "format": "engine_fault_replay",
                            "actor_id": self.actor_id,
                            "env_id": int(self.env_id_base + env_index_int),
                            "t": int(t),
                            "decision_id": int(decision_id[env_index_int]),
                            "engine_status": int(engine_status[env_index_int]),
                            "terminated": bool(terminated[env_index_int]),
                            "truncated": bool(truncated[env_index_int]),
                            "episode_seed64": int(replay_episode_seed64[env_index_int]),
                            "simulator_episode_key": (
                                None if batch_episode_key is None else int(batch_episode_key[env_index_int])
                            ),
                        }
                    if self.capture_replays_on_done or env_fault_payload is not None:
                        self._flush_replay_for_env(env_index=env_index_int, fault_payload=env_fault_payload)
                    else:
                        self._clear_replay_for_env(env_index=env_index_int)

                # Advance actor-local episode counters for rows that finished.
                if self.episode_index_by_env is not None:
                    self.episode_index_by_env[done] += 1
                if self.episode_seed64_by_env is not None:
                    self.episode_seed64_by_env[done] += np.uint64(1)

                reset_done = getattr(env, "reset_done", None)
                if callable(reset_done):
                    self._resample_opponents(done_mask)
                    reset_action_sequence_state(action_sequence_state, done_mask)
                    batch = reset_done(done_mask)
                else:
                    reset_action_sequence_state(action_sequence_state, done_mask)
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
            counters={
                "empty_legal": anomaly.empty_legal,
                "engine_fault_done_rows": engine_fault_done_rows,
                "no_progress_timeout_rows": no_progress_timeout_rows,
                "natural_timeout_rows": natural_timeout_rows,
                "decision_limit_timeout_rows": decision_limit_timeout_rows,
                "tick_limit_timeout_rows": tick_limit_timeout_rows,
                "timeout_unknown_rows": timeout_unknown_rows,
                **action_counters,
            },
        )

    @property
    def current_opponent_policy_ids(self) -> tuple[str, ...]:
        if self._current_opponent_policy_ids is None:
            return ()
        return tuple(str(policy_id) for policy_id in self._current_opponent_policy_ids.tolist())

    def _resample_opponents(self, done: np.ndarray) -> None:
        if self.opponent_sampler is None:
            return
        if self._opponent_rng is None:
            self._opponent_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.actor_id, 1]))

        done_array = np.asarray(done, dtype=np.bool_)
        if done_array.shape != (self.num_envs,):
            raise ValueError(f"done must have shape ({self.num_envs},)")

        sample_count = int(np.count_nonzero(done_array))
        if sample_count == 0:
            return

        if self._current_opponent_policy_ids is None:
            self._current_opponent_policy_ids = np.empty((self.num_envs,), dtype=object)
        if self.opponent_id_by_env is None or int(self.opponent_id_by_env.shape[0]) != self.num_envs:
            self.opponent_id_by_env = np.full((self.num_envs,), "unknown", dtype=object)

        sampled_policy_ids = _sampled_opponent_policy_ids(
            self.opponent_sampler,
            count=sample_count,
            rng=self._opponent_rng,
        )
        self._current_opponent_policy_ids[done_array] = sampled_policy_ids
        self.opponent_id_by_env[done_array] = sampled_policy_ids
        if self.opponent_assignment_fn is not None:
            self.opponent_assignment_fn(done_array.copy(), self.current_opponent_policy_ids)

    def _fault_dir_path(self) -> Path:
        if self.fault_dir is not None:
            return self.fault_dir
        if self.checkpoint_dir is not None:
            return self.checkpoint_dir / "faults"
        return Path("faults")

    def _replay_dir_path(self) -> Path:
        if self.replay_dir is not None:
            return self.replay_dir
        if self.checkpoint_dir is not None:
            # runs/.../training/checkpoints -> runs/.../replays/regression
            return self.checkpoint_dir.parent.parent / "replays" / "regression"
        return Path("replays") / "regression"

    def _ensure_episode_buffers(self) -> None:
        if not self._episode_buffers_by_env:
            self._episode_buffers_by_env = [None for _ in range(self.num_envs)]

    def _resolve_replay_episode_seed64(self, episode_seed: np.ndarray | None, *, num_envs: int) -> np.ndarray:
        if episode_seed is not None:
            return episode_seed
        if self.episode_seed64_by_env is None:
            base_seed64 = np.uint64(self.seed) ^ (np.uint64(self.actor_id) << np.uint64(32))
            self.episode_seed64_by_env = (base_seed64 + np.arange(num_envs, dtype=np.uint64)).astype(
                np.uint64, copy=False
            )
        return self.episode_seed64_by_env

    def _sync_replay_episode_buffers(
        self,
        *,
        episode_seed64: np.ndarray,
        simulator_episode_key: np.ndarray | None,
    ) -> None:
        self._ensure_episode_buffers()
        if self.episode_index_by_env is None:
            self.episode_index_by_env = np.zeros((self.num_envs,), dtype=np.int64)
        for env_index in range(self.num_envs):
            next_seed = int(episode_seed64[env_index])
            next_key = None if simulator_episode_key is None else int(simulator_episode_key[env_index])
            current = self._episode_buffers_by_env[env_index]
            if current is not None:
                same_seed = int(current.episode_seed64) == next_seed
                same_key = current.simulator_episode_key == next_key
                if same_seed and same_key:
                    continue
            self._episode_buffers_by_env[env_index] = ReplayEpisodeBuffer(
                actor_episode_index=int(self.episode_index_by_env[env_index]),
                episode_seed64=next_seed,
                simulator_episode_key=next_key,
            )

    def _clear_replay_for_env(self, *, env_index: int) -> None:
        if not self._episode_buffers_by_env:
            return
        self._episode_buffers_by_env[env_index] = None

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
        if self.spec_hash256 is None:
            return
        self._ensure_episode_buffers()
        buffer = self._episode_buffers_by_env[env_index]
        if buffer is None:
            return
        fp = compute_legal_fingerprint64(
            spec_hash256=self.spec_hash256,
            decision_id=int(decision_id),
            legal_ids=legal_ids,
        )
        buffer.steps.append(
            ReplayStep(
                t=int(t),
                decision_id=int(decision_id),
                actor=int(actor),
                action=int(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                engine_status=int(engine_status),
                legal_fingerprint64=int(fp),
            )
        )

    def _flush_replay_for_env(self, *, env_index: int, fault_payload: dict[str, Any] | None = None) -> None:
        # Cannot write deterministic replay bundle without stable IDs.
        if self.run_id256 is None or self.spec_hash256 is None:
            return
        if self.episode_index_by_env is None or not self._episode_buffers_by_env:
            return

        buffer = self._episode_buffers_by_env[env_index]
        if buffer is None or not buffer.steps:
            self._clear_replay_for_env(env_index=env_index)
            return

        meta = make_replay_bundle_meta(
            simulator_episode_key=buffer.simulator_episode_key,
            run_id256=self.run_id256,
            spec_hash256=self.spec_hash256,
            actor_id=int(self.actor_id),
            env_id=int(self.env_id_base + env_index),
            episode_index=int(buffer.actor_episode_index),
            episode_seed64=int(buffer.episode_seed64),
            rerun_contract=self.replay_rerun_contract,
        )
        write_replay_bundle(
            out_dir=self._replay_dir_path(),
            meta=meta,
            steps=buffer.steps,
            fault_payload=fault_payload,
        )
        self._clear_replay_for_env(env_index=env_index)

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
        payload: dict[str, Any] = {
            "format": "numeric_fault_bundle",
            "component": "actor_worker",
            "reason": reason,
            "actor_id": self.actor_id,
            "layout_name": self.layout_name,
            "update_count": self.update_count,
            "observed_checkpoint_update": self.observed_checkpoint_update,
            "step": step,
            "obs": obs,
            "to_play": to_play,
            "decision_id": decision_id,
            "episode_seed": episode_seed,
            "episode_key": episode_key,
            "logits": logits,
            "logits_nonfinite_indices": _nonfinite_indices(logits),
        }
        if actions is not None:
            payload["actions"] = actions
        if logp is not None:
            payload["logp"] = logp
            payload["logp_nonfinite_indices"] = _nonfinite_indices(logp)
        if entropy is not None:
            payload["entropy"] = entropy
            payload["entropy_nonfinite_indices"] = _nonfinite_indices(entropy)
        if legal_ids is not None:
            payload["legal_ids"] = legal_ids
        if legal_offsets is not None:
            payload["legal_offsets"] = legal_offsets
        if legal_mask is not None:
            payload["legal_mask"] = legal_mask
        fault_path = write_fault_bundle(
            fault_dir=self._fault_dir_path(),
            prefix="actor_numeric_fault",
            payload=payload,
        )
        # Best-effort replay capture (uses per-env buffered steps).
        try:
            self._ensure_episode_buffers()
            for env_index in range(self.num_envs):
                self._flush_replay_for_env(env_index=env_index, fault_payload=payload)
        except Exception:
            pass
        raise RuntimeError(f"{reason}; wrote fault bundle to {fault_path}")

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
