"""Queue-based single-process runtime for deterministic and throughput-aware training."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import make_env_pool_from_config
from weiss_rl.eval.harness import game_result_from_step
from weiss_rl.league.opponent_pool import OpponentPoolSampler, sample_opponent_snapshot_ids
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.masking import (
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
)
from weiss_rl.model import PolicyValueModel

QueueRuntimeMode = Literal["train_ordered", "train_async_fast"]
_MIRROR_OPPONENT_POLICY_ID = "latest_policy_mirror"
_FIXED_OPPONENT_EXCLUSIONS = frozenset({"b1_noleague_baseline"})


@dataclass(frozen=True, slots=True)
class QueueRuntimeConfig:
    mode: QueueRuntimeMode
    actor_count: int
    envs_per_actor: int
    unroll_length: int
    batch_unrolls_per_update: int
    queue_capacity_unrolls: int
    profile: str
    base_seed: int
    pass_action_id: int
    actor_reload_interval_updates: int

    @property
    def total_envs(self) -> int:
        return int(self.actor_count * self.envs_per_actor)


@dataclass(frozen=True, slots=True)
class RuntimeUnroll:
    actor_id: int
    unroll_seq: int
    behavior_policy_version: int
    unroll_hash: str
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    logits: np.ndarray
    values: np.ndarray
    legal_actions: LegalActionBatch
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray
    initial_hidden_state: np.ndarray
    final_hidden_state: np.ndarray
    episode_seed: np.ndarray
    policy_train_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class RuntimeBatch:
    learner_batch: dict[str, Any]
    runtime_metrics: dict[str, float]


@dataclass(slots=True)
class _ActorState:
    actor_id: int
    env: DecisionBoundaryEnv
    model: PolicyValueModel
    rng: np.random.Generator
    seat_hidden: torch.Tensor
    current_batch: DecisionBoundaryBatch
    layout_name: str
    focal_seat_by_env: np.ndarray
    opponent_policy_id_by_env: np.ndarray
    opponent_hidden: torch.Tensor
    snapshot_version: int = 0
    next_unroll_seq: int = 0


class PerformanceLogger:
    """Write runtime performance records as JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class QueueRuntime:
    """Single-process actor queue runtime with deterministic ordered mode."""

    def __init__(
        self,
        *,
        stack: StackConfig,
        config: QueueRuntimeConfig,
        model: PolicyValueModel,
        observation_dim: int,
        action_dim: int,
        run_dir: Path | None = None,
        performance_log_path: Path | None = None,
    ) -> None:
        if config.actor_count < 1:
            raise ValueError("actor_count must be >= 1")
        if config.envs_per_actor < 1:
            raise ValueError("envs_per_actor must be >= 1")
        if config.batch_unrolls_per_update < 1:
            raise ValueError("batch_unrolls_per_update must be >= 1")
        if config.queue_capacity_unrolls < config.batch_unrolls_per_update:
            raise ValueError("queue_capacity_unrolls must be >= batch_unrolls_per_update")
        if config.mode == "train_ordered" and config.batch_unrolls_per_update < config.actor_count:
            raise ValueError("train_ordered requires batch_unrolls_per_update >= actor_count")

        self.stack = stack
        self.config = config
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self._device = torch.device("cpu")
        self._run_dir = None if run_dir is None else Path(run_dir)
        self._artifact_layout = None if self._run_dir is None else ArtifactLayout.from_run_dir(self._run_dir)
        training_config = stack.config.training_family_a
        self._training_mode = "" if training_config is None else str(training_config.mode).strip()
        self._league_config = stack.config.league
        self._league_enabled = bool(
            self._artifact_layout is not None
            and self._league_config is not None
            and self._league_config.enabled
            and self._training_mode != "b1_no_league"
        )
        self._registry_path = (
            None if self._artifact_layout is None else self._artifact_layout.training_snapshots_dir / REGISTRY_FILENAME
        )
        self._opponent_models: dict[str, PolicyValueModel] = {}
        self._opponent_sampler: OpponentPoolSampler | None = None
        self._opponent_candidate_ids: tuple[str, ...] = ()
        self._outcomes = OnlineOutcomeTracker(
            window_size=(50_000 if self._league_config is None else int(self._league_config.pfsp_window_episodes))
        )
        self._current_learner_update = 0
        self._pfsp_pool_size = 0
        self._pfsp_last_sampled_envs = 0
        self._pfsp_last_mirror_envs = 0
        self._actors = [
            self._build_actor_state(model=model, actor_id=actor_id) for actor_id in range(int(config.actor_count))
        ]
        self._pending_unrolls: deque[RuntimeUnroll] = deque()
        self._next_actor_index = 0
        self._last_published_snapshot_version = 0
        self._performance_logger = None if performance_log_path is None else PerformanceLogger(performance_log_path)
        self._runtime_start = time.time()
        self.refresh_opponent_pool()

    def close(self) -> None:
        for actor in self._actors:
            actor.env.close()

    def maybe_publish_snapshot(self, *, learner_model: PolicyValueModel, learner_update_count: int) -> dict[str, float]:
        self._current_learner_update = int(learner_update_count)
        if learner_update_count <= 0:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}
        if learner_update_count == self._last_published_snapshot_version:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}
        if learner_update_count % int(self.config.actor_reload_interval_updates) != 0:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}

        publish_started = time.perf_counter()
        state_dict = {key: value.detach().cpu().clone() for key, value in learner_model.state_dict().items()}
        publish_latency_ms = (time.perf_counter() - publish_started) * 1000.0

        apply_started = time.perf_counter()
        for actor in self._actors:
            actor.model.load_state_dict(state_dict)
            actor.model.eval()
            actor.snapshot_version = int(learner_update_count)
        apply_latency_ms = (time.perf_counter() - apply_started) * 1000.0
        self._last_published_snapshot_version = int(learner_update_count)
        return {
            "snapshot_publish_latency_ms": publish_latency_ms,
            "snapshot_apply_latency_ms": apply_latency_ms,
        }

    def refresh_opponent_pool(self) -> None:
        if not self._league_enabled or self._registry_path is None or not self._registry_path.is_file():
            self._opponent_sampler = None
            self._opponent_candidate_ids = ()
            self._opponent_models = {}
            self._pfsp_pool_size = 0
            return
        assert self._league_config is not None
        registry = SnapshotRegistry.load(self._registry_path)
        sampler = OpponentPoolSampler(
            registry=registry,
            recent_size=int(self._league_config.snapshot_pool_recent_size),
            champion_size=int(self._league_config.snapshot_pool_champion_size),
            power=float(self._league_config.pfsp_power),
            eps_uniform=float(self._league_config.pfsp_epsilon_uniform),
        )
        self._opponent_sampler = sampler
        candidate_ids = tuple(
            policy_id for policy_id in sampler.snapshot_ids() if policy_id not in _FIXED_OPPONENT_EXCLUSIONS
        )
        self._opponent_candidate_ids = candidate_ids
        self._pfsp_pool_size = len(candidate_ids)
        models: dict[str, PolicyValueModel] = {}
        snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
        for policy_id in candidate_ids:
            snapshot = snapshots_by_id.get(policy_id)
            if snapshot is None:
                continue
            models[policy_id] = self._load_snapshot_model(snapshot.path)
        self._opponent_models = models

    def collect_update_batch(
        self,
        *,
        gamma: float,
        truncation_bootstrap_value: bool,
        vtrace_rho_bar: float,
        vtrace_c_bar: float,
    ) -> RuntimeBatch:
        occupancy_samples: list[float] = []
        while len(self._pending_unrolls) < int(self.config.batch_unrolls_per_update):
            occupancy_samples.append(len(self._pending_unrolls) / float(self.config.queue_capacity_unrolls))
            actor = self._actors[self._next_actor_index]
            self._next_actor_index = (self._next_actor_index + 1) % len(self._actors)
            self._pending_unrolls.append(self._collect_actor_unroll(actor))

        selected = self._select_pending_unrolls()
        selected_keys = {(item.actor_id, item.unroll_seq) for item in selected}
        self._pending_unrolls = deque(
            item for item in self._pending_unrolls if (item.actor_id, item.unroll_seq) not in selected_keys
        )

        learner_batch = self._build_learner_batch(
            selected,
            gamma=gamma,
            truncation_bootstrap_value=truncation_bootstrap_value,
            vtrace_rho_bar=vtrace_rho_bar,
            vtrace_c_bar=vtrace_c_bar,
        )
        runtime_metrics = self._runtime_metrics(selected, occupancy_samples=occupancy_samples)
        if self._performance_logger is not None:
            elapsed = time.time() - self._runtime_start
            self._performance_logger.log(
                {
                    "kind": "runtime_performance_v1",
                    "wall_clock_seconds": elapsed,
                    **runtime_metrics,
                }
            )
        return RuntimeBatch(learner_batch=learner_batch, runtime_metrics=runtime_metrics)

    def _select_pending_unrolls(self) -> list[RuntimeUnroll]:
        batch_size = int(self.config.batch_unrolls_per_update)
        if self.config.mode != "train_ordered":
            return list(self._pending_unrolls)[:batch_size]
        ordered = sorted(
            self._pending_unrolls,
            key=lambda item: (item.behavior_policy_version, item.unroll_seq, item.actor_id),
        )
        if not ordered:
            raise RuntimeError("train_ordered selection requires at least one pending unroll")
        oldest_version = int(ordered[0].behavior_policy_version)
        selected: list[RuntimeUnroll] = []
        current_group: list[RuntimeUnroll] = []
        current_seq: int | None = None
        for item in ordered:
            if int(item.behavior_policy_version) != oldest_version:
                break
            if current_seq is None or int(item.unroll_seq) == current_seq:
                current_group.append(item)
                current_seq = int(item.unroll_seq)
                continue
            if len(selected) + len(current_group) > batch_size:
                break
            selected.extend(current_group)
            current_group = [item]
            current_seq = int(item.unroll_seq)
        if current_group and len(selected) + len(current_group) <= batch_size:
            selected.extend(current_group)
        if not selected:
            raise RuntimeError("train_ordered selection could not produce a same-version batch")
        return selected

    def _build_actor_state(self, *, model: PolicyValueModel, actor_id: int) -> _ActorState:
        env, layout_name = self._build_env(seed=_actor_seed(self.config.base_seed, actor_id))
        actor_model = copy.deepcopy(model).to(self._device)
        actor_model.eval()
        current_batch = env.reset(seed=_actor_seed(self.config.base_seed, actor_id))
        state = _ActorState(
            actor_id=actor_id,
            env=env,
            model=actor_model,
            rng=np.random.default_rng(_actor_seed(self.config.base_seed, actor_id)),
            seat_hidden=actor_model.initial_seat_hidden(self.config.envs_per_actor, device=self._device),
            current_batch=current_batch,
            layout_name=layout_name,
            focal_seat_by_env=np.zeros((int(self.config.envs_per_actor),), dtype=np.int64),
            opponent_policy_id_by_env=np.full(
                (int(self.config.envs_per_actor),),
                _MIRROR_OPPONENT_POLICY_ID,
                dtype=object,
            ),
            opponent_hidden=actor_model.initial_seat_hidden(self.config.envs_per_actor, device=self._device),
        )
        self._assign_episode_roles(state, np.ones((int(self.config.envs_per_actor),), dtype=np.bool_), initial=True)
        return state

    def _build_env(self, *, seed: int) -> tuple[DecisionBoundaryEnv, str]:
        environment_config = self.stack.config.environment
        if environment_config is None:
            raise RuntimeError("stack config is missing environment config")
        pool, layout_name = make_env_pool_from_config(
            {
                "max_decisions": int(environment_config.max_decisions),
                "max_ticks": int(environment_config.max_ticks),
                "observation_visibility": environment_config.observation_visibility,
                "seed": int(seed),
            },
            profile=self.config.profile,  # type: ignore[arg-type]
            num_envs=int(self.config.envs_per_actor),
        )
        legality = "ids_offsets" if layout_name == "i16_legal_ids" else "mask"
        env = DecisionBoundaryEnv(
            pool,
            legality=legality,  # type: ignore[arg-type]
            pass_action_id=int(self.config.pass_action_id),
            engine_status_policy="hard_fail",
        )
        return env, str(layout_name)

    def _load_snapshot_model(self, snapshot_path: str) -> PolicyValueModel:
        if self._run_dir is None:
            raise RuntimeError("QueueRuntime cannot load opponent snapshots without a canonical run_dir")
        payload = torch.load(self._run_dir / snapshot_path, map_location="cpu", weights_only=True)
        model_state_dict = payload.get("model_state_dict")
        if not isinstance(model_state_dict, dict):
            raise RuntimeError(f"snapshot weights payload missing model_state_dict: {snapshot_path}")
        model_config = self.stack.config.model
        if model_config is None:
            raise RuntimeError("stack config is missing model config")
        model = PolicyValueModel(
            observation_dim=self.observation_dim,
            config=model_config,
            action_dim=self.action_dim,
        ).to(self._device)
        model.load_state_dict(model_state_dict)
        model.eval()
        return model

    def _assign_episode_roles(self, actor: _ActorState, done: np.ndarray, *, initial: bool = False) -> None:
        done_array = np.asarray(done, dtype=np.bool_)
        if done_array.shape != actor.focal_seat_by_env.shape:
            raise ValueError(f"done must have shape {actor.focal_seat_by_env.shape}, got {done_array.shape}")
        if not np.any(done_array):
            return
        if initial:
            actor.focal_seat_by_env[done_array] = (actor.actor_id + np.flatnonzero(done_array)) % 2
        else:
            actor.focal_seat_by_env[done_array] = 1 - actor.focal_seat_by_env[done_array]

        sampled_policy_ids = self._sample_opponent_policy_ids(count=int(np.count_nonzero(done_array)), rng=actor.rng)
        actor.opponent_policy_id_by_env[done_array] = np.asarray(sampled_policy_ids, dtype=object)

    def _sample_opponent_policy_ids(self, *, count: int, rng: np.random.Generator) -> tuple[str, ...]:
        if count <= 0:
            return ()
        if not self._league_enabled or not self._pfsp_sampling_ready():
            self._pfsp_last_sampled_envs = 0
            self._pfsp_last_mirror_envs = count
            return tuple(_MIRROR_OPPONENT_POLICY_ID for _ in range(count))
        assert self._league_config is not None
        sampled_policy_ids = sample_opponent_snapshot_ids(
            self._opponent_candidate_ids,
            count=count,
            rng=rng,
            win_rates_by_snapshot_id={
                policy_id: self._outcomes.win_rate(policy_id) for policy_id in self._opponent_candidate_ids
            },
            power=float(self._league_config.pfsp_power),
            eps_uniform=float(self._league_config.pfsp_epsilon_uniform),
        )
        self._pfsp_last_sampled_envs = count
        self._pfsp_last_mirror_envs = 0
        return tuple(str(policy_id) for policy_id in sampled_policy_ids)

    def _pfsp_sampling_ready(self) -> bool:
        if not self._league_enabled or self._league_config is None or self._opponent_sampler is None:
            return False
        if self._current_learner_update < int(self._league_config.warmup.first_updates):
            return False
        return bool(self._opponent_candidate_ids) and bool(self._opponent_models)

    def _fill_policy_outputs_mask(
        self,
        *,
        actor: _ActorState,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        focal_rows: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray,
        values_out: np.ndarray,
        actions_out: np.ndarray,
        logp_out: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        focal_indices = np.flatnonzero(focal_rows)
        if focal_indices.size:
            self._apply_policy_rows_mask(
                model=actor.model,
                hidden_state=actor.seat_hidden,
                row_indices=focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
            )
        opponent_indices = np.flatnonzero(~focal_rows)
        if opponent_indices.size:
            self._apply_opponent_rows_mask(
                actor=actor,
                row_indices=opponent_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
            )

    def _fill_policy_outputs_ids(
        self,
        *,
        actor: _ActorState,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        focal_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        logits_out: np.ndarray,
        values_out: np.ndarray,
        actions_out: np.ndarray,
        logp_out: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        focal_indices = np.flatnonzero(focal_rows)
        if focal_indices.size:
            self._apply_policy_rows_ids(
                model=actor.model,
                hidden_state=actor.seat_hidden,
                row_indices=focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
            )
        opponent_indices = np.flatnonzero(~focal_rows)
        if opponent_indices.size:
            self._apply_opponent_rows_ids(
                actor=actor,
                row_indices=opponent_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
            )

    def _apply_opponent_rows_mask(
        self,
        *,
        actor: _ActorState,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray,
        values_out: np.ndarray,
        actions_out: np.ndarray,
        logp_out: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in row_indices.tolist()}):
            policy_rows = row_indices[actor.opponent_policy_id_by_env[row_indices] == policy_id]
            if not policy_rows.size:
                continue
            if policy_id == _MIRROR_OPPONENT_POLICY_ID:
                self._apply_policy_rows_mask(
                    model=actor.model,
                    hidden_state=actor.seat_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_mask=legal_mask,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                )
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            self._advance_hidden_only(
                model=actor.model,
                hidden_state=actor.seat_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
            )
            self._apply_policy_rows_mask(
                model=model,
                hidden_state=actor.opponent_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
            )

    def _apply_opponent_rows_ids(
        self,
        *,
        actor: _ActorState,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        logits_out: np.ndarray,
        values_out: np.ndarray,
        actions_out: np.ndarray,
        logp_out: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in row_indices.tolist()}):
            policy_rows = row_indices[actor.opponent_policy_id_by_env[row_indices] == policy_id]
            if not policy_rows.size:
                continue
            if policy_id == _MIRROR_OPPONENT_POLICY_ID:
                self._apply_policy_rows_ids(
                    model=actor.model,
                    hidden_state=actor.seat_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                )
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            self._advance_hidden_only(
                model=actor.model,
                hidden_state=actor.seat_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
            )
            self._apply_policy_rows_ids(
                model=model,
                hidden_state=actor.opponent_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
            )

    def _apply_policy_rows_mask(
        self,
        *,
        model: PolicyValueModel,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray,
        values_out: np.ndarray,
        actions_out: np.ndarray,
        logp_out: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        with torch.inference_mode():
            logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                torch.as_tensor(obs_step[row_indices], device=self._device),
                torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                hidden_state[row_indices],
            )
        hidden_state[row_indices] = torch.as_tensor(next_hidden, device=self._device).clone()
        logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        value_subset = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        action_subset, logp_subset, _entropy = sample_actions_from_mask(
            logits_subset,
            legal_mask[row_indices],
            rng=rng,
            pass_action_id=self.config.pass_action_id,
        )
        logits_out[row_indices] = logits_subset
        values_out[row_indices] = value_subset
        actions_out[row_indices] = action_subset
        logp_out[row_indices] = logp_subset

    def _apply_policy_rows_ids(
        self,
        *,
        model: PolicyValueModel,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        logits_out: np.ndarray,
        values_out: np.ndarray,
        actions_out: np.ndarray,
        logp_out: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        with torch.inference_mode():
            logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                torch.as_tensor(obs_step[row_indices], device=self._device),
                torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                hidden_state[row_indices],
            )
        hidden_state[row_indices] = torch.as_tensor(next_hidden, device=self._device).clone()
        logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        value_subset = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        subset_ids, subset_offsets = _slice_packed_rows(legal_ids, legal_offsets, row_indices)
        action_subset, logp_subset, _entropy = sample_actions_from_legal_ids(
            logits_subset,
            subset_ids,
            subset_offsets,
            rng=rng,
            pass_action_id=self.config.pass_action_id,
        )
        logits_out[row_indices] = logits_subset
        values_out[row_indices] = value_subset
        actions_out[row_indices] = action_subset
        logp_out[row_indices] = logp_subset

    def _update_outcomes(
        self,
        *,
        actor: _ActorState,
        acting_seat: np.ndarray,
        terminal_batch: DecisionBoundaryBatch,
        done: np.ndarray,
    ) -> None:
        if not np.any(done):
            return
        for env_index in np.flatnonzero(done):
            opponent_policy_id = str(actor.opponent_policy_id_by_env[env_index])
            if opponent_policy_id == _MIRROR_OPPONENT_POLICY_ID:
                continue
            result = game_result_from_step(
                terminal_batch,
                env_index=int(env_index),
                acting_seat=int(acting_seat[int(env_index)]),
            )
            focal_seat = int(actor.focal_seat_by_env[int(env_index)])
            if result.truncated or result.winner_seat is None:
                outcome = "d"
            elif int(result.winner_seat) == focal_seat:
                outcome = "w"
            else:
                outcome = "l"
            self._outcomes.update(opponent_policy_id, outcome)

    def _advance_hidden_only(
        self,
        *,
        model: PolicyValueModel,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
    ) -> None:
        with torch.inference_mode():
            _logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
                torch.as_tensor(obs_step[row_indices], device=self._device),
                torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                hidden_state[row_indices],
            )
        hidden_state[row_indices] = torch.as_tensor(next_hidden, device=self._device).clone()

    def _collect_actor_unroll(self, actor: _ActorState) -> RuntimeUnroll:
        T = int(self.config.unroll_length)
        N = int(self.config.envs_per_actor)
        obs = np.zeros((T, N, self.observation_dim), dtype=np.float32)
        actions = np.zeros((T, N), dtype=np.int64)
        rewards = np.zeros((T, N), dtype=np.float32)
        terminated = np.zeros((T, N), dtype=np.bool_)
        truncated = np.zeros((T, N), dtype=np.bool_)
        to_play_seat = np.zeros((T, N), dtype=np.int64)
        behavior_logp = np.zeros((T, N), dtype=np.float32)
        logits = np.zeros((T, N, self.action_dim), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        episode_seed = np.zeros((T, N), dtype=np.uint64)
        policy_train_mask = np.zeros((T, N), dtype=np.bool_)
        packed_ids: list[np.ndarray] = []
        packed_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
        mask_steps: list[np.ndarray] = []

        batch = actor.current_batch
        initial_hidden_state = actor.seat_hidden.detach().cpu().numpy().copy()
        for step_index in range(T):
            obs_step = np.asarray(batch.obs, dtype=np.float32)
            actor_step = np.asarray(batch.actor, dtype=np.int64)
            if obs_step.shape != (N, self.observation_dim):
                raise RuntimeError(f"unexpected actor obs shape: {obs_step.shape}")
            if np.any((actor_step != 0) & (actor_step != 1)):
                raise RuntimeError(f"actor runtime only supports live seat rows, got {actor_step.tolist()}")
            focal_rows = actor_step == actor.focal_seat_by_env
            logits_step = np.zeros((N, self.action_dim), dtype=np.float32)
            value_step = np.zeros((N,), dtype=np.float32)
            action_step = np.zeros((N,), dtype=np.int64)
            logp_step = np.zeros((N,), dtype=np.float32)
            policy_train_mask[step_index] = focal_rows

            if actor.layout_name == "i16_legal_ids":
                legal_ids, legal_offsets = _require_ids_offsets(batch)
                offset_base = int(packed_offsets[-1][-1])
                packed_ids.append(np.asarray(legal_ids, dtype=np.uint32))
                packed_offsets.append(np.asarray(legal_offsets[1:] + offset_base, dtype=np.uint32))
                self._fill_policy_outputs_ids(
                    actor=actor,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    focal_rows=focal_rows,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    logits_out=logits_step,
                    values_out=value_step,
                    actions_out=action_step,
                    logp_out=logp_step,
                    rng=actor.rng,
                )
            else:
                legal_mask = _require_mask(batch)
                mask_steps.append(np.asarray(legal_mask, dtype=np.bool_))
                self._fill_policy_outputs_mask(
                    actor=actor,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    focal_rows=focal_rows,
                    legal_mask=legal_mask,
                    logits_out=logits_step,
                    values_out=value_step,
                    actions_out=action_step,
                    logp_out=logp_step,
                    rng=actor.rng,
                )

            next_batch = actor.env.step(action_step.astype(np.uint32, copy=False))
            done = np.logical_or(next_batch.terminated, next_batch.truncated)

            obs[step_index] = obs_step
            actions[step_index] = action_step
            rewards[step_index] = np.asarray(next_batch.reward, dtype=np.float32)
            terminated[step_index] = np.asarray(next_batch.terminated, dtype=np.bool_)
            truncated[step_index] = np.asarray(next_batch.truncated, dtype=np.bool_)
            to_play_seat[step_index] = actor_step
            behavior_logp[step_index] = logp_step
            logits[step_index] = logits_step
            values[step_index] = value_step
            episode_seed[step_index] = np.asarray(next_batch.episode_seed, dtype=np.uint64)

            if np.any(done):
                self._update_outcomes(
                    actor=actor,
                    acting_seat=actor_step,
                    terminal_batch=next_batch,
                    done=done.astype(np.bool_, copy=False),
                )
                reset_hidden = actor.model.initial_seat_hidden(int(np.count_nonzero(done)), device=self._device)
                done_mask = torch.as_tensor(done, dtype=torch.bool, device=self._device)
                actor.seat_hidden[done_mask] = reset_hidden
                actor.opponent_hidden[done_mask] = reset_hidden
                self._assign_episode_roles(actor, done.astype(np.bool_, copy=False))
                batch = self._reset_done_rows(actor, done.astype(np.bool_, copy=False))
            else:
                batch = next_batch

        actor.current_batch = batch
        unroll = RuntimeUnroll(
            actor_id=actor.actor_id,
            unroll_seq=actor.next_unroll_seq,
            behavior_policy_version=actor.snapshot_version,
            unroll_hash=_hash_unroll(actions=actions, rewards=rewards, episode_seed=episode_seed),
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            to_play_seat=to_play_seat,
            behavior_logp=behavior_logp,
            logits=logits,
            values=values,
            legal_actions=(
                LegalActionBatch.from_packed(
                    np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                    np.concatenate(packed_offsets, axis=0),
                )
                if actor.layout_name == "i16_legal_ids"
                else LegalActionBatch.from_mask(np.stack(mask_steps, axis=0))
            ),
            bootstrap_obs=np.asarray(batch.obs, dtype=np.float32),
            bootstrap_actor=np.asarray(batch.actor, dtype=np.int64),
            initial_hidden_state=initial_hidden_state,
            final_hidden_state=actor.seat_hidden.detach().cpu().numpy().copy(),
            episode_seed=episode_seed,
            policy_train_mask=policy_train_mask,
        )
        actor.next_unroll_seq += 1
        return unroll

    def _build_learner_batch(
        self,
        unrolls: Sequence[RuntimeUnroll],
        *,
        gamma: float,
        truncation_bootstrap_value: bool,
        vtrace_rho_bar: float,
        vtrace_c_bar: float,
    ) -> dict[str, Any]:
        masks = [
            unroll.legal_actions.to_mask(
                expected_shape=(int(unroll.obs.shape[0]), int(unroll.obs.shape[1])),
                action_space=self.action_dim,
            )
            for unroll in unrolls
        ]
        bootstrap_values = [self._bootstrap_values(unroll) for unroll in unrolls]
        obs = np.concatenate([unroll.obs for unroll in unrolls], axis=1)
        actions = np.concatenate([unroll.actions for unroll in unrolls], axis=1)
        rewards = np.concatenate([unroll.rewards for unroll in unrolls], axis=1)
        terminated = np.concatenate([unroll.terminated for unroll in unrolls], axis=1)
        truncated = np.concatenate([unroll.truncated for unroll in unrolls], axis=1)
        to_play_seat = np.concatenate([unroll.to_play_seat for unroll in unrolls], axis=1)
        behavior_logp = np.concatenate([unroll.behavior_logp for unroll in unrolls], axis=1)
        logits = np.concatenate([unroll.logits for unroll in unrolls], axis=1)
        legal_mask = np.concatenate(masks, axis=1)
        initial_hidden_state = np.concatenate([unroll.initial_hidden_state for unroll in unrolls], axis=0)
        policy_train_mask = np.concatenate([unroll.policy_train_mask for unroll in unrolls], axis=1)

        target_logp_parts = [
            (
                masked_logp_from_legal_ids(
                    unroll.logits.reshape(-1, self.action_dim),
                    unroll.legal_actions.ids,
                    unroll.legal_actions.offsets,
                    unroll.actions.reshape(-1),
                    pass_action_id=self.config.pass_action_id,
                ).reshape(unroll.actions.shape)
                if unroll.legal_actions.ids is not None and unroll.legal_actions.offsets is not None
                else masked_logp_from_mask(
                    unroll.logits.reshape(-1, self.action_dim),
                    cast(np.ndarray, unroll.legal_actions.mask).reshape(-1, self.action_dim),
                    unroll.actions.reshape(-1),
                    pass_action_id=self.config.pass_action_id,
                ).reshape(unroll.actions.shape)
            )
            for unroll in unrolls
        ]
        target_logp = np.concatenate(target_logp_parts, axis=1)
        discounts = np.logical_not(terminated).astype(np.float32) * float(gamma)
        if not truncation_bootstrap_value:
            discounts *= np.logical_not(truncated).astype(np.float32)

        value_prefix = np.concatenate([unroll.values for unroll in unrolls], axis=1)
        value_suffix = np.concatenate(bootstrap_values, axis=0)[np.newaxis, :]
        values = np.concatenate([value_prefix, value_suffix], axis=0)
        vtrace_result: VTraceTargets = compute_vtrace_targets(
            rewards,
            values,
            discounts,
            behavior_logp,
            target_logp,
            rho_bar=vtrace_rho_bar,
            c_bar=vtrace_c_bar,
        )
        return {
            "obs": obs,
            "actions": actions,
            "legal_actions": LegalActionBatch.from_mask(legal_mask),
            "legal_mask": legal_mask,
            "to_play_seat": to_play_seat,
            "actor": to_play_seat,
            "initial_hidden_state": initial_hidden_state,
            "rewards": rewards,
            "discounts": discounts,
            "behavior_logp": behavior_logp,
            "behavior_logits": logits,
            "logits": logits,
            "vtrace_result": vtrace_result,
            "vtrace_rho_bar": float(vtrace_rho_bar),
            "vtrace_c_bar": float(vtrace_c_bar),
            "policy_train_mask": policy_train_mask,
        }

    def _bootstrap_values(self, unroll: RuntimeUnroll) -> np.ndarray:
        bootstrap_value = np.zeros((unroll.bootstrap_obs.shape[0],), dtype=np.float32)
        valid_rows = (unroll.bootstrap_actor == 0) | (unroll.bootstrap_actor == 1)
        if not np.any(valid_rows):
            return bootstrap_value
        actor_model = self._actors[int(unroll.actor_id)].model
        with torch.inference_mode():
            _, value_tensor, _ = actor_model.forward_seat_aware(
                torch.as_tensor(unroll.bootstrap_obs[valid_rows], device=self._device),
                torch.as_tensor(unroll.bootstrap_actor[valid_rows], device=self._device, dtype=torch.long),
                torch.as_tensor(unroll.final_hidden_state[valid_rows], device=self._device),
            )
        bootstrap_value[valid_rows] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return bootstrap_value

    def _runtime_metrics(
        self,
        selected: Sequence[RuntimeUnroll],
        *,
        occupancy_samples: Sequence[float],
    ) -> dict[str, float]:
        total_env_steps = sum(int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in selected)
        elapsed = max(time.time() - self._runtime_start, 1e-6)
        policy_lags = [
            float(self._last_published_snapshot_version - unroll.behavior_policy_version) for unroll in selected
        ]
        occupancy = np.asarray(tuple(occupancy_samples) or (0.0,), dtype=np.float64)
        lag_array = np.asarray(policy_lags or (0.0,), dtype=np.float64)
        return {
            "actor_env_steps_per_sec": float(total_env_steps / elapsed),
            "queue_occupancy_p50": float(np.percentile(occupancy, 50)),
            "queue_occupancy_p90": float(np.percentile(occupancy, 90)),
            "policy_version_lag_p50": float(np.percentile(lag_array, 50)),
            "policy_version_lag_p90": float(np.percentile(lag_array, 90)),
            "pfsp_pool_size": float(self._pfsp_pool_size),
            "pfsp_sampled_envs": float(self._pfsp_last_sampled_envs),
            "pfsp_mirror_envs": float(self._pfsp_last_mirror_envs),
        }

    def _reset_done_rows(self, actor: _ActorState, done: np.ndarray) -> DecisionBoundaryBatch:
        try:
            return actor.env.reset_done(done)
        except RuntimeError:
            initial_hidden = actor.model.initial_seat_hidden(
                int(self.config.envs_per_actor),
                device=self._device,
            ).clone()
            actor.seat_hidden = initial_hidden.clone()
            actor.opponent_hidden = initial_hidden
            full_reset = np.ones(actor.focal_seat_by_env.shape, dtype=np.bool_)
            self._assign_episode_roles(actor, full_reset, initial=True)
            fallback_seed = int(actor.rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
            return actor.env.reset(seed=fallback_seed)


def build_runtime_config(
    *,
    stack: StackConfig,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
) -> QueueRuntimeConfig:
    system = stack.config.system
    training = stack.config.training_family_a
    if system is None or training is None:
        raise RuntimeError("stack config is missing system or training_family_a blocks")

    actor_count = 1
    envs_per_actor = int(num_envs)
    configured_actor_count = int(system.actor_process_count)
    configured_envs_per_actor = int(system.envs_per_actor)
    if int(num_envs) == configured_actor_count * configured_envs_per_actor:
        actor_count = configured_actor_count
        envs_per_actor = configured_envs_per_actor

    return QueueRuntimeConfig(
        mode=runtime_mode,
        actor_count=actor_count,
        envs_per_actor=envs_per_actor,
        unroll_length=int(unroll_length),
        batch_unrolls_per_update=int(training.batch_unrolls_per_update),
        queue_capacity_unrolls=max(int(system.actor_queue_capacity_unrolls), int(training.batch_unrolls_per_update)),
        profile=profile,
        base_seed=int(seed),
        pass_action_id=int(pass_action_id),
        actor_reload_interval_updates=max(1, int(training.actor_reload_interval_updates)),
    )


def _actor_seed(base_seed: int, actor_id: int) -> int:
    return int(np.uint64(base_seed) ^ (np.uint64(actor_id + 1) << np.uint64(32)))


def _require_ids_offsets(batch: DecisionBoundaryBatch) -> tuple[np.ndarray, np.ndarray]:
    if batch.ids_offsets is None:
        raise RuntimeError("QueueRuntime requires ids_offsets legality batches")
    legal_ids, legal_offsets = batch.ids_offsets
    return np.asarray(legal_ids, dtype=np.uint32), np.asarray(legal_offsets, dtype=np.uint32)


def _require_mask(batch: DecisionBoundaryBatch) -> np.ndarray:
    if batch.mask is None:
        raise RuntimeError("QueueRuntime expected dense mask legality for this actor batch")
    return np.asarray(batch.mask, dtype=np.bool_)


def _slice_packed_rows(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_ids: list[np.ndarray] = []
    offsets = [0]
    for row_index in row_indices.tolist():
        start = int(legal_offsets[int(row_index)])
        stop = int(legal_offsets[int(row_index) + 1])
        row_ids = np.asarray(legal_ids[start:stop], dtype=np.uint32)
        selected_ids.append(row_ids)
        offsets.append(offsets[-1] + int(row_ids.size))
    return (
        np.concatenate(selected_ids, axis=0) if selected_ids else np.zeros((0,), dtype=np.uint32),
        np.asarray(offsets, dtype=np.uint32),
    )


def _hash_unroll(*, actions: np.ndarray, rewards: np.ndarray, episode_seed: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (actions, rewards, episode_seed):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
