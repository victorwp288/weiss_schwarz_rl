from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.autoscale import (
    ResolvedTrainingTopology,
    ScalingRequest,
    hardware_profile_from_name,
    resolve_training_topology,
)
from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import (
    StackConfig,
    apply_stack_overrides,
    canonical_config_dict,
    compute_config_hash256,
    load_stack_config,
    parse_override_tokens,
)
from weiss_rl.distributed import (
    DistributedContext,
    all_reduce_float,
    average_gradients,
    barrier as distributed_barrier,
    broadcast_object,
    destroy_process_group_if_initialized,
    distributed_context_from_env,
    init_process_group_if_needed,
    rank_seed,
    shard_env_count,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval import (
    DevEvalPolicySummary,
    EvalGameRecord,
    PayoffFoldScheme,
    Pcg32XshRrV1,
    build_matchup_export,
    build_seat_advantage_diagnostics,
    build_seat_swapped_schedule,
    game_result_from_step,
    paired_seed_scores,
    record_completed_game,
    run_seat_swapped_matchup,
    sample_action_pinned,
    summarize_game_records,
    write_episodes_jsonl,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.eval.harness import ScheduledGame, abort_on_engine_fault_eval
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_POLICY_ID,
    heuristic_public_profile_name_for_policy_id,
    select_final_policy_set_deterministic_v1,
)
from weiss_rl.eval.simulator_runner import _resolve_eval_device
from weiss_rl.league import (
    PromotionGateAnchor,
    PromotionGateAnchorResult,
    PromotionGatePosterior,
    PromotionGateRate,
    PromotionGateResult,
    build_promotion_gate_result,
    resolve_promotion_gate_anchors,
    resolve_promotion_gate_seed_file,
    run_promotion_gate,
)
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SNAPSHOT_WEIGHTS_FILENAME,
    SnapshotMeta,
    SnapshotRegistry,
    snapshot_weights_relpath,
)
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.manifest import (
    RunArtifacts,
    RunManifest,
    build_seed_file_manifest,
    default_run_dir_name,
    write_run_artifacts,
)
from weiss_rl.masking import assert_strictly_increasing_legal_ids, masked_logp_from_mask
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.repro import (
    canonical_json_bytes,
    compute_run_id64,
    compute_run_id256,
    hash_seed_file,
    parse_seed_file,
    stable_hash64,
)
from weiss_rl.runtime import QueueRuntime, QueueRuntimeMode, build_runtime_config, resolve_actor_device_layout
from weiss_rl.schedules import linear_anneal_value
from weiss_rl.simulator_contract import SimulatorContract, load_verified_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract
from weiss_rl.tensorboard_logger import TensorBoardLogger, tensorboard_unavailable_reason
from weiss_rl.toy_public_demo import (
    PUBLIC_DEMO_MODE,
    public_demo_simulator_info,
    public_demo_spec_bundle,
    public_demo_spec_hash256,
    stage_public_demo_run,
)

_SHA256_HEX_LENGTH = 64
_U64_MASK = (1 << 64) - 1
_PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"
_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT = "baseline_checkpoint.pt"
_FIXED_OPPONENT_EXCLUSIONS = frozenset({_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID})
_LATEST_CHECKPOINT_FILENAME = "latest.pt"
_BEST_CHECKPOINT_FILENAME = "best.pt"
_CHECKPOINT_TRACKER_FILENAME = "checkpoint_tracker.json"
_CHECKPOINT_TRACKER_FORMAT = "checkpoint_tracker_v2"
_PERIODIC_DEV_EVAL_SUMMARY_FORMAT = "periodic_dev_eval_summary_v2"
_B2_DISAGREEMENT_AUDIT_REQUESTS_FILENAME = "b2_disagreement_audit_requests.jsonl"
_IMPALA_ALGORITHMS = frozenset(
    {"impala_vtrace_gru", "impala_vtrace_ff", "structured_v2", "impala_vtrace_structured_v1"}
)
from weiss_rl.residual_policy import (
    FrozenStoredLogitResidual,
    TrainableLiveFrozenB1Residual,
    load_frozen_stored_logit_residual,
)
_PPO_ALGORITHMS = frozenset({"ppo_lite_masked_v1"})
_CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL = 0.1
_CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS = 0.05
_B2_FLATLINE_WINDOW = 3
_B2_FLATLINE_MAX_DELTA = 0.02
_B2_FLATLINE_LOW_SCORE = 0.35
_B2_ACTION_WARNING_SCORE_THRESHOLD = 0.25
_B2_ACTION_WARNING_MAIN_MOVE_RATE = 0.45
_B2_ACTION_WARNING_PASS_NONPASS_RATE = 0.05
_GIT_COMMIT_HEX_LENGTH = 40
_EVAL_SNAPSHOT_MODEL_CACHE_MAX_ENTRIES = 12
_EVAL_SNAPSHOT_MODEL_CACHE: OrderedDict[tuple[Any, ...], PolicyValueModel] = OrderedDict()


@dataclass(frozen=True, slots=True)
class MinimalRollout:
    obs: np.ndarray
    legal_mask: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    logits: np.ndarray
    values: np.ndarray
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    training_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    snapshots_dir: Path
    tensorboard_dir: Path
    scalars_path: Path
    performance_log_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    checkpoint_tracker_path: Path


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    checkpoint_path: Path
    update_count: int
    policy_version: int
    total_samples_processed: int


@dataclass(frozen=True, slots=True)
class PeriodicDevEvalOpponentSpec:
    policy_id: str
    display_name: str
    kind: str
    snapshot_path: str | None = None
    heuristic_profile: str | None = None


@dataclass(frozen=True, slots=True)
class PeriodicDevEvalSeedBlockJob:
    opponent_index: int
    block_index: int
    opponent_spec: PeriodicDevEvalOpponentSpec
    paired_seed_items: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PromotionGateSeedBlockJob:
    anchor_index: int
    block_index: int
    anchor_spec: PeriodicDevEvalOpponentSpec
    paired_seed_items: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class _PeriodicDevEvalActiveGame:
    env: DecisionBoundaryEnv
    scheduled_game: ScheduledGame
    batch: DecisionBoundaryBatch
    focal_hidden: torch.Tensor
    opponent_hidden: torch.Tensor | None
    seat_rngs: dict[int, Pcg32XshRrV1]
    last_acting_seat: int | None = None
    completed: bool = False


@dataclass(frozen=True, slots=True)
class AsyncPeriodicDevEvalRequest:
    stack: StackConfig
    checkpoint_path: Path
    focal_policy_id: str
    update_count: int
    policy_version: int
    run_dir: Path
    run_id256: str
    config_hash256: str
    spec_hash256: str
    artifact_dir_name: str
    artifact_scope: str
    paired_seeds: tuple[int, ...]
    opponents: tuple[PeriodicDevEvalOpponentSpec, ...]
    eval_device_override: str | None
    parallel_workers: int
    parallel_worker_devices: tuple[str, ...]


@dataclass(slots=True)
class PendingPeriodicDevEval:
    future: Future[dict[str, Any]]
    request: AsyncPeriodicDevEvalRequest
    pinned_snapshot_ids: tuple[str, ...]
    latest_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class AsyncPromotionGateRequest:
    stack: StackConfig
    run_dir: Path
    candidate_policy_id: str
    candidate_snapshot_path: str
    update_count: int
    policy_version: int
    run_id256: str
    config_hash256: str
    spec_hash256: str
    anchor_policy_ids: dict[str, str]
    anchor_specs: tuple[PeriodicDevEvalOpponentSpec, ...]
    eval_device_override: str | None


@dataclass(slots=True)
class PendingPromotionGate:
    future: Future[dict[str, Any]]
    request: AsyncPromotionGateRequest
    pinned_snapshot_ids: tuple[str, ...]


class _PeriodicDevEvalRunner:
    def __init__(
        self,
        *,
        stack: StackConfig,
        model: PolicyValueModel,
        opponent_policy_id: str,
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        focal_policy_id: str,
        require_sorted_legal_ids: bool,
        eval_device: torch.device | str | None = None,
        opponent_model: PolicyValueModel | None = None,
        heuristic_policy: HeuristicPublicPolicy | None = None,
    ) -> None:
        self.stack = stack
        self.model = model
        self.opponent_policy_id = opponent_policy_id
        self.opponent_model = opponent_model
        self.heuristic_policy = heuristic_policy
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pass_action_id = pass_action_id
        self.artifact_dir = artifact_dir
        self.focal_policy_id = focal_policy_id
        self.require_sorted_legal_ids = require_sorted_legal_ids
        self._baseline_logits = np.zeros((action_dim,), dtype=np.float32)
        self._device = _resolve_eval_device(stack, eval_device=eval_device)
        self._persistent_env: DecisionBoundaryEnv | None = None
        self._batched_envs: list[DecisionBoundaryEnv] = []
        self._runtime_seconds: dict[str, float] = {}
        self._runtime_counts: dict[str, int] = {}

    def close(self) -> None:
        env = self._persistent_env
        self._persistent_env = None
        if env is not None:
            env.close()
        batched_envs = self._batched_envs
        self._batched_envs = []
        for batched_env in batched_envs:
            batched_env.close()

    def drain_runtime_counters(self) -> dict[str, Any]:
        seconds = {key: float(value) for key, value in sorted(self._runtime_seconds.items())}
        counts = {key: int(value) for key, value in sorted(self._runtime_counts.items())}
        self._runtime_seconds.clear()
        self._runtime_counts.clear()
        return {
            "seconds": seconds,
            "counts": counts,
        }

    def run_scheduled_games_batched(
        self,
        scheduled_games: Sequence[ScheduledGame],
    ) -> tuple[tuple[ScheduledGame, Any], ...]:
        if not scheduled_games:
            return ()
        env_seed = int(scheduled_games[0].episode_seed)
        active_games = [self._start_active_game(scheduled_game, env_seed=env_seed) for scheduled_game in scheduled_games]
        completed: list[tuple[ScheduledGame, Any]] = []
        remaining = len(active_games)
        while remaining > 0:
            ready_slots: list[tuple[int, _PeriodicDevEvalActiveGame]] = []
            for index, active_game in enumerate(active_games):
                if active_game.completed:
                    continue
                batch = active_game.batch
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    completed.append(
                        (
                            active_game.scheduled_game,
                            game_result_from_step(
                                batch,
                                env_index=0,
                                acting_seat=active_game.last_acting_seat,
                                episode_seed=active_game.scheduled_game.episode_seed,
                                max_decisions=getattr(active_game.env, "max_decisions", None),
                                max_ticks=getattr(active_game.env, "max_ticks", None),
                                max_no_progress_decisions=getattr(active_game.env, "max_no_progress_decisions", None),
                            ),
                        )
                    )
                    active_game.completed = True
                    remaining -= 1
                    continue
                ready_slots.append((index, active_game))

            if not ready_slots:
                continue

            actions = self._select_batched_actions(ready_slots)
            for index, action in actions.items():
                active_game = active_games[index]
                active_game.last_acting_seat = int(active_game.batch.actor[0])
                started = time.perf_counter()
                active_game.batch = active_game.env.step(np.asarray([action], dtype=np.uint32))
                self._add_seconds("env_step", time.perf_counter() - started)
                self._abort_on_fault(active_game.batch)
        completed.sort(key=lambda item: (item[0].episode_index, item[0].pair_index, item[0].swap_index))
        return tuple(completed)

    def run_game(self, scheduled_game: ScheduledGame):
        self._add_count("games")
        env = self._env_for_game(seed=scheduled_game.episode_seed)
        focal_hidden = self.model.initial_seat_hidden(1, device=self._device)
        opponent_hidden = (
            None if self.opponent_model is None else self.opponent_model.initial_seat_hidden(1, device=self._device)
        )
        seat_rngs = {
            seat: Pcg32XshRrV1(_periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=seat)) for seat in (0, 1)
        }
        last_acting_seat: int | None = None

        started = time.perf_counter()
        batch = env.reset(seed=scheduled_game.episode_seed)
        self._add_seconds("env_reset", time.perf_counter() - started)
        self._abort_on_fault(batch)
        while True:
            if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                return game_result_from_step(
                    batch,
                    env_index=0,
                    acting_seat=last_acting_seat,
                    episode_seed=scheduled_game.episode_seed,
                    max_decisions=getattr(env, "max_decisions", None),
                    max_ticks=getattr(env, "max_ticks", None),
                    max_no_progress_decisions=getattr(env, "max_no_progress_decisions", None),
                )

            current_seat = int(batch.actor[0])
            self._add_count("decisions")
            started = time.perf_counter()
            action, focal_hidden, opponent_hidden = self._select_action(
                batch=batch,
                scheduled_game=scheduled_game,
                current_seat=current_seat,
                focal_hidden=focal_hidden,
                opponent_hidden=opponent_hidden,
                rng=seat_rngs[current_seat],
            )
            self._add_seconds("select_action", time.perf_counter() - started)
            last_acting_seat = current_seat
            started = time.perf_counter()
            batch = env.step(np.asarray([action], dtype=np.uint32))
            self._add_seconds("env_step", time.perf_counter() - started)
            self._abort_on_fault(batch)

    def _env_for_game(self, *, seed: int) -> DecisionBoundaryEnv:
        if self._persistent_env is None:
            self._persistent_env = _build_ids_eval_env(
                self.stack,
                seed=seed,
                pass_action_id=self.pass_action_id,
            )
        return self._persistent_env

    def _start_active_game(self, scheduled_game: ScheduledGame, *, env_seed: int) -> _PeriodicDevEvalActiveGame:
        self._add_count("games")
        env = _build_ids_eval_env(
            self.stack,
            seed=env_seed,
            pass_action_id=self.pass_action_id,
        )
        self._batched_envs.append(env)
        started = time.perf_counter()
        batch = env.reset(seed=scheduled_game.episode_seed)
        self._add_seconds("env_reset", time.perf_counter() - started)
        self._abort_on_fault(batch)
        return _PeriodicDevEvalActiveGame(
            env=env,
            scheduled_game=scheduled_game,
            batch=batch,
            focal_hidden=self.model.initial_seat_hidden(1, device=self._device),
            opponent_hidden=(
                None
                if self.opponent_model is None
                else self.opponent_model.initial_seat_hidden(1, device=self._device)
            ),
            seat_rngs={
                seat: Pcg32XshRrV1(_periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=seat))
                for seat in (0, 1)
            },
        )

    def _select_batched_actions(
        self,
        ready_slots: Sequence[tuple[int, _PeriodicDevEvalActiveGame]],
    ) -> dict[int, int]:
        actions: dict[int, int] = {}
        focal_requests: list[tuple[int, _PeriodicDevEvalActiveGame, np.ndarray]] = []
        opponent_requests: list[tuple[int, _PeriodicDevEvalActiveGame, np.ndarray]] = []

        for index, active_game in ready_slots:
            batch = active_game.batch
            current_seat = int(batch.actor[0])
            self._add_count("decisions")
            legal_ids = _legal_ids_for_env_row(
                batch=batch,
                env_index=0,
                require_sorted=self.require_sorted_legal_ids,
            )
            current_policy_id = (
                active_game.scheduled_game.seat0_policy_id
                if current_seat == 0
                else active_game.scheduled_game.seat1_policy_id
            )
            if current_policy_id == self.focal_policy_id:
                self._add_count("focal_model_actions")
                focal_requests.append((index, active_game, legal_ids))
                continue
            if self.opponent_model is not None and current_policy_id == self.opponent_policy_id:
                self._add_count("opponent_model_actions")
                opponent_requests.append((index, active_game, legal_ids))
                continue
            if self.heuristic_policy is not None and current_policy_id == self.opponent_policy_id:
                self._add_count("heuristic_actions")
                actions[index] = int(
                    self.heuristic_policy.choose_action(
                        np.asarray(batch.obs[0], dtype=np.float32),
                        legal_ids,
                    )
                )
                continue
            self._add_count("random_legal_actions")
            action, _ = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=active_game.seat_rngs[current_seat],
                pass_action_id=self.pass_action_id,
            )
            actions[index] = int(action)

        actions.update(
            self._sample_batched_model_actions(
                model=self.model,
                requests=focal_requests,
                hidden_name="focal_hidden",
            )
        )
        if self.opponent_model is not None:
            actions.update(
                self._sample_batched_model_actions(
                    model=self.opponent_model,
                    requests=opponent_requests,
                    hidden_name="opponent_hidden",
                )
            )
        return actions

    def _sample_batched_model_actions(
        self,
        *,
        model: PolicyValueModel,
        requests: Sequence[tuple[int, _PeriodicDevEvalActiveGame, np.ndarray]],
        hidden_name: str,
    ) -> dict[int, int]:
        if not requests:
            return {}
        started = time.perf_counter()
        obs = torch.as_tensor(
            np.stack([np.asarray(active_game.batch.obs[0], dtype=np.float32) for _index, active_game, _legal in requests]),
            device=self._device,
        )
        acting_seats = torch.as_tensor(
            [int(active_game.batch.actor[0]) for _index, active_game, _legal in requests],
            device=self._device,
            dtype=torch.long,
        )
        hidden_tensors: list[torch.Tensor] = []
        for _index, active_game, _legal in requests:
            hidden = getattr(active_game, hidden_name)
            if hidden is None:
                raise RuntimeError(f"Missing periodic dev-eval hidden state for {hidden_name}")
            hidden_tensors.append(hidden)
        seat_hidden = torch.cat(hidden_tensors, dim=0)
        with torch.inference_mode():
            logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
                obs,
                acting_seats,
                seat_hidden,
                scoring_mode="learner",
            )
        self._add_seconds("model_forward", time.perf_counter() - started)
        self._add_count("model_forward_calls")
        self._add_count("model_forward_rows", len(requests))
        started = time.perf_counter()
        logits_batch = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        self._add_seconds("logits_to_cpu", time.perf_counter() - started)

        actions: dict[int, int] = {}
        started = time.perf_counter()
        for row, (index, active_game, legal_ids) in enumerate(requests):
            current_seat = int(active_game.batch.actor[0])
            action, _ = sample_action_pinned(
                logits_batch[row],
                legal_ids,
                rng=active_game.seat_rngs[current_seat],
                pass_action_id=self.pass_action_id,
            )
            setattr(active_game, hidden_name, next_hidden[row : row + 1].detach())
            actions[index] = int(action)
        self._add_seconds("sample_action", time.perf_counter() - started)
        return actions

    def _select_action(
        self,
        *,
        batch: DecisionBoundaryBatch,
        scheduled_game: ScheduledGame,
        current_seat: int,
        focal_hidden: torch.Tensor,
        opponent_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
    ) -> tuple[int, torch.Tensor, torch.Tensor | None]:
        legal_ids = _legal_ids_for_env_row(
            batch=batch,
            env_index=0,
            require_sorted=self.require_sorted_legal_ids,
        )
        current_policy_id = scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
        if current_policy_id == self.focal_policy_id:
            self._add_count("focal_model_actions")
            action, focal_hidden = self._sample_model_action(
                model=self.model,
                seat_hidden=focal_hidden,
                batch=batch,
                current_seat=current_seat,
                legal_ids=legal_ids,
                rng=rng,
            )
            return action, focal_hidden, opponent_hidden
        if self.opponent_model is not None and current_policy_id == self.opponent_policy_id:
            assert opponent_hidden is not None
            self._add_count("opponent_model_actions")
            action, opponent_hidden = self._sample_model_action(
                model=self.opponent_model,
                seat_hidden=opponent_hidden,
                batch=batch,
                current_seat=current_seat,
                legal_ids=legal_ids,
                rng=rng,
            )
            return action, focal_hidden, opponent_hidden
        if self.heuristic_policy is not None and current_policy_id == self.opponent_policy_id:
            self._add_count("heuristic_actions")
            action = self.heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), focal_hidden, opponent_hidden
        self._add_count("random_legal_actions")
        action, _ = sample_action_pinned(
            self._baseline_logits,
            legal_ids,
            rng=rng,
            pass_action_id=self.pass_action_id,
        )
        return action, focal_hidden, opponent_hidden

    def _sample_model_action(
        self,
        *,
        model: PolicyValueModel,
        seat_hidden: torch.Tensor,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        legal_ids: np.ndarray,
        rng: Pcg32XshRrV1,
    ) -> tuple[int, torch.Tensor]:
        started = time.perf_counter()
        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                seat_hidden,
                scoring_mode="learner",
            )
        self._add_seconds("model_forward", time.perf_counter() - started)
        started = time.perf_counter()
        logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        self._add_seconds("logits_to_cpu", time.perf_counter() - started)
        started = time.perf_counter()
        action, _ = sample_action_pinned(
            logits,
            legal_ids,
            rng=rng,
            pass_action_id=self.pass_action_id,
        )
        self._add_seconds("sample_action", time.perf_counter() - started)
        return action, next_seat_hidden

    def _add_seconds(self, key: str, value: float) -> None:
        self._runtime_seconds[key] = self._runtime_seconds.get(key, 0.0) + max(0.0, float(value))

    def _add_count(self, key: str, value: int = 1) -> None:
        self._runtime_counts[key] = self._runtime_counts.get(key, 0) + int(value)

    def _abort_on_fault(self, batch: DecisionBoundaryBatch) -> None:
        abort_on_engine_fault_eval(
            run_dir=self.artifact_dir,
            engine_status=batch.engine_status,
            decision_id=batch.decision_id,
            episode_key=batch.episode_key,
            note="engine_status!=0 during periodic dev eval",
        )


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = _normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_snapshot_artifact(
    *,
    snapshots_dir: Path,
    run_dir: Path,
    checkpoint_path: Path,
    policy_id: str,
    update: int,
    config_hash256: str,
    device: torch.device,
    model_state_dict: dict[str, Any],
    public_heuristic_logit_bias_scale: float | None = None,
    public_heuristic_actor_logit_bias_scale: float | None = None,
) -> tuple[Path, str]:
    snapshot_dir = snapshots_dir / policy_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    weights_path = snapshot_dir / "weights.pt"
    weights_payload = {
        "format": "minimal_train_snapshot_weights_v1",
        "policy_id": policy_id,
        "update": int(update),
        "device": str(device),
        "config_hash256": config_hash256,
        "model_state_dict": model_state_dict,
        "public_heuristic_logit_bias_scale": public_heuristic_logit_bias_scale,
        "public_heuristic_actor_logit_bias_scale": public_heuristic_actor_logit_bias_scale,
    }
    torch.save(weights_payload, weights_path)
    weights_sha256 = _sha256_file(weights_path)

    _write_json_file(
        snapshot_dir / SNAPSHOT_METADATA_FILENAME,
        {
            "format": "minimal_train_snapshot_metadata_v1",
            "policy_id": policy_id,
            "update": int(update),
            "weights_path": snapshot_weights_relpath(policy_id),
            "weights_sha256": weights_sha256,
            "source_checkpoint_path": checkpoint_path.relative_to(run_dir).as_posix(),
        },
    )
    return weights_path, weights_sha256


def _sync_snapshot_registry_retention(stack: StackConfig, registry: SnapshotRegistry) -> None:
    league = stack.config.league
    if league is None:
        return
    registry.recent_size = int(league.snapshot_pool_recent_size)
    registry.champion_size = int(league.snapshot_pool_champion_size)


def _snapshot_artifact_dir_for_prune(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    snapshot: SnapshotMeta,
) -> Path:
    snapshots_root = training_paths.snapshots_dir.resolve()
    weights_path = (run_dir / snapshot.path).resolve()
    try:
        weights_path.relative_to(snapshots_root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to delete snapshot artifact outside {snapshots_root}: {snapshot.path}") from exc
    if weights_path.name != SNAPSHOT_WEIGHTS_FILENAME:
        raise RuntimeError(f"refusing to delete unexpected snapshot artifact path: {snapshot.path}")

    snapshot_dir = weights_path.parent
    try:
        snapshot_dir.relative_to(snapshots_root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to delete snapshot directory outside {snapshots_root}: {snapshot_dir}") from exc
    if snapshot_dir == snapshots_root or snapshot_dir.name != snapshot.policy_id:
        raise RuntimeError(f"refusing to delete unexpected snapshot directory: {snapshot_dir}")
    return snapshot_dir


def _delete_pruned_snapshot_artifacts(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    pruned_snapshots: list[SnapshotMeta],
) -> None:
    for snapshot in pruned_snapshots:
        snapshot_dir = _snapshot_artifact_dir_for_prune(
            training_paths=training_paths,
            run_dir=run_dir,
            snapshot=snapshot,
        )
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)


def _save_snapshot_registry_with_retention(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    registry: SnapshotRegistry,
) -> None:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    _sync_snapshot_registry_retention(stack, registry)
    pruned_snapshots = registry.prune()
    registry.save(registry_path)
    _delete_pruned_snapshot_artifacts(
        training_paths=training_paths,
        run_dir=run_dir,
        pruned_snapshots=pruned_snapshots,
    )


def _pin_snapshot_ids(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    snapshot_ids: Sequence[str],
) -> tuple[str, ...]:
    requested_ids = tuple(dict.fromkeys(str(snapshot_id).strip() for snapshot_id in snapshot_ids if str(snapshot_id).strip()))
    if not requested_ids:
        return ()
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    existing_pins = set(registry.pinned_snapshots)
    newly_pinned: list[str] = []
    for snapshot_id in requested_ids:
        registry.pin_snapshot(snapshot_id)
        if snapshot_id not in existing_pins:
            newly_pinned.append(snapshot_id)
    _save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    return tuple(newly_pinned)


def _unpin_snapshot_ids(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    snapshot_ids: Sequence[str],
) -> None:
    removable_ids = {str(snapshot_id).strip() for snapshot_id in snapshot_ids if str(snapshot_id).strip()}
    if not removable_ids:
        return
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    registry.pinned_snapshots = [
        snapshot_id for snapshot_id in registry.pinned_snapshots if snapshot_id not in removable_ids
    ]
    registry.normalize()
    _save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )


def _persist_snapshot_registry_entry(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    checkpoint_path: Path,
    model_state_dict: dict[str, Any],
    config_hash256: str,
    device: torch.device,
    update: int,
    policy_version: int,
    model: PolicyValueModel | None = None,
) -> str:
    policy_id = f"policy_{int(policy_version):06d}"
    guidance_payload = _model_guidance_payload(model)
    weights_path, weights_sha256 = _write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=policy_id,
        update=update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=model_state_dict,
        public_heuristic_logit_bias_scale=guidance_payload.get("public_heuristic_logit_bias_scale"),
        public_heuristic_actor_logit_bias_scale=guidance_payload.get("public_heuristic_actor_logit_bias_scale"),
    )

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    reg = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, reg)
    reg.add_snapshot(
        policy_id=policy_id,
        update=int(update),
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    _save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=reg,
    )
    return policy_id


def _require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )
    return result.stdout.strip()


def _git_commit() -> str:
    override = str(os.environ.get("WEISS_RL_GIT_COMMIT", "")).strip().lower()
    if len(override) == _GIT_COMMIT_HEX_LENGTH and all(char in "0123456789abcdef" for char in override):
        return override
    try:
        return _git_output(["rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_dirty() -> bool:
    try:
        return bool(_git_output(["status", "--short"]))
    except (OSError, subprocess.CalledProcessError):
        return False


def _start_nonce() -> int:
    return time.time_ns() & _U64_MASK


def _hardware_summary(
    learner_device: torch.device | str = "cpu",
    *,
    actor_device: torch.device | str = "cpu",
    actor_device_layout: Sequence[str] | None = None,
) -> dict[str, str | int]:
    learner_device_name = str(learner_device)
    payload: dict[str, str | int] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "learner_device": learner_device_name,
        "actor_device": str(actor_device),
    }
    if actor_device_layout:
        payload["actor_device_layout"] = ",".join(str(device_name) for device_name in actor_device_layout)
        payload["actor_device_unique_count"] = len(
            dict.fromkeys(str(device_name) for device_name in actor_device_layout)
        )
    return payload


def _scaling_request_from_config(training_config: Any) -> ScalingRequest:
    scaling = getattr(training_config, "scaling", None)
    return ScalingRequest(
        learner_parallelism=str(getattr(scaling, "learner_parallelism", "auto")),
        learner_gpu_count=str(getattr(scaling, "learner_gpu_count", "auto")),
        actor_topology=str(getattr(scaling, "actor_topology", "auto")),
        target_envs_per_gpu=int(getattr(scaling, "target_envs_per_gpu", 512)),
        min_envs_per_actor=int(getattr(scaling, "min_envs_per_actor", 32)),
        max_envs_per_actor=int(getattr(scaling, "max_envs_per_actor", 64)),
        max_actor_process_count=int(getattr(scaling, "max_actor_process_count", 64)),
        reserve_cpu_cores=int(getattr(scaling, "reserve_cpu_cores", 4)),
        learner_cpu_cores_per_gpu=int(getattr(scaling, "learner_cpu_cores_per_gpu", 2)),
        queue_depth_multiplier=int(getattr(scaling, "queue_depth_multiplier", 2)),
        ram_queue_fraction=float(getattr(scaling, "ram_queue_fraction", 0.25)),
        vram_fraction=float(getattr(scaling, "vram_fraction", 0.85)),
    )


def _resolve_autoscale_topology(
    *,
    stack: StackConfig,
    hardware_profile_name: str,
    runtime_mode: QueueRuntimeMode,
) -> ResolvedTrainingTopology:
    if stack.config.system is None or stack.config.training is None:
        raise RuntimeError("autoscale requires system and training config blocks")
    hardware = hardware_profile_from_name(hardware_profile_name)
    return resolve_training_topology(
        hardware=hardware,
        request=_scaling_request_from_config(stack.config.training),
        configured_actor_count=int(stack.config.system.actor_process_count),
        configured_envs_per_actor=int(stack.config.system.envs_per_actor),
        configured_batch_unrolls_per_update=int(stack.config.training.batch_unrolls_per_update),
        configured_queue_capacity_unrolls=int(stack.config.system.actor_queue_capacity_unrolls),
        runtime_mode=str(runtime_mode),
    )


def _manifest_actor_device_layout(
    *,
    stack: StackConfig,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
    learner_device: torch.device,
    resolved_topology: ResolvedTrainingTopology | None = None,
) -> tuple[str, ...] | None:
    if stack.config.system is None or stack.config.training is None:
        return None
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
        resolved_actor_count=None if resolved_topology is None else int(resolved_topology.actor_count),
        resolved_envs_per_actor=None if resolved_topology is None else int(resolved_topology.envs_per_actor),
        resolved_batch_unrolls_per_update=(
            None if resolved_topology is None else int(resolved_topology.batch_unrolls_per_update)
        ),
        resolved_queue_capacity_unrolls=(
            None if resolved_topology is None else int(resolved_topology.queue_capacity_unrolls)
        ),
    )
    return tuple(
        str(device_name)
        for device_name in resolve_actor_device_layout(
            stack,
            actor_count=int(runtime_config.actor_count),
            learner_device=learner_device,
            prefer_process_collectors=True,
        )
    )


def _evaluation_pinning(stack: StackConfig) -> dict[str, str | bool]:
    if stack.config.evaluation is None:
        return {}
    evaluation = stack.config.evaluation
    return {
        "eval_device": evaluation.eval_device,
        "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
        "eval_inference_mode": evaluation.eval_inference_mode,
        "seat_swap": evaluation.seat_swap,
        "legal_fingerprint_version": evaluation.legal_fingerprint_checks.version,
        "legal_fingerprint_mismatch_policy": evaluation.legal_fingerprint_checks.mismatch_policy,
    }


def _manifest_source_path(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def _apply_training_flag_overrides(
    stack: StackConfig,
    *,
    enable_profile_timers: bool,
    enable_torch_profiler: bool,
) -> StackConfig:
    training_config = stack.config.training
    if training_config is None:
        return stack
    overrides: dict[str, Any] = {}
    if enable_profile_timers and not bool(training_config.profile_timers):
        overrides["training.profile_timers"] = True
    if enable_torch_profiler and not bool(training_config.torch_profiler):
        overrides["training.torch_profiler"] = True
    return apply_stack_overrides(stack, overrides)


def _experiment_role(stack: StackConfig) -> str:
    experiment = stack.config.experiment
    return "" if experiment is None else str(experiment.role).strip()


def _is_noleague_baseline_role(role: str) -> bool:
    normalized = str(role).strip()
    return normalized == "baseline_noleague" or normalized.startswith("baseline_noleague_")


def _canonical_config_sections(config_canonical: Mapping[str, Any]) -> Mapping[str, Any]:
    config = config_canonical.get("config")
    return config if isinstance(config, Mapping) else config_canonical


def _role_from_config_canonical(config_canonical: Mapping[str, Any]) -> str:
    experiment = _canonical_config_sections(config_canonical).get("experiment", {})
    if isinstance(experiment, Mapping):
        role = str(experiment.get("role", "")).strip()
        if role:
            return role
    return ""


def _legacy_noleague_baseline_mode(config_canonical: Mapping[str, Any]) -> str:
    training_family = _canonical_config_sections(config_canonical).get("training_family_a", {})
    if isinstance(training_family, Mapping):
        return str(training_family.get("mode", "")).strip()
    return ""


def _config_marks_noleague_baseline(config_canonical: Mapping[str, Any]) -> bool:
    role = _role_from_config_canonical(config_canonical)
    if role:
        return _is_noleague_baseline_role(role)
    legacy_mode = _legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode:
        return legacy_mode == "b1_no_league"
    return False


def _assert_noleague_baseline_config(config_canonical: Mapping[str, Any]) -> None:
    role = _role_from_config_canonical(config_canonical)
    if role:
        if not _is_noleague_baseline_role(role):
            raise RuntimeError(
                f"Imported B1 baseline must come from a dedicated baseline_noleague run, got experiment.role={role!r}"
            )
        return
    legacy_mode = _legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode and legacy_mode != "b1_no_league":
        raise RuntimeError(
            "Imported B1 baseline must come from a dedicated baseline_noleague run, "
            f"got training_family_a.mode={legacy_mode!r}"
        )


def _read_optional_hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _validate_imported_snapshot_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    def _model_section_for_contract(section: Any) -> Any:
        if not isinstance(section, Mapping):
            return section
        normalized = dict(section)
        # These are per-snapshot guidance values stored in weights.pt and restored
        # when loading a policy. They do not change the state_dict contract.
        normalized.pop("public_heuristic_logit_bias_scale", None)
        normalized.pop("public_heuristic_actor_logit_bias_scale", None)
        # Schedule-only guidance fields affect future training updates, not
        # the imported snapshot tensor contract or restored snapshot behavior.
        normalized.pop("public_heuristic_logit_bias_start_updates", None)
        normalized.pop("public_heuristic_logit_bias_end_updates", None)
        normalized.pop("public_heuristic_logit_bias_final_scale", None)
        return normalized

    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    manifest_path = source_layout.manifest_path
    source_manifest = (
        _load_json_object(manifest_path, label="imported B1 manifest") if manifest_path.is_file() else None
    )
    source_config_canonical = source_manifest.get("config_canonical") if isinstance(source_manifest, dict) else None
    if isinstance(source_config_canonical, dict):
        source_config_sections = _canonical_config_sections(source_config_canonical)
        _assert_noleague_baseline_config(source_config_canonical)
        if isinstance(expected_config_canonical, dict):
            expected_config_sections = _canonical_config_sections(expected_config_canonical)
            for section_name in ("model", "environment"):
                source_section = source_config_sections.get(section_name)
                expected_section = expected_config_sections.get(section_name)
                if source_section is None or expected_section is None:
                    continue
                if section_name == "model":
                    source_section = _model_section_for_contract(source_section)
                    expected_section = _model_section_for_contract(expected_section)
                if source_section != expected_section:
                    raise RuntimeError(
                        f"Imported B1 baseline config does not match the current run for section={section_name!r}"
                    )

    if expected_spec_hash256 is not None:
        source_spec_hash = _read_optional_hash_file(source_layout.spec_hash_path)
        if source_spec_hash is not None and source_spec_hash != expected_spec_hash256:
            raise RuntimeError(
                "Imported B1 baseline spec hash does not match the current run: "
                f"source={source_spec_hash} expected={expected_spec_hash256}"
            )

    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload is missing model_state_dict: {source_run_dir}")
    source_keys = set(source_model_state_dict)
    expected_keys = set(expected_model_state_dict)
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise RuntimeError(
            "Imported B1 baseline model contract does not match the current run: "
            f"missing_keys={missing} extra_keys={extra}"
        )
    for key in sorted(expected_keys):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                "Imported B1 baseline tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def _load_snapshot_registry(path: Path) -> SnapshotRegistry:
    if not path.exists():
        raise FileNotFoundError(path)
    return SnapshotRegistry.load(path)


def _load_dev_eval_summaries(path: Path) -> dict[str, float | DevEvalPolicySummary]:
    payload = _load_json_object(path, label="dev-eval summaries")
    summaries: dict[str, float | DevEvalPolicySummary] = {}
    for policy_id, raw_summary in payload.items():
        if isinstance(raw_summary, bool):
            raise TypeError(f"dev-eval summary for {policy_id!r} cannot be a boolean")
        if isinstance(raw_summary, (int, float)):
            summaries[policy_id] = float(raw_summary)
            continue
        if not isinstance(raw_summary, dict):
            raise TypeError(
                "dev-eval summary values must be numbers or objects with aggregate_score/anchor_scores, "
                f"got {type(raw_summary).__name__} for {policy_id!r}"
            )
        aggregate_score = raw_summary.get("aggregate_score")
        if isinstance(aggregate_score, bool) or not isinstance(aggregate_score, (int, float)):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include numeric aggregate_score")
        anchor_scores = raw_summary.get("anchor_scores", {})
        if not isinstance(anchor_scores, dict) or any(not isinstance(key, str) for key in anchor_scores):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include object anchor_scores")
        summaries[policy_id] = DevEvalPolicySummary(
            policy_id=policy_id,
            aggregate_score=float(aggregate_score),
            anchor_scores=anchor_scores,
        )
    return summaries


def _selection_requires_snapshot_registry(stack: StackConfig) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    return selection.include_final_champion_snapshot or bool(selection.include_spaced_snapshots_near_percent_updates)


def _selection_requires_dev_eval_summaries(stack: StackConfig) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    fixed_slots = int(selection.include_random_legal_baseline_b0) + int(selection.include_no_league_baseline_b1)
    fixed_slots += int(selection.include_final_champion_snapshot)
    fixed_slots += len(selection.include_spaced_snapshots_near_percent_updates)
    if selection.include_heuristic_public_b2_if_exists:
        return True
    return evaluation.final_policy_set_size > fixed_slots


def _policy_set_selection(
    stack: StackConfig,
    *,
    snapshot_registry: SnapshotRegistry | None = None,
    dev_eval_summaries: Mapping[str, float | DevEvalPolicySummary] | None = None,
) -> list[str]:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return []
    selection = evaluation.final_policy_set_selection
    if selection.version != "deterministic_v1":
        raise ValueError(f"unsupported final_policy_set_selection.version: {selection.version!r}")
    return select_final_policy_set_deterministic_v1(
        snapshot_registry=snapshot_registry or SnapshotRegistry(),
        dev_eval_summaries=dev_eval_summaries or {},
        config=selection,
        final_policy_set_size=evaluation.final_policy_set_size,
    )


def _resolve_policy_set_selection(
    stack: StackConfig,
    *,
    snapshot_registry_path: Path | None = None,
    dev_eval_summaries_path: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    evaluation = stack.config.evaluation
    source_paths = {
        "snapshot_registry_json": None
        if snapshot_registry_path is None
        else _manifest_source_path(snapshot_registry_path, root=stack.root),
        "dev_eval_summaries_json": None
        if dev_eval_summaries_path is None
        else _manifest_source_path(dev_eval_summaries_path, root=stack.root),
    }
    if evaluation is None:
        return [], {"mode": "not_configured", "status": "not_configured", "source_paths": source_paths}

    snapshot_registry = None if snapshot_registry_path is None else _load_snapshot_registry(snapshot_registry_path)
    dev_eval_summaries = None if dev_eval_summaries_path is None else _load_dev_eval_summaries(dev_eval_summaries_path)

    missing_inputs: list[str] = []
    if _selection_requires_snapshot_registry(stack) and snapshot_registry is None:
        missing_inputs.append("snapshot_registry_json")
    if _selection_requires_dev_eval_summaries(stack) and dev_eval_summaries is None:
        missing_inputs.append("dev_eval_summaries_json")

    details: dict[str, Any] = {
        "mode": evaluation.final_policy_set_selection.version,
        "status": "resolved",
        "version": evaluation.final_policy_set_selection.version,
        "final_policy_set_size": evaluation.final_policy_set_size,
        "source_paths": source_paths,
        "missing_inputs": missing_inputs,
    }
    if missing_inputs:
        details["mode"] = "unresolved"
        details["status"] = "unresolved"
        details["reason"] = "deterministic final policy set inputs were not provided"
        return [], details

    policy_ids = _policy_set_selection(
        stack,
        snapshot_registry=snapshot_registry,
        dev_eval_summaries=dev_eval_summaries,
    )
    details["selected_policy_count"] = len(policy_ids)
    return policy_ids, details


def _spec_mismatch_policy(stack: StackConfig) -> str:
    return "hard_fail"


def _resolve_run_label(parser: argparse.ArgumentParser, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def _require_positive_int(name: str, value: int) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return number


def _require_positive_optional_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0, got {value}")
    return number


def _wall_clock_budget_seconds(max_wall_clock_minutes: float | None) -> float | None:
    if max_wall_clock_minutes is None:
        return None
    return float(max_wall_clock_minutes) * 60.0


def _wall_clock_budget_reached(
    *,
    start_time: float,
    max_wall_clock_seconds: float | None,
    now: float | None = None,
) -> bool:
    if max_wall_clock_seconds is None:
        return False
    current_time = time.time() if now is None else float(now)
    return (current_time - float(start_time)) >= float(max_wall_clock_seconds)


def _resolve_runtime_profile(stack: StackConfig, profile_override: str) -> str:
    if profile_override.strip():
        return profile_override.strip()
    system_config = stack.config.system
    if system_config is None:
        return "fast"
    return system_config.profile.local_iteration


def _resolve_device(stack: StackConfig, device_override: str) -> torch.device:
    requested = device_override.strip()
    if not requested:
        system_config = stack.config.system
        requested = "cpu" if system_config is None else getattr(system_config, "learner_device", "cpu")
    normalized = str(requested).strip().lower()
    if normalized in {"auto", "cuda:auto"}:
        requested = "cuda:0" if torch.cuda.is_available() and int(torch.cuda.device_count()) > 0 else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "Requested CUDA device is unavailable; falling back to cpu for the canonical single-node run.",
            file=sys.stderr,
        )
        requested = "cpu"
    return torch.device(requested)


def _resolve_seed(stack: StackConfig, seed_override: int | None) -> int:
    if seed_override is not None:
        return int(seed_override)
    reproducibility = stack.config.reproducibility
    if reproducibility is None:
        return 7
    return int(reproducibility.seed_derivation.base_seed64)


def _manifest_scaffold_only_reason(stack: StackConfig) -> str | None:
    missing_blocks: list[str] = []
    if stack.config.environment is None:
        missing_blocks.append("environment")
    if stack.config.training is None:
        missing_blocks.append("training")
    if stack.config.model is None:
        missing_blocks.append("model")
    if missing_blocks:
        return f"missing config blocks: {', '.join(missing_blocks)}"
    return None


def _runtime_training_prerequisite_failure(stack: StackConfig) -> str | None:
    if _manifest_scaffold_only_reason(stack) is not None:
        return None

    try:
        weiss_sim = importlib.import_module("weiss_sim")
    except ModuleNotFoundError:
        return "weiss_sim is not importable in the active interpreter"

    missing_runtime_attrs = [
        attr_name for attr_name in ("fast", "inspect", "rl", "PASS_ACTION_ID") if not hasattr(weiss_sim, attr_name)
    ]
    if missing_runtime_attrs:
        return f"active weiss_sim runtime is missing stepping APIs: {', '.join(missing_runtime_attrs)}"

    rl_module = weiss_sim.rl
    missing_rl_attrs = [attr_name for attr_name in ("reset_rl", "step_rl") if not hasattr(rl_module, attr_name)]
    if missing_rl_attrs:
        return f"active weiss_sim.rl is missing runtime methods: {', '.join(missing_rl_attrs)}"

    return None


def _print_manifest_only_message(reason: str) -> None:
    print("Manifest scaffold only: no learner training or rollout collection was executed.")
    print(f"Reason: {reason}.")


def _raise_runtime_prerequisite_failure(reason: str) -> None:
    raise RuntimeError(
        "Canonical simulator-backed training requires a weiss_sim runtime with stepping support. "
        f"Startup failed because {reason}."
    )


def _training_paths(run_dir: Path) -> TrainingPaths:
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.ensure_directories()
    training_dir = layout.training_dir
    checkpoints_dir = layout.training_checkpoints_dir
    logs_dir = layout.training_logs_dir
    snapshots_dir = layout.training_snapshots_dir
    return TrainingPaths(
        training_dir=training_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        snapshots_dir=snapshots_dir,
        tensorboard_dir=layout.tensorboard_dir,
        scalars_path=logs_dir / "scalars.jsonl",
        performance_log_path=layout.performance_log_path,
        latest_checkpoint_path=checkpoints_dir / _LATEST_CHECKPOINT_FILENAME,
        best_checkpoint_path=checkpoints_dir / _BEST_CHECKPOINT_FILENAME,
        checkpoint_tracker_path=checkpoints_dir / _CHECKPOINT_TRACKER_FILENAME,
    )


def _run_artifacts_from_existing_run_dir(run_dir: Path) -> RunArtifacts:
    resolved_run_dir = Path(run_dir).resolve()
    layout = ArtifactLayout.from_run_dir(resolved_run_dir)
    layout.ensure_directories()
    return RunArtifacts(
        run_dir=resolved_run_dir,
        run_dir_name=resolved_run_dir.name,
        layout=layout,
        manifest_path=layout.manifest_path,
        spec_bundle_path=layout.spec_bundle_path,
        spec_hash_path=layout.spec_hash_path,
        config_hash_path=layout.config_hash_path,
        config_json_path=layout.config_json_path,
        environment_path=layout.environment_path,
        run_summary_path=layout.run_summary_path,
        determinism_report_path=layout.determinism_report_path,
        paper_readiness_summary_path=layout.paper_readiness_summary_path,
        performance_log_path=layout.performance_log_path,
    )


def _configure_torch_threads(stack: StackConfig) -> None:
    system_config = stack.config.system
    if system_config is None:
        return
    torch.set_num_threads(int(system_config.learner_torch_threads))
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


@contextmanager
def _torch_num_threads_scope(num_threads: int | None):
    if num_threads is None:
        yield
        return
    target = int(num_threads)
    if target < 1:
        raise ValueError("num_threads must be >= 1")
    previous = int(torch.get_num_threads())
    if previous == target:
        yield
        return
    torch.set_num_threads(target)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _central_runtime_actor_torch_threads(stack: StackConfig, runtime: QueueRuntime) -> int | None:
    system_config = stack.config.system
    if system_config is None:
        return None
    if str(system_config.actor_device).strip().lower() != "cpu":
        return None
    if bool(getattr(runtime, "_use_process_collectors", False)):
        return None
    if not bool(getattr(runtime, "_use_central_batched_collection", False)):
        return None
    return int(system_config.actor_torch_threads)


def _spec_dimensions(contract: SimulatorContract) -> tuple[int, int]:
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    return observation_dim, action_dim


def _env_pool_config(stack: StackConfig, *, seed: int) -> dict[str, Any]:
    return build_env_config_from_stack(stack, seed=int(seed))


def _build_env(
    stack: StackConfig,
    *,
    profile: str,
    num_envs: int,
    seed: int,
) -> DecisionBoundaryEnv:
    env_config = _env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        env_config,
        profile=profile,  # type: ignore[arg-type]
        num_envs=num_envs,
    )
    if layout_name != "mask":
        raise RuntimeError(
            "The compatibility training path expects mask legality because ImpalaLearner consumes legal_mask. "
            f"Profile {profile!r} resolved to layout {layout_name!r}."
        )
    max_no_progress_decisions = None
    curriculum = stack.config.curriculum
    if curriculum is not None:
        raw_limit = curriculum.simulator.get("max_no_progress_decisions")
        if raw_limit is not None:
            max_no_progress_decisions = int(raw_limit)
    return DecisionBoundaryEnv(
        pool,
        legality="mask",
        engine_status_policy="hard_fail",
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
        max_no_progress_decisions=max_no_progress_decisions,
    )


def _build_ids_eval_env(
    stack: StackConfig,
    *,
    seed: int,
    pass_action_id: int,
) -> DecisionBoundaryEnv:
    env_config = _env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        env_config,
        profile="fast",
        num_envs=1,
    )
    if layout_name != "i16_legal_ids":
        raise RuntimeError(
            "Periodic dev eval requires ids-based legality for the pinned eval protocol. "
            f"Profile 'fast' resolved to layout {layout_name!r}."
        )
    max_no_progress_decisions = None
    curriculum = stack.config.curriculum
    if curriculum is not None:
        raw_limit = curriculum.simulator.get("max_no_progress_decisions")
        if raw_limit is not None:
            max_no_progress_decisions = int(raw_limit)
    return DecisionBoundaryEnv(
        pool,
        legality="ids_offsets",
        pass_action_id=pass_action_id,
        engine_status_policy="hard_fail",
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
        max_no_progress_decisions=max_no_progress_decisions,
    )


def _bootstrap_values(
    model: PolicyValueModel,
    rollout: MinimalRollout,
    final_seat_hidden: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    bootstrap_value = np.zeros((rollout.bootstrap_obs.shape[0],), dtype=np.float32)
    valid_rows = (rollout.bootstrap_actor == 0) | (rollout.bootstrap_actor == 1)
    if not np.any(valid_rows):
        return bootstrap_value

    with torch.inference_mode():
        _, value_tensor, _ = model.forward_seat_aware(
            torch.as_tensor(rollout.bootstrap_obs[valid_rows], device=device),
            torch.as_tensor(rollout.bootstrap_actor[valid_rows], device=device, dtype=torch.long),
            final_seat_hidden[valid_rows],
        )
    bootstrap_value[valid_rows] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return bootstrap_value


def _build_learner_batch(
    stack: StackConfig,
    rollout: MinimalRollout,
    bootstrap_value: np.ndarray,
    *,
    action_dim: int,
    initial_hidden_state: torch.Tensor,
    pass_action_id: int,
) -> dict[str, Any]:
    training_config = stack.config.training
    rewards_config = stack.config.rewards
    if training_config is None or rewards_config is None:
        raise RuntimeError("The canonical single-node path requires training and rewards config blocks")

    target_logp = masked_logp_from_mask(
        rollout.logits.reshape(-1, action_dim),
        rollout.legal_mask.reshape(-1, action_dim),
        rollout.actions.reshape(-1),
        pass_action_id=pass_action_id,
    ).reshape(rollout.actions.shape)

    rewards = np.asarray(rollout.rewards, dtype=np.float32)

    discounts = np.logical_not(rollout.terminated).astype(np.float32) * float(rewards_config.gamma)
    if not bool(rewards_config.truncation.bootstrap_value):
        discounts *= np.logical_not(rollout.truncated).astype(np.float32)

    values = np.concatenate([rollout.values, bootstrap_value[np.newaxis, :]], axis=0)
    vtrace_result: VTraceTargets = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        rollout.behavior_logp,
        target_logp,
        rho_bar=training_config.vtrace_rho_bar,
        c_bar=training_config.vtrace_c_bar,
    )

    return {
        "obs": rollout.obs,
        "actions": rollout.actions,
        "legal_mask": rollout.legal_mask,
        "to_play_seat": rollout.to_play_seat,
        "actor": rollout.to_play_seat,
        "initial_hidden_state": initial_hidden_state.detach().cpu().numpy(),
        "rewards": rewards,
        "discounts": discounts,
        "behavior_logp": rollout.behavior_logp,
        "behavior_logits": rollout.logits,
        "logits": rollout.logits,
        "vtrace_result": vtrace_result,
        "vtrace_rho_bar": float(training_config.vtrace_rho_bar),
        "vtrace_c_bar": float(training_config.vtrace_c_bar),
    }


def _write_scalars_record(
    *,
    scalars_path: Path,
    learner: ImpalaLearner,
    metrics: dict[str, float],
    start_time: float,
) -> None:
    wall_clock_seconds = time.time() - start_time
    record = {
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_ms": int(wall_clock_seconds * 1000),
        **metrics,
    }
    with scalars_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Cannot write a checkpoint without a learner model")

    payload = {
        "format": "minimal_train_checkpoint_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "device": str(device),
        "config_hash256": compute_config_hash256(stack),
        "spec_hash256": spec_hash256,
        "algorithm": algorithm,
        "recurrent_core": getattr(stack.config.model, "recurrent_core", None),
        "total_samples_processed": int(getattr(learner, "total_samples_processed", 0)),
        "model_state_dict": learner.model.state_dict(),
        **_model_guidance_payload(learner.model),
        "optimizer_state_dict": None if learner.optimizer is None else learner.optimizer.state_dict(),
        "grad_scaler_state_dict": (
            None if getattr(learner, "_grad_scaler", None) is None else learner._grad_scaler.state_dict()
        ),
    }
    torch.save(payload, checkpoint_path)
    return payload


def _relative_path_text(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_checkpoint_tracker(training_paths: TrainingPaths) -> dict[str, Any]:
    if not training_paths.checkpoint_tracker_path.is_file():
        return {"format": _CHECKPOINT_TRACKER_FORMAT, "latest": None, "best": None, "secondary": {}}
    payload = json.loads(training_paths.checkpoint_tracker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint tracker must be a JSON object: {training_paths.checkpoint_tracker_path}")
    payload.setdefault("format", _CHECKPOINT_TRACKER_FORMAT)
    payload.setdefault("latest", None)
    payload.setdefault("best", None)
    secondary = payload.get("secondary")
    if not isinstance(secondary, dict):
        payload["secondary"] = {}
    return payload


def _write_checkpoint_tracker(training_paths: TrainingPaths, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["format"] = _CHECKPOINT_TRACKER_FORMAT
    secondary = payload.get("secondary")
    if not isinstance(secondary, dict):
        payload["secondary"] = {}
    training_paths.checkpoint_tracker_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint_guard_log_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "checkpoint_guard.jsonl"


def _b2_disagreement_audit_requests_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / _B2_DISAGREEMENT_AUDIT_REQUESTS_FILENAME


def _build_checkpoint_record(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    artifacts: RunArtifacts,
    learner: ImpalaLearner,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias_name,
        "alias_path": _relative_path_text(alias_path, root=artifacts.run_dir),
        "source_checkpoint_path": _relative_path_text(source_checkpoint_path, root=artifacts.run_dir),
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "metric_kind": metric_kind,
        "metric_value": metric_value,
    }


def _build_secondary_checkpoint_record(
    *,
    source_checkpoint_path: Path,
    artifacts: RunArtifacts,
    update_count: int,
    policy_version: int,
    metric_kind: str,
    metric_value: float,
    aggregate_score: float | None,
    dev_eval_ineligibility_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_checkpoint_path": _relative_path_text(source_checkpoint_path, root=artifacts.run_dir),
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "metric_kind": str(metric_kind),
        "metric_value": float(metric_value),
    }
    if aggregate_score is not None and np.isfinite(float(aggregate_score)):
        record["aggregate_score"] = float(aggregate_score)
    if dev_eval_ineligibility_reasons:
        record["dev_eval_ineligibility_reasons"] = [str(reason) for reason in dev_eval_ineligibility_reasons]
    return record


def _checkpoint_secondary_records(tracker: dict[str, Any]) -> dict[str, Any]:
    secondary = tracker.get("secondary")
    if isinstance(secondary, dict):
        return secondary
    tracker["secondary"] = {}
    return cast(dict[str, Any], tracker["secondary"])


def _dev_eval_aggregate_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    aggregate_score = dev_eval_summary.get("aggregate_score")
    if isinstance(aggregate_score, (int, float)) and np.isfinite(float(aggregate_score)):
        return float(aggregate_score)
    uncertainty = dev_eval_summary.get("uncertainty")
    if isinstance(uncertainty, Mapping):
        mean_value = uncertainty.get("mean")
        if isinstance(mean_value, (int, float)) and np.isfinite(float(mean_value)):
            return float(mean_value)
    return None


def _periodic_dev_eval_anchor_weight_map(stack: StackConfig) -> dict[str, float]:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return {}
    raw_weights = getattr(evaluation, "periodic_dev_eval_anchor_weights", {}) or {}
    if not isinstance(raw_weights, Mapping):
        return {}
    weights: dict[str, float] = {}
    for anchor_name, value in raw_weights.items():
        name = str(anchor_name).strip()
        if not name:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        weight = float(value)
        if not np.isfinite(weight) or weight < 0.0:
            continue
        weights[name] = weight
    return weights


def _weighted_dev_eval_aggregate(
    anchor_scores: Mapping[str, float],
    *,
    anchor_weights: Mapping[str, float],
) -> tuple[float, dict[str, float], float]:
    if not anchor_scores:
        return 0.0, {}, 0.0
    active_weights: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for anchor_name, score in anchor_scores.items():
        weight = float(anchor_weights.get(anchor_name, 1.0))
        if weight <= 0.0:
            continue
        active_weights[str(anchor_name)] = weight
        weighted_sum += float(score) * weight
        total_weight += weight
    if total_weight <= 0.0:
        for anchor_name, score in anchor_scores.items():
            active_weights[str(anchor_name)] = 1.0
            weighted_sum += float(score)
            total_weight += 1.0
    return float(weighted_sum / total_weight), active_weights, float(total_weight)


def _league_eval_warmup_gate_status(
    stack: StackConfig,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    league = stack.config.league
    if league is None or not bool(league.enabled):
        return {"enabled": False, "open": True, "reasons": []}
    warmup = league.warmup
    if not bool(getattr(warmup, "eval_gate_enabled", False)):
        return {"enabled": False, "open": True, "reasons": []}
    reasons: list[str] = []
    if dev_eval_summary is None:
        return {"enabled": True, "open": False, "reasons": ["missing_dev_eval"]}
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    min_aggregate_score = getattr(warmup, "eval_gate_min_aggregate_score", None)
    if min_aggregate_score is not None:
        if current_score is None or float(current_score) < float(min_aggregate_score):
            reasons.append("aggregate_score")
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        anchor_scores = {}
    failed_anchors: dict[str, dict[str, float | None]] = {}
    for anchor_name, min_score in dict(getattr(warmup, "eval_gate_min_anchor_scores", {}) or {}).items():
        anchor_name_text = str(anchor_name)
        value = anchor_scores.get(anchor_name_text)
        if value is None and anchor_name_text in {"Latest recent snapshot", "Previous recent snapshot"}:
            continue
        score = float(value) if isinstance(value, (int, float)) and np.isfinite(float(value)) else None
        if score is None or score < float(min_score):
            failed_anchors[anchor_name_text] = {
                "score": score,
                "min_score": float(min_score),
            }
    if failed_anchors:
        reasons.append("anchor_scores")
    return {
        "enabled": True,
        "open": not reasons,
        "reasons": reasons,
        "failed_anchors": failed_anchors,
        "aggregate_score": current_score,
        "min_aggregate_score": min_aggregate_score,
    }


def _sync_runtime_league_eval_warmup_gate(
    *,
    runtime: QueueRuntime,
    stack: StackConfig,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = _league_eval_warmup_gate_status(stack, dev_eval_summary)
    setter = getattr(runtime, "set_league_eval_warmup_gate", None)
    if callable(setter):
        setter(open=bool(status["open"]))
    return status


def _dev_eval_surface(dev_eval_summary: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if dev_eval_summary is None:
        return {}
    surface = dev_eval_summary.get("evaluation_surface")
    if isinstance(surface, Mapping):
        return cast(Mapping[str, Any], surface)
    return {}


def _dev_eval_is_authoritative(dev_eval_summary: Mapping[str, Any] | None) -> bool:
    if dev_eval_summary is None:
        return False
    surface = _dev_eval_surface(dev_eval_summary)
    authoritative = surface.get("authoritative")
    if isinstance(authoritative, bool):
        return authoritative
    # Older summaries predate the surface contract and are canonical scalar by construction.
    return True


def _dev_eval_batched_screen_enabled(dev_eval_summary: Mapping[str, Any] | None) -> bool:
    surface = _dev_eval_surface(dev_eval_summary)
    return str(surface.get("kind", "")).strip() == "fast_batched_screen"


def _periodic_dev_eval_fast_screens_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "periodic_dev_eval_fast_screens.json"


def _persist_periodic_dev_eval_fast_screen(
    *,
    training_paths: TrainingPaths,
    payload: Mapping[str, Any],
) -> None:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    if not focal_policy_id:
        return
    path = _periodic_dev_eval_fast_screens_path(training_paths)
    summaries = _load_json_object(path, label="periodic dev-eval fast screens") if path.is_file() else {}
    summaries[focal_policy_id] = {
        "format": "periodic_dev_eval_fast_screen_v1",
        "aggregate_score": payload.get("aggregate_score"),
        "anchor_scores": dict(cast(Mapping[str, Any], payload.get("anchor_scores", {}))),
        "update_count": int(payload.get("update_count", 0)),
        "policy_version": int(payload.get("policy_version", 0)),
        "evaluation_surface": dict(_dev_eval_surface(payload)),
        "periodic_dev_eval_parallel": dict(cast(Mapping[str, Any], payload.get("periodic_dev_eval_parallel", {}))),
        "periodic_dev_eval_runtime": dict(cast(Mapping[str, Any], payload.get("periodic_dev_eval_runtime", {}))),
    }
    _write_json(path, summaries)


def _dev_eval_worst_truncation_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        worst_rate = stall_monitor.get("worst_truncation_rate")
        if isinstance(worst_rate, (int, float)) and np.isfinite(float(worst_rate)):
            return float(worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        games = summary.get("games")
        truncations = summary.get("truncations")
        if not isinstance(games, (int, float)) or not isinstance(truncations, (int, float)):
            continue
        if float(games) <= 0:
            continue
        rate = float(truncations) / float(games)
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def _summary_rate(matchup_summary: Mapping[str, Any], key: str) -> float | None:
    games = matchup_summary.get("games")
    count = matchup_summary.get(key)
    if not isinstance(games, (int, float)) or not isinstance(count, (int, float)):
        return None
    if float(games) <= 0.0:
        return None
    return float(count) / float(games)


def _dev_eval_worst_reason_rate(
    dev_eval_summary: Mapping[str, Any] | None,
    *,
    summary_key: str,
    stall_monitor_key: str,
) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        worst_rate = stall_monitor.get(stall_monitor_key)
        if isinstance(worst_rate, (int, float)) and np.isfinite(float(worst_rate)):
            return float(worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        rate = _summary_rate(summary, summary_key)
        if rate is None:
            continue
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def _dev_eval_worst_no_progress_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return _dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="no_progress_timeouts",
        stall_monitor_key="worst_no_progress_timeout_rate",
    )


def _dev_eval_worst_natural_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return _dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="natural_timeouts",
        stall_monitor_key="worst_natural_timeout_rate",
    )


def _dev_eval_worst_stall_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    no_progress_rate = _dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    if no_progress_rate is not None:
        return no_progress_rate
    return _dev_eval_worst_truncation_rate(dev_eval_summary)


def _dev_eval_confidence_stats(dev_eval_summary: Mapping[str, Any] | None) -> dict[str, float | None]:
    stats = {
        "min_prob_gt_half": None,
        "max_prob_lt_half": None,
        "max_ci_half_width": None,
    }
    if dev_eval_summary is None:
        return stats
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return stats
    min_prob_gt_half: float | None = None
    max_prob_lt_half: float | None = None
    max_ci_half_width: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        uncertainty = anchor_payload.get("uncertainty")
        if not isinstance(uncertainty, Mapping):
            continue
        prob_gt_half = uncertainty.get("prob_gt_half")
        prob_lt_half = uncertainty.get("prob_lt_half")
        ci_half_width = uncertainty.get("ci_half_width")
        if isinstance(prob_gt_half, (int, float)) and np.isfinite(float(prob_gt_half)):
            min_prob_gt_half = (
                float(prob_gt_half) if min_prob_gt_half is None else min(min_prob_gt_half, float(prob_gt_half))
            )
        if isinstance(prob_lt_half, (int, float)) and np.isfinite(float(prob_lt_half)):
            max_prob_lt_half = (
                float(prob_lt_half) if max_prob_lt_half is None else max(max_prob_lt_half, float(prob_lt_half))
            )
        if isinstance(ci_half_width, (int, float)) and np.isfinite(float(ci_half_width)):
            max_ci_half_width = (
                float(ci_half_width) if max_ci_half_width is None else max(max_ci_half_width, float(ci_half_width))
            )
    stats["min_prob_gt_half"] = min_prob_gt_half
    stats["max_prob_lt_half"] = max_prob_lt_half
    stats["max_ci_half_width"] = max_ci_half_width
    return stats


def _extract_anchor_payload(dev_eval_summary: Mapping[str, Any] | None, anchor_name: str) -> Mapping[str, Any] | None:
    if dev_eval_summary is None:
        return None
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    anchor_payload = anchors.get(anchor_name)
    if not isinstance(anchor_payload, Mapping):
        return None
    return cast(Mapping[str, Any], anchor_payload)


def _extract_anchor_score(dev_eval_summary: Mapping[str, Any] | None, anchor_name: str) -> float | None:
    if dev_eval_summary is None:
        return None
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if isinstance(anchor_scores, Mapping):
        score = anchor_scores.get(anchor_name)
        if isinstance(score, (int, float)) and np.isfinite(float(score)):
            return float(score)
    anchor_payload = _extract_anchor_payload(dev_eval_summary, anchor_name)
    if anchor_payload is None:
        return None
    uncertainty = anchor_payload.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        return None
    mean_value = uncertainty.get("mean")
    if isinstance(mean_value, (int, float)) and np.isfinite(float(mean_value)):
        return float(mean_value)
    return None


def _extract_anchor_summary(dev_eval_summary: Mapping[str, Any] | None, anchor_name: str) -> Mapping[str, Any] | None:
    anchor_payload = _extract_anchor_payload(dev_eval_summary, anchor_name)
    if anchor_payload is None:
        return None
    summary = anchor_payload.get("summary")
    if not isinstance(summary, Mapping):
        return None
    return cast(Mapping[str, Any], summary)


def _extract_anchor_uncertainty(
    dev_eval_summary: Mapping[str, Any] | None,
    anchor_name: str,
) -> Mapping[str, Any] | None:
    anchor_payload = _extract_anchor_payload(dev_eval_summary, anchor_name)
    if anchor_payload is None:
        return None
    uncertainty = anchor_payload.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        return None
    return cast(Mapping[str, Any], uncertainty)


def _summary_fraction(summary: Mapping[str, Any] | None, *, numerator_key: str, denominator_key: str) -> float | None:
    if summary is None:
        return None
    numerator = summary.get(numerator_key)
    denominator = summary.get(denominator_key)
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _b2_recent_scores_from_persisted_summaries(
    summaries: Mapping[str, Any],
    *,
    current_policy_id: str,
) -> list[float]:
    scored_entries: list[tuple[int, float]] = []
    for policy_id, raw_entry in summaries.items():
        if policy_id == current_policy_id or not isinstance(raw_entry, Mapping):
            continue
        update_count = raw_entry.get("update_count")
        b2_payload = raw_entry.get("b2")
        score: float | None = None
        if isinstance(b2_payload, Mapping):
            raw_score = b2_payload.get("score")
            if isinstance(raw_score, (int, float)) and np.isfinite(float(raw_score)):
                score = float(raw_score)
        if score is None:
            raw_anchor_scores = raw_entry.get("anchor_scores")
            if isinstance(raw_anchor_scores, Mapping):
                raw_score = raw_anchor_scores.get(HEURISTIC_PUBLIC_POLICY_ID)
                if isinstance(raw_score, (int, float)) and np.isfinite(float(raw_score)):
                    score = float(raw_score)
        if score is None:
            continue
        scored_entries.append((0 if not isinstance(update_count, int) else int(update_count), score))
    scored_entries.sort(key=lambda item: item[0])
    return [score for _update_count, score in scored_entries[-(_B2_FLATLINE_WINDOW - 1) :]]


def _build_b2_warning_flags(
    *,
    current_score: float | None,
    current_summary: Mapping[str, Any] | None,
    recent_scores: Sequence[float],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if current_score is None:
        return warnings

    score_window = [*recent_scores, float(current_score)]
    if (
        len(score_window) >= _B2_FLATLINE_WINDOW
        and max(score_window) - min(score_window) <= _B2_FLATLINE_MAX_DELTA
        and float(current_score) <= _B2_FLATLINE_LOW_SCORE
    ):
        warnings.append(
            {
                "kind": "b2_flatline_v1",
                "recent_eval_count": len(score_window),
                "min_score": float(min(score_window)),
                "max_score": float(max(score_window)),
                "score_delta": float(max(score_window) - min(score_window)),
            }
        )

    main_move_rate = _summary_fraction(current_summary, numerator_key="main_move_actions", denominator_key="total_actions")
    pass_nonpass_rate = _summary_fraction(
        current_summary,
        numerator_key="pass_with_nonpass_available",
        denominator_key="total_actions",
    )
    max_consecutive_main_moves = None if current_summary is None else current_summary.get("max_consecutive_main_moves")
    if float(current_score) <= _B2_ACTION_WARNING_SCORE_THRESHOLD and (
        (main_move_rate is not None and main_move_rate >= _B2_ACTION_WARNING_MAIN_MOVE_RATE)
        or (pass_nonpass_rate is not None and pass_nonpass_rate >= _B2_ACTION_WARNING_PASS_NONPASS_RATE)
    ):
        warning_payload: dict[str, Any] = {
            "kind": "b2_action_family_warning_v1",
            "score": float(current_score),
        }
        if main_move_rate is not None:
            warning_payload["main_move_rate"] = float(main_move_rate)
        if pass_nonpass_rate is not None:
            warning_payload["pass_with_nonpass_available_rate"] = float(pass_nonpass_rate)
        if isinstance(max_consecutive_main_moves, (int, float)):
            warning_payload["max_consecutive_main_moves"] = int(max_consecutive_main_moves)
        warnings.append(warning_payload)
    return warnings


def _build_periodic_dev_eval_summary_record(
    *,
    payload: Mapping[str, Any],
    prior_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    anchor_scores = dict(cast(Mapping[str, Any], payload.get("anchor_scores", {})))
    record: dict[str, Any] = {
        "format": _PERIODIC_DEV_EVAL_SUMMARY_FORMAT,
        "aggregate_score": float(payload.get("aggregate_score", 0.0)),
        "anchor_scores": anchor_scores,
        "update_count": int(payload.get("update_count", 0)),
        "policy_version": int(payload.get("policy_version", 0)),
    }
    for optional_key in (
        "uncertainty",
        "periodic_dev_eval_parallel",
        "stall_monitor",
        "evaluation_surface",
        "aggregate_weighting",
    ):
        optional_payload = payload.get(optional_key)
        if isinstance(optional_payload, Mapping):
            record[optional_key] = dict(optional_payload)
    unweighted_aggregate_score = payload.get("unweighted_aggregate_score")
    if isinstance(unweighted_aggregate_score, (int, float)):
        record["unweighted_aggregate_score"] = float(unweighted_aggregate_score)
    anchors = payload.get("anchors")
    if isinstance(anchors, Mapping):
        record["anchors"] = dict(cast(Mapping[str, Any], anchors))

    b2_summary = _extract_anchor_summary(payload, HEURISTIC_PUBLIC_POLICY_ID)
    b2_uncertainty = _extract_anchor_uncertainty(payload, HEURISTIC_PUBLIC_POLICY_ID)
    b2_score = _extract_anchor_score(payload, HEURISTIC_PUBLIC_POLICY_ID)
    recent_b2_scores = _b2_recent_scores_from_persisted_summaries(prior_summaries, current_policy_id=focal_policy_id)
    b2_warning_flags = _build_b2_warning_flags(
        current_score=b2_score,
        current_summary=b2_summary,
        recent_scores=recent_b2_scores,
    )
    if b2_score is not None or b2_summary is not None or b2_uncertainty is not None:
        record["b2"] = {
            "available": True,
            "score": None if b2_score is None else float(b2_score),
            "summary": None if b2_summary is None else dict(b2_summary),
            "uncertainty": None if b2_uncertainty is None else dict(b2_uncertainty),
            "warning_flags": b2_warning_flags,
        }

    warning_flags: list[dict[str, Any]] = [*b2_warning_flags]
    record["warning_flags"] = warning_flags
    return record


def _dev_eval_ineligibility_reasons(
    stack: StackConfig,
    *,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if dev_eval_summary is None:
        return ("missing",)
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return ("missing_score",)
    if not _dev_eval_is_authoritative(dev_eval_summary):
        return ("non_authoritative",)
    curriculum = stack.config.curriculum
    if curriculum is None:
        return ()
    reasons: list[str] = []
    if curriculum.stall_monitor.enabled:
        worst_rate = _dev_eval_worst_stall_rate(dev_eval_summary)
        if worst_rate is not None and worst_rate >= float(curriculum.stall_monitor.truncation_rate_threshold):
            reasons.append("truncation")
    checkpoint_guard = curriculum.checkpoint_guard
    if checkpoint_guard.enabled:
        confidence = _dev_eval_confidence_stats(dev_eval_summary)
        min_prob_gt_half = confidence["min_prob_gt_half"]
        max_ci_half_width = confidence["max_ci_half_width"]
        if min_prob_gt_half is not None and (
            float(min_prob_gt_half) < float(checkpoint_guard.promote_min_prob_gt_half)
        ):
            max_prob_lt_half = confidence["max_prob_lt_half"]
            tolerated_prob_lt_half = max(0.0, 1.0 - float(checkpoint_guard.promote_min_prob_gt_half))
            if max_prob_lt_half is None or float(max_prob_lt_half) > tolerated_prob_lt_half:
                reasons.append("confidence_prob")
        if max_ci_half_width is not None and (
            float(max_ci_half_width) > float(checkpoint_guard.promote_max_ci_half_width)
        ):
            reasons.append("confidence_ci")
    return tuple(reasons)


def _dev_eval_metric_eligible(stack: StackConfig, *, dev_eval_summary: Mapping[str, Any] | None) -> bool:
    return not _dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)


def _confirmatory_dev_eval_target_pairs(stack: StackConfig) -> int:
    evaluation = _evaluation_config_or_raise(stack)
    base_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    max_pairs = int(evaluation.final_matrix_stage2_adaptive_max_paired_seeds)
    return max(base_pairs, min(max_pairs, max(32, base_pairs * 4)))


def _expand_periodic_dev_eval_paired_seeds(
    base_paired_seeds: Sequence[int],
    *,
    requested_pairs: int,
    seed_file_sha256: str,
    update_count: int,
    policy_version: int,
    scope: str,
) -> list[int]:
    requested_pairs_i = int(requested_pairs)
    paired_seeds = [int(seed) for seed in base_paired_seeds[:requested_pairs_i]]
    seen = set(paired_seeds)
    extra_index = 0
    while len(paired_seeds) < requested_pairs_i:
        derived_seed = (
            stable_hash64(
                canonical_json_bytes(
                    {
                        "kind": "periodic_dev_eval_confirmatory_seed_v1",
                        "scope": str(scope),
                        "seed_file_sha256": str(seed_file_sha256),
                        "update_count": int(update_count),
                        "policy_version": int(policy_version),
                        "extra_index": int(extra_index),
                    }
                )
            )
            & _U64_MASK
        )
        extra_index += 1
        if derived_seed in seen:
            continue
        paired_seeds.append(int(derived_seed))
        seen.add(int(derived_seed))
    return paired_seeds


def _confirmatory_dev_eval_request(
    *,
    stack: StackConfig,
    existing_best_record: Mapping[str, Any] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    reasons = _dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)
    if any(reason not in {"confidence_prob", "confidence_ci"} for reason in reasons):
        return None
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if float(current_score) < float(checkpoint_guard.min_best_score):
        return None

    existing_metric_kind = ""
    existing_metric_value: float | None = None
    score_shortfall = 0.0
    if existing_best_record is not None:
        existing_metric_kind = str(existing_best_record.get("metric_kind", "")).strip()
        raw_existing_metric_value = existing_best_record.get("metric_value")
        if isinstance(raw_existing_metric_value, (int, float)) and np.isfinite(float(raw_existing_metric_value)):
            existing_metric_value = float(raw_existing_metric_value)
            score_shortfall = max(0.0, existing_metric_value - float(current_score))
    if (
        existing_metric_kind == "dev_eval_mean"
        and existing_metric_value is not None
        and score_shortfall > 0.0
        and score_shortfall > 2.0 * float(checkpoint_guard.rollback_score_margin)
    ):
        return None

    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    confirmatory_reasons: list[str] = []
    prob_shortfall = 0.0
    if "confidence_prob" in reasons:
        min_prob_gt_half = confidence["min_prob_gt_half"]
        if min_prob_gt_half is None:
            return None
        prob_shortfall = max(0.0, float(checkpoint_guard.promote_min_prob_gt_half) - float(min_prob_gt_half))
        if prob_shortfall <= _CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL:
            confirmatory_reasons.append("confidence_prob")
    ci_excess = 0.0
    if "confidence_ci" in reasons:
        max_ci_half_width = confidence["max_ci_half_width"]
        if max_ci_half_width is None:
            return None
        ci_excess = max(0.0, float(max_ci_half_width) - float(checkpoint_guard.promote_max_ci_half_width))
        if ci_excess <= _CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
            confirmatory_reasons.append("confidence_ci")
    if (
        existing_metric_kind == "dev_eval_mean"
        and existing_metric_value is not None
        and score_shortfall > 0.0
        and score_shortfall <= 2.0 * float(checkpoint_guard.rollback_score_margin)
    ):
        confirmatory_reasons.append("score_drop")
    if not confirmatory_reasons:
        return None
    if prob_shortfall > _CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL:
        return None
    if ci_excess > _CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
        return None

    return {
        "reasons": confirmatory_reasons,
        "current_score": float(current_score),
        "existing_best_score": existing_metric_value,
        "prob_shortfall": prob_shortfall,
        "ci_excess": ci_excess,
        "target_pairs": _confirmatory_dev_eval_target_pairs(stack),
    }


def _checkpoint_candidate_metric(
    *,
    stack: StackConfig,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str | None, float | None]:
    if _dev_eval_metric_eligible(stack, dev_eval_summary=dev_eval_summary):
        aggregate_score = _dev_eval_aggregate_score(dev_eval_summary)
        if aggregate_score is not None:
            return "dev_eval_mean", aggregate_score
    evaluation = stack.config.evaluation
    if evaluation is not None and int(evaluation.periodic_dev_eval_interval_updates) > 0:
        return None, None
    if latest_metrics is not None:
        loss_value = latest_metrics.get("loss")
        if isinstance(loss_value, (int, float)) and np.isfinite(float(loss_value)):
            return "training_loss", float(loss_value)
    return None, None


def _should_promote_best_checkpoint(
    *,
    existing_record: Mapping[str, Any] | None,
    candidate_kind: str | None,
    candidate_value: float | None,
) -> bool:
    if candidate_kind is None:
        return False
    if existing_record is None:
        return True
    existing_kind = existing_record.get("metric_kind")
    existing_value = existing_record.get("metric_value")
    if candidate_kind == "dev_eval_mean":
        if existing_kind != "dev_eval_mean":
            return True
        if not isinstance(existing_value, (int, float)):
            return True
        return float(candidate_value) > float(existing_value)
    if candidate_kind == "training_loss":
        if existing_kind == "dev_eval_mean":
            return False
        if not isinstance(existing_value, (int, float)):
            return True
        return float(candidate_value) < float(existing_value)
    return False


def _should_update_secondary_b2_record(
    *,
    existing_record: Mapping[str, Any] | None,
    candidate_b2_score: float,
    candidate_aggregate_score: float | None,
    update_count: int,
    policy_version: int,
) -> bool:
    if existing_record is None:
        return True
    existing_metric = existing_record.get("metric_value")
    if not isinstance(existing_metric, (int, float)) or not np.isfinite(float(existing_metric)):
        return True
    existing_b2_score = float(existing_metric)
    if float(candidate_b2_score) > existing_b2_score:
        return True
    if float(candidate_b2_score) < existing_b2_score:
        return False

    existing_aggregate = existing_record.get("aggregate_score")
    if (
        candidate_aggregate_score is not None
        and isinstance(existing_aggregate, (int, float))
        and np.isfinite(float(existing_aggregate))
    ):
        if float(candidate_aggregate_score) > float(existing_aggregate):
            return True
        if float(candidate_aggregate_score) < float(existing_aggregate):
            return False

    existing_update = existing_record.get("update_count")
    if isinstance(existing_update, int) and int(update_count) != int(existing_update):
        return int(update_count) > int(existing_update)
    existing_version = existing_record.get("policy_version")
    if isinstance(existing_version, int) and int(policy_version) != int(existing_version):
        return int(policy_version) > int(existing_version)
    return False


def _update_secondary_b2_checkpoint_record(
    *,
    tracker: dict[str, Any],
    stack: StackConfig,
    artifacts: RunArtifacts,
    source_checkpoint_path: Path,
    update_count: int,
    policy_version: int,
    dev_eval_summary: Mapping[str, Any] | None,
) -> None:
    b2_score = _extract_anchor_score(dev_eval_summary, HEURISTIC_PUBLIC_POLICY_ID)
    if b2_score is None:
        return
    aggregate_score = _dev_eval_aggregate_score(dev_eval_summary)
    secondary_records = _checkpoint_secondary_records(tracker)
    existing_record = secondary_records.get("best_b2")
    existing_mapping = existing_record if isinstance(existing_record, Mapping) else None
    if not _should_update_secondary_b2_record(
        existing_record=cast(Mapping[str, Any] | None, existing_mapping),
        candidate_b2_score=float(b2_score),
        candidate_aggregate_score=aggregate_score,
        update_count=int(update_count),
        policy_version=int(policy_version),
    ):
        return
    secondary_records["best_b2"] = _build_secondary_checkpoint_record(
        source_checkpoint_path=source_checkpoint_path,
        artifacts=artifacts,
        update_count=int(update_count),
        policy_version=int(policy_version),
        metric_kind="b2_score",
        metric_value=float(b2_score),
        aggregate_score=aggregate_score,
        dev_eval_ineligibility_reasons=_dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary),
    )


def _publish_checkpoint_aliases(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tracker = _load_checkpoint_tracker(training_paths)

    shutil.copy2(checkpoint_path, training_paths.latest_checkpoint_path)
    latest_kind, latest_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )
    latest_record = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind=latest_kind,
        metric_value=latest_value,
    )
    tracker["latest"] = latest_record

    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        best_record = None
    should_update_best = latest_kind is not None and (
        best_record is None
        or _should_promote_best_checkpoint(
            existing_record=cast(Mapping[str, Any], best_record),
            candidate_kind=latest_kind,
            candidate_value=latest_value,
        )
    )
    if should_update_best:
        shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
        tracker["best"] = _build_checkpoint_record(
            alias_name="best",
            alias_path=training_paths.best_checkpoint_path,
            source_checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            learner=learner,
            metric_kind=latest_kind,
            metric_value=latest_value,
        )

    _update_secondary_b2_checkpoint_record(
        tracker=tracker,
        stack=stack,
        artifacts=artifacts,
        source_checkpoint_path=checkpoint_path,
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        dev_eval_summary=dev_eval_summary,
    )
    _write_checkpoint_tracker(training_paths, tracker)
    return tracker


def _append_checkpoint_guard_event(training_paths: TrainingPaths, payload: Mapping[str, Any]) -> None:
    path = _checkpoint_guard_log_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _append_b2_disagreement_audit_request(training_paths: TrainingPaths, payload: Mapping[str, Any]) -> None:
    path = _b2_disagreement_audit_requests_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _run_stack_config_path(artifacts: RunArtifacts) -> Path | None:
    if not artifacts.run_summary_path.is_file():
        return None
    run_summary = _load_json_object(artifacts.run_summary_path, label="run summary")
    raw_path = run_summary.get("stack_config_path")
    if not isinstance(raw_path, str) or not str(raw_path).strip():
        return None
    return Path(raw_path)


def _dev_eval_has_confidence_only_block(dev_eval_summary: Mapping[str, Any] | None, *, stack: StackConfig) -> bool:
    reasons = _dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)
    return bool(reasons) and all(reason in {"confidence_prob", "confidence_ci"} for reason in reasons)


def _maybe_request_b2_disagreement_audit(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if dev_eval_summary is None:
        return None
    if not _dev_eval_is_authoritative(dev_eval_summary):
        return None
    b2_payload = _extract_anchor_payload(dev_eval_summary, HEURISTIC_PUBLIC_POLICY_ID)
    if b2_payload is None:
        return None
    evaluation_context = b2_payload.get("evaluation_context")
    if not isinstance(evaluation_context, Mapping):
        return None
    episodes_rel = evaluation_context.get("episodes_path")
    if not isinstance(episodes_rel, str) or not str(episodes_rel).strip():
        return None
    episodes_path = artifacts.run_dir / str(episodes_rel)
    if not episodes_path.is_file():
        return None

    triggers: list[str] = []
    b2_record = dev_eval_summary.get("b2")
    if isinstance(b2_record, Mapping):
        warning_flags = b2_record.get("warning_flags")
        if isinstance(warning_flags, Sequence):
            for warning in warning_flags:
                if not isinstance(warning, Mapping):
                    continue
                if str(warning.get("kind", "")).strip() == "b2_flatline_v1":
                    triggers.append("b2_flatline")
                    break
    if _dev_eval_has_confidence_only_block(dev_eval_summary, stack=stack):
        triggers.append("confidence_only_gate")
    if not triggers:
        return None

    canonical_stack_config_path = artifacts.run_dir / "config_canonical.json"
    stack_config_path = canonical_stack_config_path if canonical_stack_config_path.is_file() else _run_stack_config_path(artifacts)
    update_count = int(dev_eval_summary.get("update_count", 0))
    policy_version = int(dev_eval_summary.get("policy_version", 0))
    audit_policy_id = f"policy_{policy_version:06d}" if policy_version > 0 else str(dev_eval_summary.get("policy_id", ""))
    output_run_dir = artifacts.run_dir / "eval" / "b2_disagreement_audit" / f"update_{update_count}"
    command: list[str] = []
    if stack_config_path is not None:
        command = [
            sys.executable,
            "python/scripts/b2_disagreement_audit.py",
            "--stack-config",
            stack_config_path.as_posix(),
            "--run-dir",
            artifacts.run_dir.as_posix(),
            "--output-run-dir",
            output_run_dir.as_posix(),
            "--episodes-jsonl",
            episodes_path.as_posix(),
            "--policy-id",
            audit_policy_id,
            "--summary-json",
            (output_run_dir / "audit" / "summary.json").as_posix(),
        ]

    payload = {
        "format": "b2_disagreement_audit_request_v1",
        "event_kind": "b2_disagreement_audit_requested_v1",
        "trigger_reasons": list(dict.fromkeys(triggers)),
        "update_count": update_count,
        "policy_version": policy_version,
        "policy_id": str(dev_eval_summary.get("policy_id", "")),
        "audit_policy_id": audit_policy_id,
        "b2_score": _extract_anchor_score(dev_eval_summary, HEURISTIC_PUBLIC_POLICY_ID),
        "episodes_path": _relative_path_text(episodes_path, root=artifacts.run_dir),
        "output_run_dir": _relative_path_text(output_run_dir, root=artifacts.run_dir),
        "command": command,
    }
    _append_b2_disagreement_audit_request(training_paths, payload)
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _maybe_log_structured_mainmove_guard(
    *,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if latest_metrics is None:
        return None
    top1_rate = latest_metrics.get("structured_main_move_0_2_top1_rate")
    move_share = latest_metrics.get("structured_main_move_share_when_play_available")
    if top1_rate is None or move_share is None:
        return None
    if not np.isfinite(float(top1_rate)) or not np.isfinite(float(move_share)):
        return None
    if float(top1_rate) < 0.15 and float(move_share) < 0.35:
        return None

    aggregate_score = _dev_eval_aggregate_score(dev_eval_summary) if dev_eval_summary is not None else None
    b2_score = _extract_structured_guard_b2_anchor_score(dev_eval_summary)
    if b2_score is not None and float(b2_score) > 0.10:
        return None
    if b2_score is None and aggregate_score is not None and float(aggregate_score) > 0.40:
        return None

    payload = {
        "format": "checkpoint_guard_event_v1",
        "event_kind": "structured_mainmove_warning_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "structured_main_move_0_2_top1_rate": float(top1_rate),
        "structured_main_move_share_when_play_available": float(move_share),
        "dev_eval_aggregate_score": None if aggregate_score is None else float(aggregate_score),
        "b2_anchor_score": None if b2_score is None else float(b2_score),
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _extract_structured_guard_b2_anchor_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        return None
    for key, value in anchor_scores.items():
        key_text = str(key).strip().lower()
        if "b2" not in key_text:
            continue
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def _demote_registry_champions_newer_than(training_paths: TrainingPaths, *, update_count: int) -> list[str]:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return []
    registry = SnapshotRegistry.load(registry_path)
    removed = registry.demote_champions_newer_than(int(update_count))
    if removed:
        registry.save(registry_path)
    return removed


def _is_current_run_train_snapshot_for_rollback(training_paths: TrainingPaths, snapshot: SnapshotMeta) -> bool:
    policy_id = str(snapshot.policy_id).strip()
    if not policy_id.startswith("policy_"):
        return False
    metadata_path = training_paths.snapshots_dir / policy_id / SNAPSHOT_METADATA_FILENAME
    if metadata_path.is_file():
        try:
            metadata = _load_json_object(metadata_path, label="snapshot metadata")
        except Exception:
            metadata = {}
        if isinstance(metadata, Mapping) and (
            "imported_from_run_dir" in metadata
            or "imported_from_policy_id" in metadata
            or bool(metadata.get("seeded_from_external_registry", False))
        ):
            return False
    return True


def _reject_registry_snapshots_newer_than(training_paths: TrainingPaths, *, update_count: int) -> list[str]:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return []
    registry = SnapshotRegistry.load(registry_path)
    update_count_i = int(update_count)
    rejected: list[str] = []
    for snapshot in registry.snapshots:
        if int(snapshot.update) <= update_count_i:
            continue
        if not _is_current_run_train_snapshot_for_rollback(training_paths, snapshot):
            continue
        registry.reject_snapshot(snapshot.policy_id)
        rejected.append(snapshot.policy_id)
    if rejected:
        registry.save(registry_path)
    return rejected


def _best_checkpoint_record(training_paths: TrainingPaths) -> Mapping[str, Any] | None:
    tracker = _load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    return best_record if isinstance(best_record, Mapping) else None


def _restore_checkpoint_to_latest_alias(
    *,
    checkpoint_path: Path,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_counters: bool = True,
) -> ResumeCheckpoint:
    resume_state = _restore_learner_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
        restore_counters=restore_counters,
    )
    shutil.copy2(checkpoint_path, training_paths.latest_checkpoint_path)
    return resume_state


def _resolve_resume_checkpoint_path(
    *,
    resume_from: str,
    resume_run_dir: Path | None,
) -> Path | None:
    normalized = str(resume_from).strip()
    if not normalized:
        if resume_run_dir is None:
            return None
        normalized = "latest"
    alias_name = normalized.lower()
    if alias_name in {"latest", "best"}:
        if resume_run_dir is None:
            raise ValueError("--resume-from latest|best requires --resume-run-dir")
        filename = _LATEST_CHECKPOINT_FILENAME if alias_name == "latest" else _BEST_CHECKPOINT_FILENAME
        checkpoint_path = Path(resume_run_dir).resolve() / "training" / "checkpoints" / filename
    else:
        checkpoint_path = Path(normalized).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _infer_seed_snapshot_run_dir_from_resume_checkpoint(
    *,
    stack: StackConfig,
    resume_checkpoint_path: Path | None,
    resume_run_dir: Path | None,
) -> Path | None:
    if resume_checkpoint_path is None or resume_run_dir is not None:
        return None
    league = stack.config.league
    if league is None or not bool(league.enabled):
        return None
    checkpoint_path = Path(resume_checkpoint_path).resolve()
    checkpoint_dir = checkpoint_path.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    source_run_dir = training_dir.parent
    registry_path = source_run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    if not registry_path.is_file():
        return None
    return source_run_dir


def _infer_run_dir_from_checkpoint_path(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    resolved = Path(checkpoint_path).resolve()
    checkpoint_dir = resolved.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    return training_dir.parent


def _seed_snapshot_import_max_update(
    *,
    resume_state: ResumeCheckpoint | None,
    seed_snapshot_run_dir: Path | None,
    seed_snapshot_run_dir_auto_inferred: bool,
) -> int | None:
    if resume_state is None or seed_snapshot_run_dir is None:
        return None
    if not bool(seed_snapshot_run_dir_auto_inferred):
        return None
    return int(resume_state.update_count)


def _resolve_resume_source_checkpoint_path(
    *,
    source_run_dir: Path,
    record: Mapping[str, Any],
    key: str,
) -> Path | None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = source_run_dir / path
    return path.resolve()


def _seed_checkpoint_tracker_from_resume_best(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    resume_checkpoint_path: Path,
) -> dict[str, Any] | None:
    target_tracker = _load_checkpoint_tracker(training_paths)
    if isinstance(target_tracker.get("best"), Mapping):
        return None

    checkpoint_path = Path(resume_checkpoint_path).resolve()
    checkpoint_dir = checkpoint_path.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    source_run_dir = training_dir.parent
    source_tracker_path = source_run_dir / "training" / "checkpoints" / _CHECKPOINT_TRACKER_FILENAME
    if not source_tracker_path.is_file():
        return None
    source_tracker = json.loads(source_tracker_path.read_text(encoding="utf-8"))
    if not isinstance(source_tracker, Mapping):
        return None
    source_best = source_tracker.get("best")
    if not isinstance(source_best, Mapping):
        return None
    metric_kind = source_best.get("metric_kind")
    metric_value = source_best.get("metric_value")
    if str(metric_kind).strip() != "dev_eval_mean":
        return None
    if not isinstance(metric_value, (int, float)) or not np.isfinite(float(metric_value)):
        return None

    candidate_paths = tuple(
        path
        for path in (
            _resolve_resume_source_checkpoint_path(
                source_run_dir=source_run_dir,
                record=source_best,
                key="source_checkpoint_path",
            ),
            _resolve_resume_source_checkpoint_path(
                source_run_dir=source_run_dir,
                record=source_best,
                key="alias_path",
            ),
        )
        if path is not None
    )
    if checkpoint_path not in candidate_paths:
        return None

    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(checkpoint_payload, Mapping):
        return None
    checkpoint_config_hash = str(checkpoint_payload.get("config_hash256", "")).strip().lower()
    if checkpoint_config_hash != compute_config_hash256(stack):
        return None

    update_count = source_best.get("update_count")
    policy_version = source_best.get("policy_version")
    if not isinstance(update_count, int) or not isinstance(policy_version, int):
        return None

    shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
    seeded_record = {
        "alias": "best",
        "alias_path": _relative_path_text(training_paths.best_checkpoint_path, root=artifacts.run_dir),
        "source_checkpoint_path": _relative_path_text(checkpoint_path, root=artifacts.run_dir),
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "metric_kind": "dev_eval_mean",
        "metric_value": float(metric_value),
        "seeded_from_run_dir": source_run_dir.as_posix(),
    }
    target_tracker["best"] = seeded_record
    _write_checkpoint_tracker(training_paths, target_tracker)
    return seeded_record


def _load_resume_checkpoint_dev_eval_summary(
    *,
    stack: StackConfig,
    resume_checkpoint_path: Path,
    update_count: int,
    allow_config_hash_mismatch: bool = False,
) -> dict[str, Any] | None:
    checkpoint_path = Path(resume_checkpoint_path).resolve()
    checkpoint_dir = checkpoint_path.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(checkpoint_payload, Mapping):
        return None
    checkpoint_config_hash = str(checkpoint_payload.get("config_hash256", "")).strip().lower()
    if checkpoint_config_hash != compute_config_hash256(stack) and not bool(allow_config_hash_mismatch):
        return None

    source_run_dir = training_dir.parent
    for artifact_dir_name in ("dev_eval_confirmatory", "dev_eval"):
        summary_path = source_run_dir / "eval" / artifact_dir_name / f"update_{int(update_count)}" / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and _dev_eval_is_authoritative(payload):
            return payload
    return None


def _restore_learner_from_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_counters: bool = True,
    restore_optimizer_state: bool = True,
    allow_config_hash_mismatch: bool = False,
) -> ResumeCheckpoint:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict: {checkpoint_path}")
    if str(payload.get("format", "")).strip() != "minimal_train_checkpoint_v1":
        raise RuntimeError(f"unsupported checkpoint format in {checkpoint_path}")
    payload_config_hash = str(payload.get("config_hash256", "")).strip().lower()
    expected_config_hash = compute_config_hash256(stack)
    if payload_config_hash != expected_config_hash:
        if not allow_config_hash_mismatch:
            raise RuntimeError(
                f"checkpoint config hash mismatch for {checkpoint_path}: expected {expected_config_hash}, got {payload_config_hash}"
            )
        print(
            "Warning: resuming checkpoint under a different config hash "
            f"(checkpoint={payload_config_hash}, current={expected_config_hash}). "
            "Use this only for explicit research continuations."
        )
    payload_spec_hash = payload.get("spec_hash256")
    if payload_spec_hash is not None and str(payload_spec_hash).strip().lower() != expected_spec_hash256:
        raise RuntimeError(
            f"checkpoint spec hash mismatch for {checkpoint_path}: expected {expected_spec_hash256}, got {payload_spec_hash}"
        )
    payload_algorithm = payload.get("algorithm")
    if payload_algorithm is not None and str(payload_algorithm).strip() and str(payload_algorithm).strip() != algorithm:
        raise RuntimeError(
            f"checkpoint algorithm mismatch for {checkpoint_path}: expected {algorithm}, got {payload_algorithm}"
        )
    model_state_dict = payload.get("model_state_dict")
    if learner.model is None or not isinstance(model_state_dict, dict):
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    learner.model.load_state_dict(model_state_dict)
    _restore_model_guidance_from_payload(learner.model, payload)
    optimizer_state_dict = payload.get("optimizer_state_dict")
    if restore_optimizer_state and optimizer_state_dict is not None:
        optimizer = learner._optimizer_for_step()
        optimizer.load_state_dict(optimizer_state_dict)
        for group in optimizer.param_groups:
            group["lr"] = float(learner.learning_rate)
    grad_scaler_state_dict = payload.get("grad_scaler_state_dict")
    if (
        restore_optimizer_state
        and grad_scaler_state_dict is not None
        and getattr(learner, "_grad_scaler", None) is not None
    ):
        learner._grad_scaler.load_state_dict(grad_scaler_state_dict)
    if restore_counters:
        learner.update_count = int(payload.get("update_count", 0))
        learner.policy_version = int(payload.get("policy_version", 0))
        learner.total_samples_processed = int(payload.get("total_samples_processed", 0))
        learner.start_time = time.time()
    return ResumeCheckpoint(
        checkpoint_path=checkpoint_path.resolve(),
        update_count=learner.update_count,
        policy_version=learner.policy_version,
        total_samples_processed=learner.total_samples_processed,
    )


def _build_training_learner(
    *,
    algorithm: str,
    model: PolicyValueModel,
    compiled_model: nn.Module | None,
    training_config: Any,
    training_paths: TrainingPaths,
    pass_action_id: int,
    checkpoint_interval_updates: int,
    gradient_sync: Any | None = None,
    artifact_writes_enabled: bool = True,
) -> ImpalaLearner | PpoLiteLearner:
    common_kwargs = {
        "model": model,
        "compiled_model": compiled_model,
        "learning_rate": training_config.learning_rate,
        "policy_loss_coef": float(getattr(training_config, "policy_loss_coef", 1.0)),
        "value_loss_coef": training_config.value_loss_coef,
        "entropy_coef": training_config.entropy_coef,
        "grad_norm_clip": training_config.grad_norm_clip,
        "optimizer_backend": str(getattr(training_config, "optimizer_backend", "auto")),
        "mixed_precision": bool(training_config.mixed_precision),
        "checkpoint_dir": training_paths.checkpoints_dir if artifact_writes_enabled else None,
        "checkpoint_interval_updates": int(checkpoint_interval_updates),
        "logs_dir": training_paths.logs_dir if artifact_writes_enabled else None,
        "logging_interval_updates": 1,
        "pass_action_id": pass_action_id,
        "teacher_family_coef": training_config.teacher_family_coef,
        "teacher_slot_coef": training_config.teacher_slot_coef,
        "teacher_move_source_coef": training_config.teacher_move_source_coef,
        "teacher_attack_type_coef": training_config.teacher_attack_type_coef,
        "teacher_action_coef": training_config.teacher_action_coef,
        "teacher_same_family_action_coef": training_config.teacher_same_family_action_coef,
        "teacher_public_heuristic_coef": training_config.teacher_public_heuristic_coef,
        "teacher_public_main_move_coef": training_config.teacher_public_main_move_coef,
        "teacher_development_pass_suppression_coef": training_config.teacher_development_pass_suppression_coef,
        "teacher_public_heuristic_temperature": training_config.teacher_public_heuristic_temperature,
        "teacher_public_heuristic_families": training_config.teacher_public_heuristic_families,
        "teacher_public_heuristic_profiles": training_config.teacher_public_heuristic_profiles,
        "teacher_public_heuristic_profile_mode": training_config.teacher_public_heuristic_profile_mode,
        "teacher_public_heuristic_profiles_end_updates": training_config.teacher_public_heuristic_profiles_end_updates,
        "behavior_action_bc_coef": float(getattr(training_config, "behavior_action_bc_coef", 0.0)),
        "reference_policy_top_action_bc_coef": float(
            getattr(training_config, "reference_policy_top_action_bc_coef", 0.0)
        ),
        "b1_opponent_reference_policy_top_action_bc_coef": float(
            getattr(training_config, "b1_opponent_reference_policy_top_action_bc_coef", 0.0)
        ),
        "b1_second_seat_positive_advantage_policy_coef": float(
            getattr(training_config, "b1_second_seat_positive_advantage_policy_coef", 0.0)
        ),
        "b1_second_seat_reference_top_action_avoidance_coef": float(
            getattr(training_config, "b1_second_seat_reference_top_action_avoidance_coef", 0.0)
        ),
        "reference_policy_top_action_family_bc_coef": float(
            getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0)
        ),
        "raw_b1_distill_coef": float(getattr(getattr(training_config, "raw_b1_distill", None), "coef", 0.0)),
        "raw_b1_distill_teacher_bias_scale": float(
            getattr(
                getattr(training_config, "raw_b1_distill", None),
                "teacher_public_heuristic_bias_scale",
                0.0,
            )
        ),
        "raw_b1_distill_student_bias_scale": float(
            getattr(
                getattr(training_config, "raw_b1_distill", None),
                "student_public_heuristic_bias_scale",
                0.0,
            )
        ),
        "raw_b1_distill_top_k": int(getattr(getattr(training_config, "raw_b1_distill", None), "top_k", 16)),
        "raw_b1_distill_temperature": float(
            getattr(getattr(training_config, "raw_b1_distill", None), "temperature", 1.5)
        ),
        "raw_b1_distill_top_action_ce_coef": float(
            getattr(getattr(training_config, "raw_b1_distill", None), "top_action_ce_coef", 0.0)
        ),
        "counterfactual_positive_label_dirs": tuple(
            getattr(getattr(training_config, "counterfactual_positive", None), "label_dirs", ()) or ()
        ),
        "counterfactual_positive_coef": float(
            getattr(getattr(training_config, "counterfactual_positive", None), "coef", 0.0)
        ),
        "counterfactual_positive_margin_coef": float(
            getattr(getattr(training_config, "counterfactual_positive", None), "margin_coef", 0.0)
        ),
        "counterfactual_positive_margin": float(
            getattr(getattr(training_config, "counterfactual_positive", None), "margin", 1.0)
        ),
        "counterfactual_positive_max_labels": int(
            getattr(getattr(training_config, "counterfactual_positive", None), "max_labels", 0)
        ),
        "profile_timers": bool(getattr(training_config, "profile_timers", False)),
        "structured_metrics_mode": str(getattr(training_config, "structured_metrics_mode", "full")),
        "teacher_aux_mode": str(getattr(training_config, "teacher_aux_mode", "always")),
        "gradient_sync": gradient_sync,
    }
    if algorithm in _IMPALA_ALGORITHMS:
        return ImpalaLearner(
            **common_kwargs,
            vtrace_rho_bar=training_config.vtrace_rho_bar,
            vtrace_c_bar=training_config.vtrace_c_bar,
        )
    if algorithm in _PPO_ALGORITHMS:
        return PpoLiteLearner(
            **common_kwargs,
            ppo_clip_epsilon=training_config.ppo_clip_epsilon,
            value_clip_epsilon=training_config.ppo_value_clip_epsilon,
            ppo_epochs=int(training_config.ppo_epochs),
            target_kl=training_config.ppo_target_kl,
            normalize_advantages=bool(training_config.ppo_normalize_advantages),
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


def _entropy_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    start = float(training_config.entropy_coef)
    target = float(training_config.entropy_anneal_to)
    steps = max(1, int(training_config.entropy_anneal_steps_updates))
    progress = min(max(int(update_count), 0), steps) / float(steps)
    return float(start + (target - start) * progress)


def _teacher_public_heuristic_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return float(
        linear_anneal_value(
            initial_value=float(training_config.teacher_public_heuristic_coef),
            final_value=float(getattr(training_config, "teacher_public_heuristic_final_coef", 0.0)),
            start_update=int(getattr(training_config, "teacher_public_heuristic_start_updates", 0)),
            end_update=int(getattr(training_config, "teacher_public_heuristic_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _reference_policy_top_action_bc_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    initial = float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(training_config, "reference_policy_top_action_bc_final_coef", initial)),
            start_update=int(getattr(training_config, "reference_policy_top_action_bc_start_updates", 0)),
            end_update=int(getattr(training_config, "reference_policy_top_action_bc_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _reference_policy_top_action_family_bc_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    initial = float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(training_config, "reference_policy_top_action_family_bc_final_coef", initial)),
            start_update=int(getattr(training_config, "reference_policy_top_action_family_bc_start_updates", 0)),
            end_update=int(getattr(training_config, "reference_policy_top_action_family_bc_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _raw_b1_distill_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    raw_b1_distill = getattr(training_config, "raw_b1_distill", None)
    if raw_b1_distill is None or not bool(getattr(raw_b1_distill, "enabled", False)):
        return 0.0
    initial = float(getattr(raw_b1_distill, "coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(raw_b1_distill, "final_coef", initial)),
            start_update=int(getattr(raw_b1_distill, "start_updates", 0)),
            end_update=int(getattr(raw_b1_distill, "end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _counterfactual_positive_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    counterfactual_positive = getattr(training_config, "counterfactual_positive", None)
    if counterfactual_positive is None or not bool(getattr(counterfactual_positive, "enabled", False)):
        return 0.0
    initial = float(getattr(counterfactual_positive, "coef", 0.0))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=float(getattr(counterfactual_positive, "final_coef", initial)),
            start_update=int(getattr(counterfactual_positive, "start_updates", 0)),
            end_update=int(getattr(counterfactual_positive, "end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _public_heuristic_logit_bias_scale_for_next_update(model_config: Any, *, update_count: int) -> float:
    return float(
        linear_anneal_value(
            initial_value=float(getattr(model_config, "public_heuristic_logit_bias_scale", 0.0)),
            final_value=float(
                getattr(
                    model_config,
                    "public_heuristic_logit_bias_final_scale",
                    getattr(model_config, "public_heuristic_logit_bias_scale", 0.0),
                )
            ),
            start_update=int(getattr(model_config, "public_heuristic_logit_bias_start_updates", 0)),
            end_update=int(getattr(model_config, "public_heuristic_logit_bias_end_updates", -1)),
            update_count=int(update_count),
        )
    )


def _public_heuristic_actor_logit_bias_scale_for_next_update(
    model_config: Any,
    *,
    learner_bias_scale: float,
) -> float:
    configured_actor_scale = float(getattr(model_config, "public_heuristic_actor_logit_bias_scale", -1.0))
    if configured_actor_scale < 0.0:
        return float(learner_bias_scale)
    return configured_actor_scale


def _apply_guidance_schedule_for_next_update(
    *,
    learner: ImpalaLearner,
    model: PolicyValueModel | None,
    stack: StackConfig,
    update_count: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    training_config = stack.config.training
    if training_config is not None:
        teacher_coef = _teacher_public_heuristic_coef_for_next_update(training_config, update_count=update_count)
        learner.set_teacher_aux_coefs(public_heuristic=teacher_coef)
        metrics["teacher_public_heuristic_coef_active"] = float(teacher_coef)
        reference_top_action_coef = _reference_policy_top_action_bc_coef_for_next_update(
            training_config,
            update_count=update_count,
        )
        reference_family_coef = _reference_policy_top_action_family_bc_coef_for_next_update(
            training_config,
            update_count=update_count,
        )
        learner.set_reference_policy_bc_coefs(
            top_action=reference_top_action_coef,
            top_action_family=reference_family_coef,
        )
        raw_b1_distill_coef = _raw_b1_distill_coef_for_next_update(training_config, update_count=update_count)
        if hasattr(learner, "set_raw_b1_distill_coef"):
            learner.set_raw_b1_distill_coef(raw_b1_distill_coef)
        counterfactual_positive_coef = _counterfactual_positive_coef_for_next_update(
            training_config,
            update_count=update_count,
        )
        if hasattr(learner, "set_counterfactual_positive_coef"):
            learner.set_counterfactual_positive_coef(counterfactual_positive_coef)
        metrics["reference_policy_top_action_bc_coef_active"] = float(reference_top_action_coef)
        metrics["reference_policy_top_action_family_bc_coef_active"] = float(reference_family_coef)
        metrics["raw_b1_distill_coef_active"] = float(raw_b1_distill_coef)
        metrics["counterfactual_positive_coef_active"] = float(counterfactual_positive_coef)
    model_config = stack.config.model
    if model is not None and model_config is not None:
        set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
        if callable(set_bias_scale):
            learner_bias_scale = _public_heuristic_logit_bias_scale_for_next_update(
                model_config,
                update_count=update_count,
            )
            actor_bias_scale = _public_heuristic_actor_logit_bias_scale_for_next_update(
                model_config,
                learner_bias_scale=learner_bias_scale,
            )
            set_bias_scale(learner_bias_scale, actor_value=actor_bias_scale)
            metrics["public_heuristic_logit_bias_scale_active"] = float(learner_bias_scale)
            metrics["public_heuristic_actor_logit_bias_scale_active"] = float(actor_bias_scale)
    return metrics


def _model_guidance_payload(model: PolicyValueModel | None) -> dict[str, float]:
    if model is None:
        return {}
    get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
    if not callable(get_bias_scale):
        return {}
    return {
        "public_heuristic_logit_bias_scale": float(get_bias_scale(scoring_mode="learner")),
        "public_heuristic_actor_logit_bias_scale": float(get_bias_scale(scoring_mode="actor")),
    }


def _restore_model_guidance_from_payload(
    model: PolicyValueModel | None,
    payload: Mapping[str, Any],
) -> None:
    if model is None:
        return
    set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
    if not callable(set_bias_scale):
        return
    learner_scale = payload.get("public_heuristic_logit_bias_scale")
    actor_scale = payload.get("public_heuristic_actor_logit_bias_scale")
    if learner_scale is None and actor_scale is None:
        return
    resolved_learner_scale = None if learner_scale is None else float(learner_scale)
    resolved_actor_scale = None if actor_scale is None else float(actor_scale)
    if resolved_learner_scale is None and resolved_actor_scale is not None:
        current_learner_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
        if callable(current_learner_scale):
            resolved_learner_scale = float(current_learner_scale(scoring_mode="learner"))
    if resolved_learner_scale is None:
        return
    set_bias_scale(resolved_learner_scale, actor_value=resolved_actor_scale)


def _maybe_compile_learner_model(
    *,
    model: PolicyValueModel,
    training_config: Any,
    device: torch.device,
) -> nn.Module | None:
    if not bool(getattr(training_config, "compile_learner", False)):
        return None
    if device.type != "cuda":
        print(
            "Learner compile note: compile_learner is enabled but the learner device is not CUDA; skipping torch.compile."
        )
        return None
    if bool(getattr(model, "supports_legal_candidate_scoring", False)):
        enable_trunk_compile = getattr(model, "enable_trunk_compile", None)
        if callable(enable_trunk_compile):
            try:
                enable_trunk_compile(mode="reduce-overhead")
            except Exception as exc:
                print(f"Learner compile note: structured trunk compile failed; skipping torch.compile ({exc!r}).")
                return None
            print("Enabled torch.compile for the structured learner trunk (mode=reduce-overhead).")
            return model
        print(
            "Learner compile note: structured legal scoring is enabled but no trunk compile hook exists; skipping torch.compile."
        )
        return None
    compiled = torch.compile(model, mode="reduce-overhead")
    print("Enabled torch.compile for the learner forward path (mode=reduce-overhead).")
    return compiled


@contextmanager
def _profile_block(enabled: bool, name: str):
    if not enabled:
        yield
        return
    with torch.autograd.profiler.record_function(name):
        yield


def _build_training_profiler(
    *,
    enabled: bool,
    run_dir: Path,
    device: torch.device,
) -> tuple[torch.profiler.profile | None, Any, Path | None]:
    if not enabled:
        return None, nullcontext(), None

    profile_dir = run_dir / "profiling" / "torch_profiler"
    profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    return profiler, profiler, profile_dir


def _collect_training_batch(
    *,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
) -> Any:
    pass_with_nonpass_penalty = float(
        getattr(getattr(rewards_config, "shaping", None), "pass_with_nonpass_penalty", 0.0)
    )
    if str(getattr(rewards_config, "objective", "")).strip().lower() == "terminal_only_pm1":
        pass_with_nonpass_penalty = 0.0
    if algorithm in _IMPALA_ALGORITHMS:
        return runtime.collect_update_batch(
            gamma=float(rewards_config.gamma),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
            pass_with_nonpass_penalty=pass_with_nonpass_penalty,
            vtrace_rho_bar=float(training_config.vtrace_rho_bar),
            vtrace_c_bar=float(training_config.vtrace_c_bar),
        )
    if algorithm in _PPO_ALGORITHMS:
        return runtime.collect_policy_batch(
            gamma=float(rewards_config.gamma),
            gae_lambda=float(training_config.ppo_gae_lambda),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
            pass_with_nonpass_penalty=pass_with_nonpass_penalty,
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


def _collect_training_batch_prefetch(
    *,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    actor_torch_threads: int | None,
) -> Any:
    with _torch_num_threads_scope(actor_torch_threads):
        return _collect_training_batch(
            runtime=runtime,
            algorithm=algorithm,
            training_config=training_config,
            rewards_config=rewards_config,
        )


def _run_structured_warmstart(
    *,
    learner: ImpalaLearner,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    training_paths: TrainingPaths,
    tensorboard_logger: TensorBoardLogger | None,
    start_time: float,
    profile_timers: bool = False,
    actor_torch_threads: int | None = None,
    learner_torch_threads: int | None = None,
) -> dict[str, float]:
    if not bool(getattr(training_config, "structured_warmstart_enabled", False)):
        return {}
    if algorithm not in _IMPALA_ALGORITHMS:
        raise RuntimeError("structured warmstart currently supports only IMPALA learners")
    warmstart_cfg = training_config.structured_warmstart
    updates = int(warmstart_cfg.updates)
    if updates <= 0:
        return {}

    previous_family = float(training_config.teacher_family_coef)
    previous_slot = float(training_config.teacher_slot_coef)
    previous_move_source = float(training_config.teacher_move_source_coef)
    previous_attack_type = float(training_config.teacher_attack_type_coef)
    previous_action = float(training_config.teacher_action_coef)
    previous_same_family_action = float(training_config.teacher_same_family_action_coef)
    previous_public_heuristic = float(training_config.teacher_public_heuristic_coef)
    previous_public_heuristic_temperature = float(training_config.teacher_public_heuristic_temperature)
    previous_public_heuristic_families = tuple(training_config.teacher_public_heuristic_families)
    previous_public_heuristic_profiles = tuple(training_config.teacher_public_heuristic_profiles)
    previous_public_heuristic_profile_mode = str(training_config.teacher_public_heuristic_profile_mode)
    previous_public_heuristic_profiles_end_updates = int(training_config.teacher_public_heuristic_profiles_end_updates)
    learner.set_teacher_aux_coefs(
        family=float(warmstart_cfg.teacher_family_coef),
        slot=float(warmstart_cfg.teacher_slot_coef),
        move_source=float(warmstart_cfg.teacher_move_source_coef),
        attack_type=float(warmstart_cfg.teacher_attack_type_coef),
        action=float(warmstart_cfg.teacher_action_coef),
        same_family_action=float(warmstart_cfg.teacher_same_family_action_coef),
        public_heuristic=float(warmstart_cfg.teacher_public_heuristic_coef),
        public_heuristic_temperature=float(warmstart_cfg.teacher_public_heuristic_temperature),
        public_heuristic_families=tuple(warmstart_cfg.teacher_public_heuristic_families),
        public_heuristic_profiles=tuple(warmstart_cfg.teacher_public_heuristic_profiles),
        public_heuristic_profile_mode=str(warmstart_cfg.teacher_public_heuristic_profile_mode),
        public_heuristic_profiles_end_updates=int(warmstart_cfg.teacher_public_heuristic_profiles_end_updates),
    )
    latest_metrics: dict[str, float] = {}
    try:
        with (
            runtime.structured_warmstart_source_mix() as warmstart_source_metrics,
            runtime.disable_mirror_policy_fusion(),
        ):
            for warmstart_step in range(updates):
                with (
                    _profile_block(profile_timers, "collect_training_batch"),
                    _torch_num_threads_scope(actor_torch_threads),
                ):
                    runtime_batch = _collect_training_batch(
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                    )
                with (
                    _profile_block(profile_timers, "learner_auxiliary_update"),
                    _torch_num_threads_scope(learner_torch_threads),
                ):
                    latest_metrics = learner.auxiliary_update(runtime_batch.learner_batch)
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(warmstart_source_metrics)
                latest_metrics["warmstart_phase"] = 1.0
                latest_metrics["warmstart_step"] = float(warmstart_step + 1)
                _write_scalars_record(
                    scalars_path=training_paths.scalars_path,
                    learner=learner,
                    metrics=latest_metrics,
                    start_time=start_time,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
    finally:
        learner.set_teacher_aux_coefs(
            family=previous_family,
            slot=previous_slot,
            move_source=previous_move_source,
            attack_type=previous_attack_type,
            action=previous_action,
            same_family_action=previous_same_family_action,
            public_heuristic=previous_public_heuristic,
            public_heuristic_temperature=previous_public_heuristic_temperature,
            public_heuristic_families=previous_public_heuristic_families,
            public_heuristic_profiles=previous_public_heuristic_profiles,
            public_heuristic_profile_mode=previous_public_heuristic_profile_mode,
            public_heuristic_profiles_end_updates=previous_public_heuristic_profiles_end_updates,
        )
    return latest_metrics


def _validate_algorithm_model_contract(*, algorithm: str, recurrent_core: str, encoder_kind: str) -> None:
    normalized_core = str(recurrent_core).strip().lower()
    normalized_encoder = str(encoder_kind).strip().lower()
    if algorithm == "impala_vtrace_gru" and normalized_core != "gru":
        raise RuntimeError("impala_vtrace_gru requires model.recurrent_core=gru")
    if algorithm == "impala_vtrace_ff" and normalized_core != "none":
        raise RuntimeError("impala_vtrace_ff requires model.recurrent_core=none")
    if algorithm in {"structured_v2", "impala_vtrace_structured_v1"} and normalized_core not in {"gru", "none"}:
        raise RuntimeError(f"{algorithm} requires a supported model.recurrent_core value")
    if algorithm in {"structured_v2", "impala_vtrace_structured_v1"} and normalized_encoder != "structured_v2":
        raise RuntimeError(f"{algorithm} requires model.encoder_kind=structured_v2")


def _json_relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug_policy_id(value: str) -> str:
    parts = [
        "".join(char.lower() for char in chunk if char.isalnum())
        for chunk in str(value).replace("-", " ").replace("_", " ").split()
    ]
    return "_".join(part for part in parts if part)


def _promotion_anchor_policy_id_candidates(anchor_name: str) -> tuple[str, ...]:
    if anchor_name == _PROMOTION_GATE_RANDOMLEGAL_NAME:
        return (_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,)
    if anchor_name == _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME:
        return (_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID, anchor_name)
    if heuristic_public_profile_name_for_policy_id(anchor_name) is not None:
        return (anchor_name,)
    normalized = _slug_policy_id(anchor_name)
    if not normalized:
        return ()
    return tuple(dict.fromkeys((normalized, anchor_name)))


def _resolve_symbolic_promotion_anchor_policy_id(
    anchor_name: str,
    *,
    registry: SnapshotRegistry,
    promotion_gate_enabled: bool = False,
) -> str | None:
    if anchor_name in {"Latest champion snapshot", "Latest promoted champion snapshot"}:
        champion_ids = registry.latest_active_champion_ids(
            1,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if not champion_ids else str(champion_ids[-1])
    if anchor_name in {"Previous champion snapshot", "Previous promoted champion snapshot"}:
        champion_ids = registry.latest_active_champion_ids(
            2,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if len(champion_ids) < 2 else str(champion_ids[-2])
    if anchor_name in {"Latest recent snapshot", "Latest local candidate snapshot"}:
        recent_ids = _true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if not recent_ids else str(recent_ids[-1])
    if anchor_name in {"Previous recent snapshot", "Previous local candidate snapshot"}:
        recent_ids = _true_local_recent_snapshot_ids(registry, promotion_gate_enabled=promotion_gate_enabled)
        return None if len(recent_ids) < 2 else str(recent_ids[-2])
    if anchor_name == "Latest imported seed history snapshot":
        seed_ids = registry.latest_seed_history_ids(
            1,
            exclude_rejected=True,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if not seed_ids else str(seed_ids[-1])
    if anchor_name == "Previous imported seed history snapshot":
        seed_ids = registry.latest_seed_history_ids(
            2,
            exclude_rejected=True,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
        return None if len(seed_ids) < 2 else str(seed_ids[-2])
    return None


def _true_local_recent_snapshot_ids(
    registry: SnapshotRegistry,
    *,
    promotion_gate_enabled: bool = False,
) -> tuple[str, ...]:
    if promotion_gate_enabled:
        return tuple(
            registry.latest_active_champion_ids(
                len(getattr(registry, "champion_snapshots", ())),
                exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
            )
        )
    return tuple(
        registry.latest_local_candidate_ids(
            len(getattr(registry, "snapshots", ())),
            include_league_import=True,
            exclude_rejected=True,
            exclude_policy_ids=_FIXED_OPPONENT_EXCLUSIONS,
        )
    )


def _build_heuristic_public_policy(
    spec_bundle: Mapping[str, object],
    *,
    scoring_profile: str,
) -> HeuristicPublicPolicy:
    factory = HeuristicPublicPolicy.from_spec_bundle
    supports_scoring_profile = False
    try:
        supports_scoring_profile = "scoring_profile" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        supports_scoring_profile = False
    if supports_scoring_profile:
        return factory(spec_bundle, scoring_profile=scoring_profile)
    return factory(spec_bundle)


def _find_noleague_baseline_snapshot(run_dir: Path) -> SnapshotMeta | None:
    layout = ArtifactLayout.from_run_dir(run_dir)
    registry_path = layout.training_snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return None
    registry = SnapshotRegistry.load(registry_path)
    snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    for policy_id in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME):
        snapshot = snapshots_by_id.get(policy_id)
        if snapshot is not None:
            return snapshot

    manifest_path = layout.manifest_path
    if not manifest_path.is_file():
        return None
    manifest = _load_json_object(manifest_path, label="run manifest")
    config_canonical = manifest.get("config_canonical", {})
    if not isinstance(config_canonical, dict):
        return None
    if not _config_marks_noleague_baseline(config_canonical):
        return None
    if not registry.snapshots:
        return None
    return max(registry.snapshots, key=lambda snapshot: snapshot.sort_key())


def _import_noleague_baseline_anchor(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    baseline_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> tuple[Path, str, int]:
    source_run_dir = Path(baseline_run_dir).resolve()
    source_snapshot = _find_noleague_baseline_snapshot(source_run_dir)
    if source_snapshot is None:
        raise FileNotFoundError(
            "Could not resolve the canonical B1 no-league baseline snapshot in "
            f"{source_run_dir}. Run a dedicated baseline_noleague training job first."
        )

    source_weights_path = source_run_dir / source_snapshot.path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"Resolved B1 baseline snapshot is missing its weights artifact: {source_weights_path}")

    snapshot_dir = training_paths.snapshots_dir / _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
    payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload must be a dict: {source_weights_path}")
    _validate_imported_snapshot_contract(
        source_run_dir=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )
    imported_payload = dict(payload)
    imported_payload["policy_id"] = _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
    imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
    imported_payload["imported_from_snapshot_path"] = source_snapshot.path
    torch.save(imported_payload, weights_path)
    weights_sha256 = _sha256_file(weights_path)

    _write_json_file(
        snapshot_dir / SNAPSHOT_METADATA_FILENAME,
        {
            "format": "imported_train_snapshot_metadata_v1",
            "policy_id": _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            "update": int(source_snapshot.update),
            "weights_path": snapshot_weights_relpath(_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID),
            "weights_sha256": weights_sha256,
            "imported_from_run_dir": source_run_dir.as_posix(),
            "imported_from_policy_id": source_snapshot.policy_id,
            "imported_from_snapshot_path": source_snapshot.path,
        },
    )
    return weights_path, weights_sha256, int(source_snapshot.update)


def _attach_reference_policy_model_if_configured(
    *,
    learner: ImpalaLearner | PpoLiteLearner,
    training_config: Any,
    training_paths: TrainingPaths,
    model_config: Any,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, Any] | None,
    spec_bundle: Mapping[str, Any],
    device: torch.device,
) -> None:
    if not isinstance(learner, ImpalaLearner):
        return
    coef = float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0))
    family_coef = float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0))
    raw_b1_distill = getattr(training_config, "raw_b1_distill", None)
    raw_b1_distill_enabled = bool(getattr(raw_b1_distill, "enabled", False)) and (
        float(getattr(raw_b1_distill, "coef", 0.0)) != 0.0
        or float(getattr(raw_b1_distill, "final_coef", 0.0)) != 0.0
    )
    if coef == 0.0 and family_coef == 0.0 and not raw_b1_distill_enabled:
        return
    policy_id = str(getattr(training_config, "reference_policy_id", "") or "").strip()
    if raw_b1_distill_enabled:
        raw_policy_id = str(getattr(raw_b1_distill, "teacher_policy_id", "") or "").strip()
        if raw_policy_id:
            if policy_id and raw_policy_id != policy_id and (coef != 0.0 or family_coef != 0.0):
                raise ValueError(
                    "training.raw_b1_distill.teacher_policy_id must match training.reference_policy_id "
                    "when both reference BC and raw B1 distill are enabled"
                )
            policy_id = raw_policy_id
    if not policy_id:
        policy_id = _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    weights_path = training_paths.snapshots_dir / policy_id / SNAPSHOT_WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"reference policy weights not found for policy_id={policy_id!r}: {weights_path}")
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"reference policy weights payload is missing model_state_dict: {weights_path}")
    reference_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(device)
    reference_model.load_state_dict(payload["model_state_dict"])
    _restore_model_guidance_from_payload(reference_model, payload)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    learner.reference_policy_model = reference_model
    print(
        "Attached frozen reference policy: "
        f"policy_id={policy_id} coef={coef:g} family_coef={family_coef:g} "
        f"raw_b1_distill={raw_b1_distill_enabled} weights={weights_path}"
    )


def _validate_seed_snapshot_import_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    def _model_section_for_contract(section: Any) -> Any:
        if not isinstance(section, Mapping):
            return section
        normalized = dict(section)
        # Snapshot guidance values are carried in weights.pt and restored per
        # policy when the snapshot is loaded.
        normalized.pop("public_heuristic_logit_bias_scale", None)
        normalized.pop("public_heuristic_actor_logit_bias_scale", None)
        normalized.pop("public_heuristic_logit_bias_start_updates", None)
        normalized.pop("public_heuristic_logit_bias_end_updates", None)
        normalized.pop("public_heuristic_logit_bias_final_scale", None)
        return normalized

    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    manifest_path = source_layout.manifest_path
    source_manifest = (
        _load_json_object(manifest_path, label="seed snapshot manifest") if manifest_path.is_file() else None
    )
    source_config_canonical = source_manifest.get("config_canonical") if isinstance(source_manifest, dict) else None
    if isinstance(source_config_canonical, dict) and isinstance(expected_config_canonical, dict):
        source_config_sections = _canonical_config_sections(source_config_canonical)
        expected_config_sections = _canonical_config_sections(expected_config_canonical)
        for section_name in ("model", "environment"):
            source_section = source_config_sections.get(section_name)
            expected_section = expected_config_sections.get(section_name)
            if source_section is None or expected_section is None:
                continue
            if section_name == "model":
                source_section = _model_section_for_contract(source_section)
                expected_section = _model_section_for_contract(expected_section)
            if source_section != expected_section:
                raise RuntimeError(
                    f"Imported seed snapshot config does not match the current run for section={section_name!r}"
                )

    if expected_spec_hash256 is not None:
        source_spec_hash = _read_optional_hash_file(source_layout.spec_hash_path)
        if source_spec_hash is not None and source_spec_hash != expected_spec_hash256:
            raise RuntimeError(
                "Imported seed snapshot spec hash does not match the current run: "
                f"source={source_spec_hash} expected={expected_spec_hash256}"
            )

    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"Imported seed snapshot weights payload is missing model_state_dict: {source_run_dir}")
    source_keys = set(source_model_state_dict)
    expected_keys = set(expected_model_state_dict)
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise RuntimeError(
            "Imported seed snapshot model contract does not match the current run: "
            f"missing_keys={missing} extra_keys={extra}"
        )
    for key in sorted(expected_keys):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                "Imported seed snapshot tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def _validate_snapshot_tensor_contract(
    *,
    label: str,
    source_path: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
) -> None:
    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"{label} weights payload is missing model_state_dict: {source_path}")
    source_keys = set(source_model_state_dict)
    expected_keys = set(expected_model_state_dict)
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise RuntimeError(f"{label} model contract does not match the current run: missing_keys={missing} extra_keys={extra}")
    for key in sorted(expected_keys):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                f"{label} tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def _seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
    source_hash = hashlib.sha1(source_run_dir.as_posix().encode("utf-8")).hexdigest()[:10]
    safe_policy_id = str(source_policy_id).replace("/", "_").replace("\\", "_").strip()
    return f"seed_{source_hash}_{safe_policy_id}"


def _import_seed_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    seed_snapshot_run_dir: Path,
    max_update: int | None = None,
    exclude_source_policy_ids: Sequence[str] = (),
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> list[str]:
    source_run_dir = Path(seed_snapshot_run_dir).resolve()
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    if not source_registry_path.is_file():
        raise FileNotFoundError(
            f"Could not resolve a snapshot registry in the seed snapshot run: {source_registry_path}"
        )
    source_registry = SnapshotRegistry.load(source_registry_path)
    excluded_source_policy_ids = {str(policy_id).strip() for policy_id in exclude_source_policy_ids}
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if snapshot.policy_id not in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
        and snapshot.policy_id not in excluded_source_policy_ids
        and (max_update is None or int(snapshot.update) <= int(max_update))
    ]
    if not source_snapshots:
        return []

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    existing_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    source_champions = set(source_registry.champion_snapshots)
    imported_policy_ids: list[str] = []
    for source_snapshot in source_snapshots:
        imported_policy_id = _seed_snapshot_policy_id(
            source_run_dir=source_run_dir,
            source_policy_id=source_snapshot.policy_id,
        )
        if imported_policy_id in existing_policy_ids:
            imported_policy_ids.append(imported_policy_id)
            continue
        source_weights_path = source_run_dir / source_snapshot.path
        if not source_weights_path.is_file():
            raise FileNotFoundError(f"Resolved seed snapshot is missing its weights artifact: {source_weights_path}")
        payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Imported seed snapshot weights payload must be a dict: {source_weights_path}")
        _validate_seed_snapshot_import_contract(
            source_run_dir=source_run_dir,
            payload=payload,
            expected_model_state_dict=expected_model_state_dict,
            expected_config_canonical=expected_config_canonical,
            expected_spec_hash256=expected_spec_hash256,
        )
        snapshot_dir = training_paths.snapshots_dir / imported_policy_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
        imported_payload = dict(payload)
        imported_payload["policy_id"] = imported_policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["seeded_from_external_registry"] = True
        imported_payload["source_was_champion"] = source_snapshot.policy_id in source_champions
        torch.save(imported_payload, weights_path)
        weights_sha256 = _sha256_file(weights_path)
        _write_json_file(
            snapshot_dir / SNAPSHOT_METADATA_FILENAME,
            {
                "format": "seeded_train_snapshot_metadata_v1",
                "policy_id": imported_policy_id,
                "update": int(source_snapshot.update),
                "weights_path": snapshot_weights_relpath(imported_policy_id),
                "weights_sha256": weights_sha256,
                "imported_from_run_dir": source_run_dir.as_posix(),
                "imported_from_policy_id": source_snapshot.policy_id,
                "imported_from_snapshot_path": source_snapshot.path,
                "source_was_champion": source_snapshot.policy_id in source_champions,
            },
        )
        registry.add_snapshot(
            policy_id=imported_policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="seed_import",
        )
        existing_policy_ids.add(imported_policy_id)
        imported_policy_ids.append(imported_policy_id)

    if imported_policy_ids:
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported seeded snapshot pool: "
            f"count={len(imported_policy_ids)} "
            f"source_run_dir={source_run_dir.as_posix()}"
        )
    return imported_policy_ids


def _source_snapshot_is_resume_league_snapshot(snapshot: SnapshotMeta, *, rejected_policy_ids: set[str]) -> bool:
    policy_id = str(snapshot.policy_id).strip()
    if (
        not policy_id
        or policy_id in rejected_policy_ids
        or policy_id in _FIXED_OPPONENT_EXCLUSIONS
        or policy_id.startswith("seed_")
    ):
        return False
    source_kind = str(getattr(snapshot, "source_kind", "local") or "local").strip().lower()
    return source_kind not in {"seed_import", "baseline_anchor"}


def _validate_existing_resume_league_import(
    *,
    training_paths: TrainingPaths,
    source_run_dir: Path,
    source_snapshot: SnapshotMeta,
) -> None:
    policy_id = str(source_snapshot.policy_id)
    metadata_path = training_paths.snapshots_dir / policy_id / SNAPSHOT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise RuntimeError(f"Existing resume league snapshot import is missing metadata: {metadata_path}")
    metadata = _load_json_object(metadata_path, label="existing resume league snapshot metadata")
    expected_source_run_dir = source_run_dir.as_posix()
    if (
        metadata.get("format") != "resume_league_snapshot_metadata_v1"
        or str(metadata.get("imported_from_run_dir", "")) != expected_source_run_dir
        or str(metadata.get("imported_from_policy_id", "")) != policy_id
        or str(metadata.get("imported_from_snapshot_path", "")) != str(source_snapshot.path)
    ):
        raise RuntimeError(
            "Existing resume league snapshot policy_id collision does not match the requested import: "
            f"policy_id={policy_id} metadata_path={metadata_path}"
        )


def _import_resume_league_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    resume_checkpoint_path: Path,
    max_update: int,
    expected_model_state_dict: dict[str, Any],
) -> list[str]:
    source_run_dir = _infer_run_dir_from_checkpoint_path(resume_checkpoint_path)
    if source_run_dir is None or source_run_dir.resolve() == Path(run_dir).resolve():
        return []
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    if not source_registry_path.is_file():
        return []
    source_registry = SnapshotRegistry.load(source_registry_path)
    rejected_policy_ids = set(getattr(source_registry, "rejected_snapshots", ()))
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if int(snapshot.update) <= int(max_update)
        and _source_snapshot_is_resume_league_snapshot(snapshot, rejected_policy_ids=rejected_policy_ids)
    ]
    if not source_snapshots:
        return []

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    existing_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    source_champions = set(source_registry.champion_snapshots)
    imported_policy_ids: list[str] = []
    for source_snapshot in source_snapshots:
        policy_id = str(source_snapshot.policy_id)
        if policy_id in existing_policy_ids:
            _validate_existing_resume_league_import(
                training_paths=training_paths,
                source_run_dir=source_run_dir,
                source_snapshot=source_snapshot,
            )
            imported_policy_ids.append(policy_id)
            continue
        source_weights_path = source_run_dir / source_snapshot.path
        if not source_weights_path.is_file():
            raise FileNotFoundError(f"Resolved resume league snapshot is missing weights: {source_weights_path}")
        payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Imported resume league snapshot weights payload must be a dict: {source_weights_path}")
        _validate_snapshot_tensor_contract(
            label="Imported resume league snapshot",
            source_path=source_weights_path,
            payload=payload,
            expected_model_state_dict=expected_model_state_dict,
        )
        snapshot_dir = training_paths.snapshots_dir / policy_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
        imported_payload = dict(payload)
        imported_payload["policy_id"] = policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["resumed_from_league_registry"] = True
        torch.save(imported_payload, weights_path)
        weights_sha256 = _sha256_file(weights_path)
        _write_json_file(
            snapshot_dir / SNAPSHOT_METADATA_FILENAME,
            {
                "format": "resume_league_snapshot_metadata_v1",
                "policy_id": policy_id,
                "update": int(source_snapshot.update),
                "weights_path": snapshot_weights_relpath(policy_id),
                "weights_sha256": weights_sha256,
                "imported_from_run_dir": source_run_dir.as_posix(),
                "imported_from_policy_id": source_snapshot.policy_id,
                "imported_from_snapshot_path": source_snapshot.path,
                "source_kind": str(getattr(source_snapshot, "source_kind", "local") or "local"),
            },
        )
        registry.add_snapshot(
            policy_id=policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="league_import",
        )
        if source_snapshot.policy_id in source_champions:
            registry.add_champion(policy_id)
        existing_policy_ids.add(policy_id)
        imported_policy_ids.append(policy_id)

    if imported_policy_ids:
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported resume league snapshot pool: "
            f"count={len(imported_policy_ids)} "
            f"source_run_dir={source_run_dir.as_posix()}"
        )
    return imported_policy_ids


def _ensure_noleague_baseline_anchor(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    learner: ImpalaLearner,
    device: torch.device,
    config_hash256: str,
    spec_hash256: str | None = None,
    baseline_run_dir: Path | None = None,
    permit_current_run_alias: bool = False,
    source_checkpoint_path: Path | None = None,
    update: int | None = None,
) -> str | None:
    league = stack.config.league
    training_config = stack.config.training
    experiment_role = _experiment_role(stack)
    reference_policy_id = str(getattr(training_config, "reference_policy_id", "") or "").strip()
    if not reference_policy_id:
        reference_policy_id = _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    raw_b1_distill = getattr(training_config, "raw_b1_distill", None)
    raw_b1_distill_enabled = bool(getattr(raw_b1_distill, "enabled", False)) and (
        float(getattr(raw_b1_distill, "coef", 0.0)) != 0.0
        or float(getattr(raw_b1_distill, "final_coef", 0.0)) != 0.0
    )
    raw_b1_distill_policy_id = str(getattr(raw_b1_distill, "teacher_policy_id", "") or "").strip()
    reference_needs_b1_anchor = (
        float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0)) != 0.0
        or float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0)) != 0.0
        or float(getattr(training_config, "b1_opponent_reference_policy_top_action_bc_coef", 0.0)) != 0.0
        or raw_b1_distill_enabled
    ) and reference_policy_id in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
    if raw_b1_distill_enabled and raw_b1_distill_policy_id:
        reference_needs_b1_anchor = reference_needs_b1_anchor or (
            raw_b1_distill_policy_id in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
        )
    requires_anchor = bool(
        reference_needs_b1_anchor
        or (
            league is not None
            and league.enabled
            and league.promotion_gate_enabled
            and _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME in league.promotion_anchor_set_v1.required
        )
    )
    if not requires_anchor and not permit_current_run_alias:
        return None
    if learner.model is None:
        raise RuntimeError("Cannot ensure the NoLeague baseline anchor without a learner model")

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    _sync_snapshot_registry_retention(stack, registry)
    available_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    existing_policy_id = next(
        (
            candidate
            for candidate in _promotion_anchor_policy_id_candidates(_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
            if candidate in available_policy_ids
        ),
        None,
    )
    if existing_policy_id is not None and baseline_run_dir is None and permit_current_run_alias:
        existing_snapshot = next(
            (snapshot for snapshot in registry.snapshots if snapshot.policy_id == existing_policy_id),
            None,
        )
        resolved_update = int(learner.update_count if update is None else update)
        if existing_snapshot is None or int(existing_snapshot.update) < resolved_update:
            existing_policy_id = None
    if existing_policy_id is not None:
        registry.pin_snapshot(existing_policy_id)
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        return existing_policy_id

    if baseline_run_dir is not None:
        weights_path, weights_sha256, imported_update = _import_noleague_baseline_anchor(
            training_paths=training_paths,
            run_dir=run_dir,
            baseline_run_dir=baseline_run_dir,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
        registry.add_snapshot(
            policy_id=_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            update=int(imported_update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="baseline_anchor",
        )
        registry.pin_snapshot(_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported promotion anchor: "
            f"anchor={_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
            f"policy_id={_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
            f"source_run_dir={Path(baseline_run_dir).resolve()}"
        )
        return _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID

    if not permit_current_run_alias:
        if requires_anchor:
            raise RuntimeError(
                "The canonical B1 NoLeague baseline is required for this training run. "
                "Pass --b1-baseline-run-dir pointing at a completed baseline_noleague run."
            )
        return None

    resolved_update = int(learner.update_count if update is None else update)
    checkpoint_path = (
        training_paths.checkpoints_dir / _PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT
        if source_checkpoint_path is None
        else Path(source_checkpoint_path)
    )
    if source_checkpoint_path is None:
        _write_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            algorithm=str(training_config.algorithm).strip() if training_config is not None else None,
            spec_hash256=spec_hash256,
        )
    guidance_payload = _model_guidance_payload(learner.model)
    weights_path, weights_sha256 = _write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=learner.model.state_dict(),
        public_heuristic_logit_bias_scale=guidance_payload.get("public_heuristic_logit_bias_scale"),
        public_heuristic_actor_logit_bias_scale=guidance_payload.get("public_heuristic_actor_logit_bias_scale"),
    )
    registry.add_snapshot(
        policy_id=_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
        source_kind="baseline_anchor",
    )
    registry.pin_snapshot(_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
    _save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    print(
        "Persisted canonical B1 baseline alias: "
        f"anchor={_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
        f"policy_id={_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
        f"experiment_role={experiment_role or 'unknown'} update={resolved_update}"
    )
    return _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID


def _resolve_promotion_anchor_policy_ids(
    *,
    stack: StackConfig,
    registry: SnapshotRegistry,
) -> tuple[dict[str, str], tuple[str, ...]]:
    league = stack.config.league
    if league is None:
        return {}, ()

    available_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    resolved: dict[str, str] = {}
    missing_required: list[str] = []
    anchor_names = [
        *league.promotion_anchor_set_v1.required,
        *league.promotion_anchor_set_v1.optional_if_available,
    ]
    required_names = set(league.promotion_anchor_set_v1.required)
    promotion_gate_enabled = bool(getattr(league, "promotion_gate_enabled", False))

    for anchor_name in anchor_names:
        policy_id = _resolve_symbolic_promotion_anchor_policy_id(
            anchor_name,
            registry=registry,
            promotion_gate_enabled=promotion_gate_enabled,
        )
        if policy_id is None:
            candidates = _promotion_anchor_policy_id_candidates(anchor_name)
            policy_id = next((candidate for candidate in candidates if candidate in available_policy_ids), None)
        if policy_id is None and anchor_name == _PROMOTION_GATE_RANDOMLEGAL_NAME:
            policy_id = _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
        if policy_id is None and heuristic_public_profile_name_for_policy_id(anchor_name) is not None:
            policy_id = anchor_name
        if policy_id is not None:
            resolved[anchor_name] = policy_id
            continue
        if anchor_name in required_names:
            missing_required.append(anchor_name)

    return resolved, tuple(missing_required)


def _snapshot_meta_by_policy_id(registry: SnapshotRegistry) -> dict[str, Any]:
    return {snapshot.policy_id: snapshot for snapshot in registry.snapshots}


def _get_cached_eval_snapshot_model(cache_key: tuple[Any, ...]) -> PolicyValueModel | None:
    cached_model = _EVAL_SNAPSHOT_MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        _EVAL_SNAPSHOT_MODEL_CACHE.move_to_end(cache_key)
        cached_model.eval()
    return cached_model


def _remember_eval_snapshot_model(cache_key: tuple[Any, ...], eval_model: PolicyValueModel) -> None:
    _EVAL_SNAPSHOT_MODEL_CACHE[cache_key] = eval_model
    _EVAL_SNAPSHOT_MODEL_CACHE.move_to_end(cache_key)
    while len(_EVAL_SNAPSHOT_MODEL_CACHE) > _EVAL_SNAPSHOT_MODEL_CACHE_MAX_ENTRIES:
        _EVAL_SNAPSHOT_MODEL_CACHE.popitem(last=False)


def _load_snapshot_eval_model(
    *,
    run_dir: Path,
    snapshot_path: str,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    eval_device: torch.device | str | None = None,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    resolved_snapshot_path = (run_dir / snapshot_path).resolve()
    stat = resolved_snapshot_path.stat()
    resolved_eval_device = _resolve_eval_device(stack, eval_device=eval_device)
    cache_key = (
        str(resolved_snapshot_path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(resolved_eval_device),
        int(observation_dim),
        int(action_dim),
    )
    cached_model = _get_cached_eval_snapshot_model(cache_key)
    if cached_model is not None:
        return cached_model

    payload = torch.load(resolved_snapshot_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Snapshot weights payload missing model_state_dict: {snapshot_path}")

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")

    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(resolved_eval_device)
    eval_model.load_state_dict(
        {name: value.detach().to(device=resolved_eval_device).clone() for name, value in model_state_dict.items()}
    )
    _restore_model_guidance_from_payload(eval_model, payload)
    eval_model.eval()
    _remember_eval_snapshot_model(cache_key, eval_model)
    return eval_model


def _load_checkpoint_eval_model(
    *,
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    eval_device: torch.device | str | None = None,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Checkpoint payload missing model_state_dict: {checkpoint_path}")

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")

    resolved_eval_device = _resolve_eval_device(stack, eval_device=eval_device)
    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(resolved_eval_device)
    eval_model.load_state_dict(
        {name: value.detach().to(device=resolved_eval_device).clone() for name, value in model_state_dict.items()}
    )
    _restore_model_guidance_from_payload(eval_model, payload)
    eval_model.eval()
    return eval_model


class _PromotionGateRunner:
    def __init__(
        self,
        *,
        stack: StackConfig,
        focal_policy_id: str,
        focal_model: PolicyValueModel,
        anchor_models: dict[str, PolicyValueModel],
        heuristic_policies: dict[str, HeuristicPublicPolicy],
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        require_sorted_legal_ids: bool,
        eval_device: torch.device | str | None = None,
    ) -> None:
        self.stack = stack
        self.focal_policy_id = focal_policy_id
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pass_action_id = pass_action_id
        self.artifact_dir = artifact_dir
        self.require_sorted_legal_ids = require_sorted_legal_ids
        self._policy_models = {focal_policy_id: focal_model, **anchor_models}
        self._heuristic_policies = dict(heuristic_policies)
        self._baseline_logits = np.zeros((action_dim,), dtype=np.float32)
        self._device = _resolve_eval_device(stack, eval_device=eval_device)
        self._persistent_env: DecisionBoundaryEnv | None = None

    def close(self) -> None:
        env = self._persistent_env
        self._persistent_env = None
        if env is not None:
            env.close()

    def run_game(self, scheduled_game: ScheduledGame):
        env = self._env_for_game(seed=scheduled_game.episode_seed)
        seat_hidden = {
            seat: self._initial_hidden(scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id)
            for seat in (0, 1)
        }
        seat_rngs = {
            seat: Pcg32XshRrV1(_promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=seat)) for seat in (0, 1)
        }
        last_acting_seat: int | None = None

        try:
            batch = env.reset(seed=scheduled_game.episode_seed)
            self._abort_on_fault(batch)
            while True:
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    return game_result_from_step(
                        batch,
                        env_index=0,
                        acting_seat=last_acting_seat,
                        episode_seed=scheduled_game.episode_seed,
                    )

                current_seat = int(batch.actor[0])
                current_policy_id = (
                    scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
                )
                action, next_hidden = self._select_action(
                    batch=batch,
                    current_seat=current_seat,
                    current_policy_id=current_policy_id,
                    seat_hidden=seat_hidden[current_seat],
                    rng=seat_rngs[current_seat],
                )
                seat_hidden[current_seat] = next_hidden
                last_acting_seat = current_seat
                batch = env.step(np.asarray([action], dtype=np.uint32))
                self._abort_on_fault(batch)
        finally:
            pass

    def _env_for_game(self, *, seed: int) -> DecisionBoundaryEnv:
        if self._persistent_env is None:
            self._persistent_env = _build_ids_eval_env(
                self.stack,
                seed=seed,
                pass_action_id=self.pass_action_id,
            )
        return self._persistent_env

    def _initial_hidden(self, policy_id: str) -> torch.Tensor | None:
        model = self._policy_models.get(policy_id)
        if model is None:
            return None
        return model.initial_seat_hidden(1, device=self._device)

    def _select_action(
        self,
        *,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        current_policy_id: str,
        seat_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
    ) -> tuple[int, torch.Tensor | None]:
        legal_ids = _legal_ids_for_env_row(
            batch=batch,
            env_index=0,
            require_sorted=self.require_sorted_legal_ids,
        )
        heuristic_policy = self._heuristic_policies.get(current_policy_id)
        if heuristic_policy is not None:
            action = heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), seat_hidden
        model = self._policy_models.get(current_policy_id)
        if model is None:
            if current_policy_id != _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
                raise RuntimeError(f"Unsupported promotion-gate policy_id: {current_policy_id}")
            action, _ = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=rng,
                pass_action_id=self.pass_action_id,
            )
            return action, seat_hidden

        if seat_hidden is None:
            raise RuntimeError(f"Missing hidden state for promotion-gate policy_id: {current_policy_id}")

        with torch.inference_mode():
            logits_tensor, _value_tensor, next_seat_hidden = model.forward_seat_aware(
                torch.as_tensor(np.asarray(batch.obs, dtype=np.float32), device=self._device),
                torch.as_tensor([current_seat], device=self._device, dtype=torch.long),
                seat_hidden,
                scoring_mode="learner",
            )
        logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        action, _ = sample_action_pinned(
            logits,
            legal_ids,
            rng=rng,
            pass_action_id=self.pass_action_id,
        )
        return action, next_seat_hidden

    def _abort_on_fault(self, batch: DecisionBoundaryBatch) -> None:
        abort_on_engine_fault_eval(
            run_dir=self.artifact_dir,
            engine_status=batch.engine_status,
            decision_id=batch.decision_id,
            episode_key=batch.episode_key,
            note="engine_status!=0 during promotion gate",
        )


def _evaluation_config_or_raise(stack: StackConfig):
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise RuntimeError("The locked stack is missing the evaluation config block")
    return evaluation


def _validate_periodic_dev_eval_contract(stack: StackConfig) -> Any:
    evaluation = _evaluation_config_or_raise(stack)
    if not evaluation.seat_swap:
        raise RuntimeError("Periodic dev eval requires evaluation.seat_swap=true")
    if not evaluation.eval_inference_mode:
        raise RuntimeError("Periodic dev eval requires evaluation.eval_inference_mode=true")
    if evaluation.eval_sampling_algorithm != "pinned_cdf_pcg_v1":
        raise RuntimeError(
            "Periodic dev eval requires evaluation.eval_sampling_algorithm='pinned_cdf_pcg_v1', "
            f"got {evaluation.eval_sampling_algorithm!r}"
        )
    return evaluation


def _resolve_repo_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def _action_family_ids_from_spec(*, action_dim: int, spec_bundle: Mapping[str, Any]) -> tuple[torch.Tensor | None, int]:
    try:
        catalog = ActionCatalog.from_spec_bundle(spec_bundle)
    except Exception:
        return None, 0
    family_names = tuple(family.name for family in catalog.families)
    family_index = {name: index for index, name in enumerate(family_names)}
    ids = torch.full((int(action_dim),), -1, dtype=torch.long)
    for action_id in range(int(action_dim)):
        try:
            decoded = catalog.decode(action_id)
        except Exception:
            continue
        ids[action_id] = int(family_index.get(decoded.family, -1))
    return ids, len(family_names)


def _load_main_residual_base_model(
    *,
    stack: StackConfig,
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, Any] | None,
    spec_bundle: Mapping[str, Any],
    device: torch.device,
) -> PolicyValueModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"main residual base checkpoint is missing model_state_dict: {checkpoint_path}")
    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("main residual base model requires stack.config.model")
    base_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(device)
    base_model.load_state_dict(payload["model_state_dict"])
    _restore_model_guidance_from_payload(base_model, payload)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    return base_model


def _maybe_build_main_residual_model(
    *,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, Any] | None,
    spec_bundle: Mapping[str, Any],
    device: torch.device,
) -> nn.Module | None:
    training_config = stack.config.training
    if training_config is None:
        return None
    residual_config = getattr(training_config, "main_residual_policy", None)
    if residual_config is None or not bool(getattr(residual_config, "enabled", False)):
        return None
    checkpoint_path = _resolve_repo_path(stack.root, str(residual_config.base_snapshot_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"training.main_residual_policy.base_snapshot_path not found: {checkpoint_path}")
    base_model = _load_main_residual_base_model(
        stack=stack,
        checkpoint_path=checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
        device=device,
    )
    set_bias = getattr(base_model, "set_public_heuristic_logit_bias_scale", None)
    if callable(set_bias):
        scale = float(getattr(residual_config, "public_heuristic_bias_scale", 1.0))
        set_bias(scale, actor_value=scale)
    initial_state_path_text = str(getattr(residual_config, "initial_residual_state_path", "") or "").strip()
    if initial_state_path_text:
        initial_state_path = _resolve_repo_path(stack.root, initial_state_path_text)
        if not initial_state_path.is_file():
            raise FileNotFoundError(
                f"training.main_residual_policy.initial_residual_state_path not found: {initial_state_path}"
            )
        residual = load_frozen_stored_logit_residual(initial_state_path, device=device)
        if int(residual.action_dim) != int(action_dim):
            raise RuntimeError(
                "training.main_residual_policy.initial_residual_state_path action_dim mismatch: "
                f"expected {action_dim}, got {residual.action_dim}"
            )
        residual.alpha = float(getattr(residual_config, "alpha", residual.alpha))
        residual.train()
    else:
        action_family_ids: torch.Tensor | None = None
        family_count = 0
        if str(getattr(residual_config, "residual_mode", "plain")) == "family_gated":
            action_family_ids, family_count = _action_family_ids_from_spec(
                action_dim=action_dim,
                spec_bundle=spec_bundle,
            )
        residual = FrozenStoredLogitResidual(
            obs_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=int(getattr(residual_config, "hidden_dim", 256)),
            alpha=float(getattr(residual_config, "alpha", 0.1)),
            residual_mode=str(getattr(residual_config, "residual_mode", "plain")),
            action_family_ids=action_family_ids,
            family_count=family_count,
            gate_bias=float(getattr(residual_config, "gate_bias", 0.0)),
        ).to(device)
    wrapper = TrainableLiveFrozenB1Residual(base_model=base_model, residual_probe=residual).to(device)
    print(
        "Enabled trainable main residual policy: "
        f"base={checkpoint_path} alpha={residual.alpha:g} "
        f"hidden_dim={int(getattr(residual_config, 'hidden_dim', 256))} "
        f"mode={residual.residual_mode} "
        f"initial_state={initial_state_path_text or '<zero>'}"
    )
    return wrapper


def _resolve_periodic_dev_eval_seed_file(stack: StackConfig) -> tuple[Path, dict[str, str]]:
    evaluation = _evaluation_config_or_raise(stack)
    reproducibility = stack.config.reproducibility
    resolved_paths: dict[str, Path] = {}
    if "dev_eval" in stack.seed_sets:
        resolved_paths["stack.seed_sets.dev_eval"] = stack.seed_sets["dev_eval"]
    if "dev_eval" in evaluation.seed_files:
        resolved_paths["evaluation.seed_files.dev_eval"] = _resolve_repo_path(
            stack.root,
            evaluation.seed_files["dev_eval"],
        )
    if reproducibility is not None and "dev_eval" in reproducibility.seed_files:
        resolved_paths["reproducibility.seed_files.dev_eval"] = _resolve_repo_path(
            stack.root,
            reproducibility.seed_files["dev_eval"],
        )
    if not resolved_paths:
        raise RuntimeError("Periodic dev eval requires a configured dev_eval seed file")

    unique_paths = {path.resolve() for path in resolved_paths.values()}
    if len(unique_paths) != 1:
        mismatch = {name: _json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}
        raise RuntimeError(f"Periodic dev eval seed file mismatch: {mismatch}")

    seed_file = next(iter(resolved_paths.values()))
    return seed_file, {name: _json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}


def _periodic_dev_eval_schedule(stack: StackConfig) -> tuple[Path, dict[str, str], list[int], str]:
    evaluation = _validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources = _resolve_periodic_dev_eval_seed_file(stack)
    all_paired_seeds = parse_seed_file(seed_file)
    required_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    if len(all_paired_seeds) < required_pairs:
        raise RuntimeError(
            f"Periodic dev eval requires {required_pairs} paired seeds, found {len(all_paired_seeds)} in {seed_file}"
        )
    return seed_file, validated_sources, all_paired_seeds[:required_pairs], hash_seed_file(seed_file)


def _legal_ids_for_env_row(
    *,
    batch: DecisionBoundaryBatch,
    env_index: int,
    require_sorted: bool,
) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError("Expected ids_offsets legality during periodic dev eval")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[env_index])
    end = int(legal_offsets[env_index + 1])
    row = np.asarray(legal_ids[start:end], dtype=np.uint32)
    if require_sorted:
        assert_strictly_increasing_legal_ids(row)
    return row


def _periodic_dev_eval_rng_seed(*, scheduled_game: ScheduledGame, seat: int) -> int:
    payload = canonical_json_bytes(
        {
            "kind": "periodic_dev_eval_rng_v1",
            "pair_index": scheduled_game.pair_index,
            "swap_index": scheduled_game.swap_index,
            "episode_seed": scheduled_game.episode_seed,
            "seat": int(seat),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
    )
    return stable_hash64(payload)


def _promotion_gate_rng_seed(*, scheduled_game: ScheduledGame, seat: int) -> int:
    payload = canonical_json_bytes(
        {
            "kind": "promotion_gate_rng_v1",
            "pair_index": scheduled_game.pair_index,
            "swap_index": scheduled_game.swap_index,
            "episode_seed": scheduled_game.episode_seed,
            "seat": int(seat),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
    )
    return stable_hash64(payload)


def _periodic_dev_eval_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "periodic_dev_eval_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def _promotion_gate_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "promotion_gate_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def _clone_eval_model(
    *,
    learner_model: PolicyValueModel,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    eval_device: torch.device | str | None = None,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")
    resolved_eval_device = _resolve_eval_device(stack, eval_device=eval_device)
    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(resolved_eval_device)
    eval_state_dict = {
        name: value.detach().to(device=resolved_eval_device).clone() for name, value in learner_model.state_dict().items()
    }
    eval_model.load_state_dict(eval_state_dict)
    _restore_model_guidance_from_payload(eval_model, _model_guidance_payload(learner_model))
    eval_model.eval()
    return eval_model


def _current_focal_policy_id(*, learner: ImpalaLearner) -> str:
    return f"train_u{int(learner.update_count)}_p{int(learner.get_policy_version())}"


def _checkpoint_path_for_update(checkpoints_dir: Path, *, update_count: int) -> Path:
    return checkpoints_dir / f"checkpoint_{update_count}.pt"


def _ensure_current_checkpoint(
    *,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> Path:
    checkpoint_path = _checkpoint_path_for_update(
        training_paths.checkpoints_dir,
        update_count=int(learner.update_count),
    )
    if checkpoint_path.is_file():
        return checkpoint_path

    _write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    return checkpoint_path


def _should_run_periodic_dev_eval(stack: StackConfig, *, update_count: int) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    interval = int(evaluation.periodic_dev_eval_interval_updates)
    return interval > 0 and update_count % interval == 0


def _should_defer_noleague_baseline_alias_refresh(
    *,
    stack: StackConfig,
    experiment_role: str,
    update_count: int,
) -> bool:
    return _is_noleague_baseline_role(experiment_role) and _should_run_periodic_dev_eval(
        stack,
        update_count=update_count,
    )


def _periodic_dev_eval_summaries_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "periodic_dev_eval_summaries.json"


def _stall_monitor_state_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "stall_monitor.json"


def _early_cutoff_state_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "early_cutoff.json"


def _early_cutoff_events_path(training_paths: TrainingPaths) -> Path:
    return training_paths.logs_dir / "early_cutoff_events.jsonl"


def _append_early_cutoff_event(training_paths: TrainingPaths, payload: Mapping[str, Any]) -> None:
    path = _early_cutoff_events_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _periodic_dev_eval_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
) -> list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]]:
    evaluation = _evaluation_config_or_raise(stack)
    registry_path = ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path) if registry_path.is_file() else SnapshotRegistry()
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Periodic dev eval is missing required anchors: {missing_text}")

    league = stack.config.league
    anchor_names: list[str]
    if league is None:
        anchor_names = [_PROMOTION_GATE_RANDOMLEGAL_NAME, _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME]
    else:
        anchor_names = [
            *league.promotion_anchor_set_v1.required,
            *league.promotion_anchor_set_v1.optional_if_available,
        ]

    snapshot_index = _snapshot_meta_by_policy_id(registry)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    opponents: list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]] = []
    for anchor_name in anchor_names:
        policy_id = anchor_policy_ids.get(anchor_name)
        if policy_id is None:
            continue
        if policy_id == _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            opponents.append((policy_id, anchor_name, None, None))
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            try:
                heuristic_policy = _build_heuristic_public_policy(
                    contract.spec_bundle,
                    scoring_profile=heuristic_profile,
                )
            except Exception as exc:
                if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                    raise RuntimeError(
                        f"Periodic dev eval requires a heuristic-compatible simulator contract for {policy_id}"
                    ) from exc
                continue
            opponents.append((policy_id, anchor_name, None, heuristic_policy))
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                raise RuntimeError(f"Periodic dev eval could not resolve required snapshot anchor {anchor_name!r}")
            continue
        opponents.append(
            (
                policy_id,
                anchor_name,
                _load_snapshot_eval_model(
                    run_dir=run_dir,
                    snapshot_path=snapshot.path,
                    stack=stack,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    eval_device=evaluation.eval_device,
                    observation_spec=observation_spec,
                    spec_bundle=spec_bundle,
                ),
                None,
            )
        )
    return opponents


def _resolve_periodic_dev_eval_opponent_specs(
    *,
    stack: StackConfig,
    run_dir: Path,
) -> tuple[tuple[PeriodicDevEvalOpponentSpec, ...], tuple[str, ...]]:
    registry_path = ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path) if registry_path.is_file() else SnapshotRegistry()
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Periodic dev eval is missing required anchors: {missing_text}")

    league = stack.config.league
    anchor_names: list[str]
    if league is None:
        anchor_names = [_PROMOTION_GATE_RANDOMLEGAL_NAME, _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME]
    else:
        anchor_names = [
            *league.promotion_anchor_set_v1.required,
            *league.promotion_anchor_set_v1.optional_if_available,
        ]

    snapshot_index = _snapshot_meta_by_policy_id(registry)
    specs: list[PeriodicDevEvalOpponentSpec] = []
    pinned_snapshot_ids: list[str] = []
    for anchor_name in anchor_names:
        policy_id = anchor_policy_ids.get(anchor_name)
        if policy_id is None:
            continue
        if policy_id == _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="random_legal",
                )
            )
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="heuristic_public",
                    heuristic_profile=heuristic_profile,
                )
            )
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                raise RuntimeError(f"Periodic dev eval could not resolve required snapshot anchor {anchor_name!r}")
            continue
        specs.append(
            PeriodicDevEvalOpponentSpec(
                policy_id=policy_id,
                display_name=anchor_name,
                kind="snapshot",
                snapshot_path=snapshot.path,
            )
        )
        pinned_snapshot_ids.append(policy_id)
    return tuple(specs), tuple(dict.fromkeys(pinned_snapshot_ids))


def _materialize_periodic_dev_eval_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: torch.device | str | None = None,
) -> list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]]:
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    evaluation = _evaluation_config_or_raise(stack)
    opponents: list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]] = []
    for spec in opponent_specs:
        if spec.kind == "random_legal":
            opponents.append((spec.policy_id, spec.display_name, None, None))
            continue
        if spec.kind == "heuristic_public":
            heuristic_profile = str(spec.heuristic_profile or "").strip()
            if not heuristic_profile:
                raise RuntimeError(f"Periodic dev eval heuristic opponent is missing a profile: {spec.policy_id}")
            heuristic_policy = _build_heuristic_public_policy(
                contract.spec_bundle,
                scoring_profile=heuristic_profile,
            )
            opponents.append((spec.policy_id, spec.display_name, None, heuristic_policy))
            continue
        if spec.kind != "snapshot" or spec.snapshot_path is None:
            raise RuntimeError(f"Unsupported periodic dev eval opponent kind: {spec.kind!r}")
        opponents.append(
            (
                spec.policy_id,
                spec.display_name,
                _load_snapshot_eval_model(
                    run_dir=run_dir,
                    snapshot_path=spec.snapshot_path,
                    stack=stack,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
                    observation_spec=observation_spec,
                    spec_bundle=spec_bundle,
                ),
                None,
            )
        )
    return opponents


def _persist_periodic_dev_eval_summary(
    *,
    training_paths: TrainingPaths,
    payload: Mapping[str, Any],
) -> None:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    if not focal_policy_id:
        return
    path = _periodic_dev_eval_summaries_path(training_paths)
    summaries = _load_json_object(path, label="periodic dev-eval summaries") if path.is_file() else {}
    summaries[focal_policy_id] = _build_periodic_dev_eval_summary_record(
        payload=payload,
        prior_summaries=summaries,
    )
    _write_json(path, summaries)


def _build_checkpoint_record_for_update(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    artifacts: RunArtifacts,
    update_count: int,
    policy_version: int,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias_name,
        "alias_path": _relative_path_text(alias_path, root=artifacts.run_dir),
        "source_checkpoint_path": _relative_path_text(source_checkpoint_path, root=artifacts.run_dir),
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "metric_kind": metric_kind,
        "metric_value": metric_value,
    }


def _publish_best_checkpoint_from_dev_eval(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    checkpoint_path: Path,
    update_count: int,
    policy_version: int,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    tracker = _load_checkpoint_tracker(training_paths)
    candidate_kind, candidate_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics=None,
        dev_eval_summary=dev_eval_summary,
    )
    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        best_record = None
    should_update_best = candidate_kind is not None and (
        best_record is None
        or _should_promote_best_checkpoint(
            existing_record=cast(Mapping[str, Any], best_record),
            candidate_kind=candidate_kind,
            candidate_value=candidate_value,
        )
    )
    if should_update_best:
        shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
        tracker["best"] = _build_checkpoint_record_for_update(
            alias_name="best",
            alias_path=training_paths.best_checkpoint_path,
            source_checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            update_count=update_count,
            policy_version=policy_version,
            metric_kind=candidate_kind,
            metric_value=candidate_value,
        )
    _update_secondary_b2_checkpoint_record(
        tracker=tracker,
        stack=stack,
        artifacts=artifacts,
        source_checkpoint_path=checkpoint_path,
        update_count=int(update_count),
        policy_version=int(policy_version),
        dev_eval_summary=dev_eval_summary,
    )
    _write_checkpoint_tracker(training_paths, tracker)
    return tracker


def _update_stall_monitor(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.stall_monitor.enabled:
        return None
    threshold = float(curriculum.stall_monitor.truncation_rate_threshold)
    required_consecutive = int(curriculum.stall_monitor.consecutive_evals)
    anchors_raw = summary_payload.get("anchors", {})
    if not isinstance(anchors_raw, Mapping):
        return None

    anchor_truncation_rates: dict[str, float] = {}
    anchor_no_progress_rates: dict[str, float] = {}
    anchor_natural_timeout_rates: dict[str, float] = {}
    anchor_stall_rates: dict[str, float] = {}
    for anchor_name, anchor_payload in anchors_raw.items():
        if not isinstance(anchor_payload, Mapping):
            continue
        matchup_summary = anchor_payload.get("summary", {})
        if not isinstance(matchup_summary, Mapping):
            continue
        truncation_rate = _summary_rate(matchup_summary, "truncations")
        no_progress_rate = _summary_rate(matchup_summary, "no_progress_timeouts")
        natural_timeout_rate = _summary_rate(matchup_summary, "natural_timeouts")
        if truncation_rate is None and no_progress_rate is None and natural_timeout_rate is None:
            continue
        anchor_truncation_rates[anchor_name] = 0.0 if truncation_rate is None else truncation_rate
        anchor_no_progress_rates[anchor_name] = 0.0 if no_progress_rate is None else no_progress_rate
        anchor_natural_timeout_rates[anchor_name] = 0.0 if natural_timeout_rate is None else natural_timeout_rate
        anchor_stall_rates[anchor_name] = (
            anchor_no_progress_rates[anchor_name]
            if no_progress_rate is not None
            else anchor_truncation_rates[anchor_name]
        )
    if not anchor_stall_rates:
        return None

    state_path = _stall_monitor_state_path(training_paths)
    state = _load_json_object(state_path, label="stall monitor state") if state_path.is_file() else {}
    previous_consecutive = int(state.get("consecutive_trigger_count", 0))
    worst_anchor = max(anchor_stall_rates, key=anchor_stall_rates.get)
    worst_rate = float(anchor_stall_rates[worst_anchor])
    consecutive = previous_consecutive + 1 if worst_rate >= threshold else 0
    stall_risk = consecutive >= required_consecutive
    payload = {
        "enabled": True,
        "update_count": int(update_count),
        "threshold": threshold,
        "required_consecutive_evals": required_consecutive,
        "consecutive_trigger_count": consecutive,
        "stall_risk": stall_risk,
        "worst_anchor": worst_anchor,
        "stall_indicator_kind": (
            "no_progress_timeout" if anchor_no_progress_rates.get(worst_anchor, 0.0) > 0.0 else "truncation_fallback"
        ),
        "worst_stall_rate": worst_rate,
        "worst_truncation_rate": float(anchor_truncation_rates.get(worst_anchor, 0.0)),
        "worst_no_progress_timeout_rate": float(anchor_no_progress_rates.get(worst_anchor, 0.0)),
        "worst_natural_timeout_rate": float(anchor_natural_timeout_rates.get(worst_anchor, 0.0)),
        "anchor_truncation_rates": anchor_truncation_rates,
        "anchor_no_progress_timeout_rates": anchor_no_progress_rates,
        "anchor_natural_timeout_rates": anchor_natural_timeout_rates,
    }
    _write_json(state_path, payload)
    return payload


def _update_early_cutoff(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.early_cutoff.enabled:
        return None
    current_score = _dev_eval_aggregate_score(summary_payload)
    if current_score is None:
        return None

    early_cutoff = curriculum.early_cutoff
    state_path = _early_cutoff_state_path(training_paths)
    state = _load_json_object(state_path, label="early cutoff state") if state_path.is_file() else {}
    previous_best_score = state.get("best_score")
    previous_best_update = state.get("best_update_count")
    previous_consecutive_stall = int(state.get("consecutive_stall_evals", 0))

    improved = False
    if isinstance(previous_best_score, (int, float)) and np.isfinite(float(previous_best_score)):
        best_score = float(previous_best_score)
        best_update_count = (
            int(previous_best_update) if isinstance(previous_best_update, int) else int(update_count)
        )
        if float(current_score) > best_score + float(early_cutoff.min_improvement):
            best_score = float(current_score)
            best_update_count = int(update_count)
            improved = True
    else:
        best_score = float(current_score)
        best_update_count = int(update_count)
        improved = True

    patience_reference_update = max(int(best_update_count), int(early_cutoff.warmup_updates))
    no_improvement_updates = max(0, int(update_count) - patience_reference_update)
    worst_stall_rate = _dev_eval_worst_stall_rate(summary_payload)
    if (
        worst_stall_rate is not None
        and worst_stall_rate >= float(early_cutoff.stall_rate_threshold)
    ):
        consecutive_stall_evals = previous_consecutive_stall + 1
    else:
        consecutive_stall_evals = 0

    reasons: list[str] = []
    if (
        int(early_cutoff.patience_updates) > 0
        and int(update_count) >= int(early_cutoff.warmup_updates)
        and no_improvement_updates >= int(early_cutoff.patience_updates)
    ):
        reasons.append("no_improvement")
    if (
        int(early_cutoff.stall_patience_evals) > 0
        and consecutive_stall_evals >= int(early_cutoff.stall_patience_evals)
    ):
        reasons.append("stall")

    payload = {
        "enabled": True,
        "update_count": int(update_count),
        "current_score": float(current_score),
        "best_score": float(best_score),
        "best_update_count": int(best_update_count),
        "improved": bool(improved),
        "min_improvement": float(early_cutoff.min_improvement),
        "warmup_updates": int(early_cutoff.warmup_updates),
        "patience_updates": int(early_cutoff.patience_updates),
        "no_improvement_updates": int(no_improvement_updates),
        "stall_patience_evals": int(early_cutoff.stall_patience_evals),
        "stall_rate_threshold": float(early_cutoff.stall_rate_threshold),
        "worst_stall_rate": None if worst_stall_rate is None else float(worst_stall_rate),
        "consecutive_stall_evals": int(consecutive_stall_evals),
        "should_stop": bool(reasons),
        "reasons": reasons,
    }
    _write_json(state_path, payload)
    if reasons:
        _append_early_cutoff_event(
            training_paths,
            {
                "format": "early_cutoff_event_v1",
                **payload,
            },
        )
    return payload


def _maybe_rollback_to_best_checkpoint(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    model: PolicyValueModel,
    device: torch.device,
    spec_hash256: str,
    algorithm: str,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
    last_rollback_update: int | None,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if not checkpoint_guard.enabled or dev_eval_summary is None:
        return None
    if not _dev_eval_is_authoritative(dev_eval_summary):
        return None
    if last_rollback_update is not None and (int(learner.update_count) - int(last_rollback_update)) < int(
        checkpoint_guard.cooldown_updates
    ):
        return None

    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    worst_truncation_rate = _dev_eval_worst_truncation_rate(dev_eval_summary)
    worst_stall_rate = _dev_eval_worst_stall_rate(dev_eval_summary)
    worst_no_progress_timeout_rate = _dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    worst_natural_timeout_rate = _dev_eval_worst_natural_timeout_rate(dev_eval_summary)
    tracker = _load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not np.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int) or int(best_update_count) >= int(learner.update_count):
        return None
    best_score = float(best_metric_value)
    if best_score < float(checkpoint_guard.min_best_score):
        return None

    pre_rollback_update_count = int(learner.update_count)
    pre_rollback_policy_version = int(learner.get_policy_version())
    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    rollback_reasons: list[str] = []
    if current_score <= best_score - float(checkpoint_guard.rollback_score_margin):
        rollback_reasons.append("score_drop")
    if worst_stall_rate is not None and (
        worst_stall_rate >= float(checkpoint_guard.rollback_truncation_rate_threshold)
    ):
        rollback_reasons.append("truncation")
    max_prob_lt_half = confidence["max_prob_lt_half"]
    if (
        current_score < best_score
        and max_prob_lt_half is not None
        and (float(max_prob_lt_half) >= float(checkpoint_guard.rollback_max_prob_lt_half))
    ):
        rollback_reasons.append("confidence")
    if not rollback_reasons:
        return None

    best_checkpoint_path = training_paths.best_checkpoint_path
    _restore_checkpoint_to_latest_alias(
        checkpoint_path=best_checkpoint_path,
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=spec_hash256,
        algorithm=algorithm,
        restore_counters=False,
    )
    learner.update_count = pre_rollback_update_count
    learner.policy_version = max(int(learner.get_policy_version()), pre_rollback_policy_version)
    demoted_champions = _demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    rejected_snapshots = _reject_registry_snapshots_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    publish_metrics = runtime.maybe_publish_snapshot(
        learner_model=model,
        learner_update_count=int(learner.update_count),
        force=True,
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    tracker["latest"] = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval_mean",
        metric_value=best_score,
    )
    _write_checkpoint_tracker(training_paths, tracker)

    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "rollback_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "restored_weight_update_count": int(best_update_count),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "worst_stall_rate": worst_stall_rate,
        "worst_truncation_rate": worst_truncation_rate,
        "worst_no_progress_timeout_rate": worst_no_progress_timeout_rate,
        "worst_natural_timeout_rate": worst_natural_timeout_rate,
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "reasons": rollback_reasons,
        "best_checkpoint_path": _relative_path_text(best_checkpoint_path, root=artifacts.run_dir),
        "latest_checkpoint_path": _relative_path_text(training_paths.latest_checkpoint_path, root=artifacts.run_dir),
        "snapshot_publish_latency_ms": publish_metrics.get("snapshot_publish_latency_ms", 0.0),
        "snapshot_apply_latency_ms": publish_metrics.get("snapshot_apply_latency_ms", 0.0),
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "demoted_champions": demoted_champions,
        "rejected_snapshots": rejected_snapshots,
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _maybe_finalize_from_best_checkpoint(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    device: torch.device,
    spec_hash256: str,
    algorithm: str,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.checkpoint_guard.enabled:
        return None
    if dev_eval_summary is not None and not _dev_eval_is_authoritative(dev_eval_summary):
        return None
    best_record = _best_checkpoint_record(training_paths)
    if best_record is None:
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not np.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int):
        return None
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    best_score = float(best_metric_value)
    if current_score is None or current_score >= best_score:
        return None
    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    best_checkpoint_path = training_paths.best_checkpoint_path
    _restore_checkpoint_to_latest_alias(
        checkpoint_path=best_checkpoint_path,
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    demoted_champions = _demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    rejected_snapshots = _reject_registry_snapshots_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    tracker = _load_checkpoint_tracker(training_paths)
    tracker["latest"] = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval_mean",
        metric_value=best_score,
    )
    _write_checkpoint_tracker(training_paths, tracker)
    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "finalize_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "best_checkpoint_path": _relative_path_text(best_checkpoint_path, root=artifacts.run_dir),
        "latest_checkpoint_path": _relative_path_text(training_paths.latest_checkpoint_path, root=artifacts.run_dir),
        "demoted_champions": demoted_champions,
        "rejected_snapshots": rejected_snapshots,
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _resolved_periodic_dev_eval_worker_devices(
    *,
    stack: StackConfig,
    parallel_workers: int,
    explicit_worker_devices: Sequence[str],
    eval_device: str,
    learner_device: torch.device | None = None,
) -> tuple[str, ...]:
    if parallel_workers < 1:
        raise ValueError("periodic dev eval parallel_workers must be >= 1")

    normalized_explicit = tuple(device.strip() for device in explicit_worker_devices if str(device).strip())
    if normalized_explicit:
        device_pool = normalized_explicit
        _validate_parallel_worker_device_pool(device_pool, source="periodic_dev_eval_parallel_worker_devices")
    else:
        normalized_eval_device = str(eval_device).strip().lower()
        if normalized_eval_device in {"auto", "cuda:auto"} and torch.cuda.is_available():
            if learner_device is not None:
                actor_count = 1 if stack.config.system is None else int(stack.config.system.actor_process_count)
                actor_layout = resolve_actor_device_layout(
                    stack,
                    actor_count=actor_count,
                    learner_device=learner_device,
                    prefer_process_collectors=True,
                )
                actor_pool = tuple(
                    device_name for device_name in dict.fromkeys(actor_layout) if torch.device(device_name).type == "cuda"
                )
                if actor_pool:
                    device_pool = actor_pool
                else:
                    device_pool = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
            else:
                device_pool = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
        elif normalized_eval_device in {"auto", "cuda:auto"}:
            device_pool = ("cpu",)
        else:
            device_pool = (str(eval_device).strip() or "cpu",)
            _validate_parallel_worker_device_pool(device_pool, source="periodic_dev_eval eval_device")
    return tuple(device_pool[index % len(device_pool)] for index in range(parallel_workers))


def _validate_parallel_worker_device_pool(device_pool: Sequence[str], *, source: str) -> None:
    for device_text in device_pool:
        try:
            device = torch.device(str(device_text).strip())
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"{source} contains invalid device {device_text!r}") from exc
        if device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise ValueError(f"{source} requested CUDA device {device_text!r}, but CUDA is not available")
        device_count = int(torch.cuda.device_count())
        if device.index is not None and device.index >= device_count:
            raise ValueError(
                f"{source} requested CUDA device {device_text!r}, but only {device_count} CUDA device(s) are available"
            )


def _shard_periodic_dev_eval_opponents(
    *,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    shard_count: int,
) -> list[list[PeriodicDevEvalOpponentSpec]]:
    if shard_count < 1:
        raise ValueError("periodic dev eval shard_count must be >= 1")
    shards: list[list[PeriodicDevEvalOpponentSpec]] = [[] for _ in range(shard_count)]
    for index, opponent_spec in enumerate(opponent_specs):
        shards[index % shard_count].append(opponent_spec)
    return [shard for shard in shards if shard]


def _periodic_dev_eval_duplicate_policy_ids(
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
) -> set[str]:
    counts: dict[str, int] = {}
    for spec in opponent_specs:
        counts[spec.policy_id] = counts.get(spec.policy_id, 0) + 1
    return {policy_id for policy_id, count in counts.items() if count > 1}


def _periodic_dev_eval_matchup_dir(
    *,
    update_dir: Path,
    opponent_spec: PeriodicDevEvalOpponentSpec,
    duplicate_policy_ids: set[str],
) -> Path:
    if opponent_spec.policy_id not in duplicate_policy_ids:
        return update_dir / opponent_spec.policy_id
    display_hash = f"{stable_hash64(opponent_spec.display_name.encode('utf-8')):016x}"
    return update_dir / f"{opponent_spec.policy_id}__{display_hash}"


def _split_periodic_dev_eval_seed_blocks(
    paired_seeds: Sequence[int],
    *,
    block_count: int,
) -> list[tuple[tuple[int, int], ...]]:
    if block_count < 1:
        raise ValueError("periodic dev eval seed block_count must be >= 1")
    indexed_seeds = tuple((index, int(seed)) for index, seed in enumerate(paired_seeds))
    if not indexed_seeds:
        return []
    effective_block_count = min(int(block_count), len(indexed_seeds))
    base_block_size, remainder = divmod(len(indexed_seeds), effective_block_count)
    blocks: list[tuple[tuple[int, int], ...]] = []
    start = 0
    for block_index in range(effective_block_count):
        block_size = base_block_size + (1 if block_index < remainder else 0)
        blocks.append(tuple(indexed_seeds[start : start + block_size]))
        start += block_size
    return blocks


def _build_periodic_dev_eval_seed_block_jobs(
    *,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    paired_seeds: Sequence[int],
    configured_parallel_workers: int,
) -> list[PeriodicDevEvalSeedBlockJob]:
    if configured_parallel_workers < 1:
        raise ValueError("periodic dev eval configured_parallel_workers must be >= 1")
    if not opponent_specs:
        return []
    per_opponent_block_count = max(
        1,
        min(
            len(paired_seeds),
            int(math.ceil(configured_parallel_workers / max(1, len(opponent_specs)))),
        ),
    )
    seed_blocks = _split_periodic_dev_eval_seed_blocks(
        paired_seeds,
        block_count=per_opponent_block_count,
    )
    jobs: list[PeriodicDevEvalSeedBlockJob] = []
    for opponent_index, opponent_spec in enumerate(opponent_specs):
        for block_index, paired_seed_items in enumerate(seed_blocks):
            jobs.append(
                PeriodicDevEvalSeedBlockJob(
                    opponent_index=opponent_index,
                    block_index=block_index,
                    opponent_spec=opponent_spec,
                    paired_seed_items=paired_seed_items,
                )
            )
    return jobs


def _shard_periodic_dev_eval_seed_block_jobs(
    *,
    jobs: Sequence[PeriodicDevEvalSeedBlockJob],
    shard_count: int,
) -> list[list[PeriodicDevEvalSeedBlockJob]]:
    if shard_count < 1:
        raise ValueError("periodic dev eval seed-block shard_count must be >= 1")
    shards: list[list[PeriodicDevEvalSeedBlockJob]] = [[] for _ in range(shard_count)]
    for index, job in enumerate(jobs):
        shards[index % shard_count].append(job)
    return [shard for shard in shards if shard]


def _periodic_dev_eval_schedule_for_seed_items(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seed_items: Sequence[tuple[int, int]],
) -> list[ScheduledGame]:
    schedule: list[ScheduledGame] = []
    for pair_index, raw_seed in paired_seed_items:
        episode_seed = int(raw_seed)
        schedule.append(
            ScheduledGame(
                pair_index=int(pair_index),
                swap_index=0,
                episode_index=int(pair_index) * 2,
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=focal_policy_id,
                seat1_policy_id=opponent_policy_id,
                focal_seat=0,
            )
        )
        schedule.append(
            ScheduledGame(
                pair_index=int(pair_index),
                swap_index=1,
                episode_index=int(pair_index) * 2 + 1,
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=opponent_policy_id,
                seat1_policy_id=focal_policy_id,
                focal_seat=1,
            )
        )
    return schedule


def _sum_periodic_dev_eval_counter_payloads(counter_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for payload in counter_payloads:
        for key, value in cast(Mapping[str, Any], payload.get("seconds", {})).items():
            seconds[str(key)] = seconds.get(str(key), 0.0) + float(value)
        for key, value in cast(Mapping[str, Any], payload.get("counts", {})).items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    return {
        "seconds": {key: float(value) for key, value in sorted(seconds.items())},
        "counts": {key: int(value) for key, value in sorted(counts.items())},
    }


def _resolve_async_periodic_dev_eval_device(
    *,
    stack: StackConfig,
    learner_device: torch.device,
) -> str | None:
    evaluation = _evaluation_config_or_raise(stack)
    requested = str(evaluation.eval_device).strip()
    if not requested:
        return None
    normalized = requested.lower()
    if normalized not in {"auto", "cuda:auto"}:
        return requested
    actor_count = 1 if stack.config.system is None else int(stack.config.system.actor_process_count)
    actor_layout = resolve_actor_device_layout(
        stack,
        actor_count=actor_count,
        learner_device=learner_device,
        prefer_process_collectors=True,
    )
    unique_cuda_devices = [
        device_name
        for device_name in dict.fromkeys(actor_layout)
        if torch.device(device_name).type == "cuda" and str(device_name) != str(learner_device)
    ]
    if unique_cuda_devices:
        return str(unique_cuda_devices[-1])
    return requested


def _resolve_async_promotion_gate_device(
    *,
    stack: StackConfig,
    learner_device: torch.device,
) -> str | None:
    evaluation = _evaluation_config_or_raise(stack)
    requested = str(evaluation.eval_device).strip()
    if not requested:
        return None
    normalized = requested.lower()
    if normalized not in {"auto", "cuda:auto"}:
        return requested
    actor_count = 1 if stack.config.system is None else int(stack.config.system.actor_process_count)
    actor_layout = resolve_actor_device_layout(
        stack,
        actor_count=actor_count,
        learner_device=learner_device,
        prefer_process_collectors=True,
    )
    unique_cuda_devices = [
        device_name
        for device_name in dict.fromkeys(actor_layout)
        if torch.device(device_name).type == "cuda" and str(device_name) != str(learner_device)
    ]
    if unique_cuda_devices:
        return str(unique_cuda_devices[0])
    return requested


def _resolve_promotion_gate_anchor_specs(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
) -> tuple[dict[str, str], tuple[PeriodicDevEvalOpponentSpec, ...], tuple[str, ...]]:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Promotion gate is missing required anchors: {missing_text}")

    snapshot_index = _snapshot_meta_by_policy_id(registry)
    specs: list[PeriodicDevEvalOpponentSpec] = []
    pinned_snapshot_ids: list[str] = []
    for anchor_name, policy_id in anchor_policy_ids.items():
        if policy_id == _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="random_legal",
                )
            )
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            specs.append(
                PeriodicDevEvalOpponentSpec(
                    policy_id=policy_id,
                    display_name=anchor_name,
                    kind="heuristic_public",
                    heuristic_profile=heuristic_profile,
                )
            )
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            raise RuntimeError(f"Promotion gate could not resolve snapshot anchor for policy_id={policy_id}")
        specs.append(
            PeriodicDevEvalOpponentSpec(
                policy_id=policy_id,
                display_name=anchor_name,
                kind="snapshot",
                snapshot_path=snapshot.path,
            )
        )
        pinned_snapshot_ids.append(policy_id)
    return dict(anchor_policy_ids), tuple(specs), tuple(dict.fromkeys(pinned_snapshot_ids))


def _resolved_promotion_gate_worker_devices(
    *,
    stack: StackConfig,
    parallel_workers: int,
    explicit_worker_devices: Sequence[str],
    eval_device: str,
) -> tuple[str, ...]:
    if parallel_workers < 1:
        raise ValueError("promotion gate parallel_workers must be >= 1")
    normalized_explicit = tuple(device.strip() for device in explicit_worker_devices if str(device).strip())
    if normalized_explicit:
        device_pool = normalized_explicit
        _validate_parallel_worker_device_pool(device_pool, source="promotion_gate parallel_worker_devices")
    else:
        normalized_eval_device = str(eval_device).strip().lower()
        if normalized_eval_device in {"auto", "cuda:auto"} and torch.cuda.is_available():
            device_pool = tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
        elif normalized_eval_device in {"auto", "cuda:auto"}:
            device_pool = ("cpu",)
        else:
            device_pool = (str(eval_device).strip() or "cpu",)
            _validate_parallel_worker_device_pool(device_pool, source="promotion_gate eval_device")
    return tuple(device_pool[index % len(device_pool)] for index in range(parallel_workers))


def _shard_promotion_gate_anchor_specs(
    *,
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    shard_count: int,
) -> list[list[tuple[int, PeriodicDevEvalOpponentSpec]]]:
    if shard_count < 1:
        raise ValueError("promotion gate shard_count must be >= 1")
    shards: list[list[tuple[int, PeriodicDevEvalOpponentSpec]]] = [[] for _ in range(shard_count)]
    for index, anchor_spec in enumerate(anchor_specs):
        shards[index % shard_count].append((index, anchor_spec))
    return [shard for shard in shards if shard]


def _build_promotion_gate_seed_block_jobs(
    *,
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    paired_seeds: Sequence[int],
    configured_parallel_workers: int,
) -> list[PromotionGateSeedBlockJob]:
    if configured_parallel_workers < 1:
        raise ValueError("promotion gate configured_parallel_workers must be >= 1")
    if not anchor_specs:
        return []
    per_anchor_block_count = max(
        1,
        min(
            len(paired_seeds),
            int(math.ceil(configured_parallel_workers / max(1, len(anchor_specs)))),
        ),
    )
    seed_blocks = _split_periodic_dev_eval_seed_blocks(
        paired_seeds,
        block_count=per_anchor_block_count,
    )
    jobs: list[PromotionGateSeedBlockJob] = []
    for anchor_index, anchor_spec in enumerate(anchor_specs):
        for block_index, paired_seed_items in enumerate(seed_blocks):
            jobs.append(
                PromotionGateSeedBlockJob(
                    anchor_index=anchor_index,
                    block_index=block_index,
                    anchor_spec=anchor_spec,
                    paired_seed_items=paired_seed_items,
                )
            )
    return jobs


def _shard_promotion_gate_seed_block_jobs(
    *,
    jobs: Sequence[PromotionGateSeedBlockJob],
    shard_count: int,
) -> list[list[PromotionGateSeedBlockJob]]:
    if shard_count < 1:
        raise ValueError("promotion gate seed-block shard_count must be >= 1")
    shards: list[list[PromotionGateSeedBlockJob]] = [[] for _ in range(shard_count)]
    for index, job in enumerate(jobs):
        shards[index % shard_count].append(job)
    return [shard for shard in shards if shard]


def _run_periodic_dev_eval_matchups_for_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str,
    artifact_scope: str,
    paired_seeds: Sequence[int],
    scheduled_paired_seed_count: int,
    validated_sources: Mapping[str, str],
    seed_file: Path,
    seed_file_sha256: str,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: torch.device | str | None,
    batched_inference_override: bool | None = None,
) -> list[dict[str, Any]]:
    evaluation = _validate_periodic_dev_eval_contract(stack)
    observation_dim, action_dim = _spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    update_dir = run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    eval_model = _load_checkpoint_eval_model(
        checkpoint_path=checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
    )
    opponents = _materialize_periodic_dev_eval_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=opponent_specs,
        eval_device_override=eval_device_override,
    )
    matchup_results: list[dict[str, Any]] = []
    duplicate_policy_ids = _periodic_dev_eval_duplicate_policy_ids(opponent_specs)
    for opponent_spec, (opponent_policy_id, display_name, opponent_model, heuristic_policy) in zip(
        opponent_specs,
        opponents,
        strict=True,
    ):
        matchup_dir = _periodic_dev_eval_matchup_dir(
            update_dir=update_dir,
            opponent_spec=opponent_spec,
            duplicate_policy_ids=duplicate_policy_ids,
        )
        runner = _PeriodicDevEvalRunner(
            stack=stack,
            model=eval_model,
            opponent_policy_id=opponent_policy_id,
            opponent_model=opponent_model,
            heuristic_policy=heuristic_policy,
            observation_dim=observation_dim,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            artifact_dir=matchup_dir,
            focal_policy_id=focal_policy_id,
            require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
            eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
        )
        matchup_started = time.perf_counter()

        seed_usage_payload = {
            "seed_set": "dev_eval",
            "seed_file": {
                "path": _json_relative_path(seed_file, root=stack.root),
                "sha256": seed_file_sha256,
                "validated_sources": dict(validated_sources),
            },
            "artifact_scope": artifact_scope,
            "seed_schedule": {
                "configured_paired_seed_count": int(scheduled_paired_seed_count),
                "requested_paired_seed_count": len(paired_seeds),
                "expanded_beyond_seed_file": len(paired_seeds) > int(scheduled_paired_seed_count),
            },
            "paired_seed_count": len(paired_seeds),
            "paired_seeds": list(paired_seeds),
            "protocol": {
                "seat_swap": bool(evaluation.seat_swap),
                "eval_device": str(evaluation.eval_device if eval_device_override is None else eval_device_override),
                "eval_inference_mode": bool(evaluation.eval_inference_mode),
                "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
                "eval_assert_sorted_legal_ids": bool(evaluation.eval_assert_sorted_legal_ids),
            },
            "focal_policy": {
                "policy_id": focal_policy_id,
                "update_count": update_count,
                "policy_version": policy_version,
                "checkpoint_path": _json_relative_path(checkpoint_path, root=run_dir),
            },
            "opponent_policy": {
                "policy_id": opponent_policy_id,
                "display_name": display_name,
            },
        }
        _write_json(matchup_dir / "seed_usage.json", seed_usage_payload)

        batched_inference_enabled = (
            bool(getattr(evaluation, "periodic_dev_eval_batched_inference_enabled", False))
            if batched_inference_override is None
            else bool(batched_inference_override)
        )
        try:
            if batched_inference_enabled:
                scheduled_games = build_seat_swapped_schedule(
                    focal_policy_id=focal_policy_id,
                    opponent_policy_id=opponent_policy_id,
                    paired_seeds=paired_seeds,
                )
                completed_games = runner.run_scheduled_games_batched(scheduled_games)
                records = tuple(
                    record_completed_game(
                        scheduled_game=scheduled_game,
                        result=result,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    for scheduled_game, result in completed_games
                )
                write_episodes_jsonl(matchup_dir / "episodes.jsonl", records)
            else:
                matchup = run_seat_swapped_matchup(
                    focal_policy_id=focal_policy_id,
                    opponent_policy_id=opponent_policy_id,
                    paired_seeds=paired_seeds,
                    runner=runner,
                    episodes_path=matchup_dir / "episodes.jsonl",
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                )
                records = matchup.records
            runner_counters = runner.drain_runtime_counters()
        finally:
            close_runner = getattr(runner, "close", None)
            if callable(close_runner):
                close_runner()
        matchup_wall_clock_seconds = max(0.0, time.perf_counter() - matchup_started)

        matchup_payload = build_matchup_export(
            records,
            stop_rules=evaluation.stop_rules,
            max_paired_seeds=len(paired_seeds),
            scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
            sample_count=1000,
            seed=_periodic_dev_eval_bootstrap_seed(update_count=update_count, policy_version=policy_version),
        )
        seat_diagnostics = build_seat_advantage_diagnostics(records)
        matchup_payload["seat_diagnostics"] = seat_diagnostics
        matchup_payload["evaluation_context"] = {
            "artifact_scope": artifact_scope,
            "update_count": update_count,
            "policy_version": policy_version,
            "checkpoint_path": _json_relative_path(checkpoint_path, root=run_dir),
            "matchup_dir": _json_relative_path(matchup_dir, root=run_dir),
            "episodes_path": _json_relative_path(matchup_dir / "episodes.jsonl", root=run_dir),
            "seed_usage_path": _json_relative_path(matchup_dir / "seed_usage.json", root=run_dir),
            "anchor_display_name": display_name,
        }
        matchup_payload["evaluation_runtime"] = {
            "wall_clock_seconds": matchup_wall_clock_seconds,
            "games_per_sec": float(len(records) / matchup_wall_clock_seconds)
            if matchup_wall_clock_seconds > 0.0
            else 0.0,
            "game_count": int(len(records)),
            "persistent_env_reuse": True,
            "batched_model_inference": batched_inference_enabled,
            "runner_counters": runner_counters,
        }
        write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
        write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", matchup_payload)
        write_matchup_diagnostics_json(
            matchup_dir / "diagnostics.json",
            seat_diagnostics,
        )
        matchup_results.append(
            {
                "policy_id": opponent_policy_id,
                "display_name": display_name,
                "matchup_dir": matchup_dir,
                "episodes_path": matchup_dir / "episodes.jsonl",
                "matchup_payload": matchup_payload,
            }
        )
    return matchup_results


def _run_periodic_dev_eval_matchup_worker(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str,
    artifact_scope: str,
    paired_seeds: Sequence[int],
    scheduled_paired_seed_count: int,
    validated_sources: Mapping[str, str],
    seed_file: Path,
    seed_file_sha256: str,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: str,
    batched_inference_override: bool | None = None,
) -> list[dict[str, Any]]:
    return _run_periodic_dev_eval_matchups_for_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        focal_policy_id=focal_policy_id,
        update_count=update_count,
        policy_version=policy_version,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        artifact_dir_name=artifact_dir_name,
        artifact_scope=artifact_scope,
        paired_seeds=paired_seeds,
        scheduled_paired_seed_count=scheduled_paired_seed_count,
        validated_sources=validated_sources,
        seed_file=seed_file,
        seed_file_sha256=seed_file_sha256,
        opponent_specs=opponent_specs,
        eval_device_override=eval_device_override,
        batched_inference_override=batched_inference_override,
    )


def _run_periodic_dev_eval_seed_block_worker(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str,
    jobs: Sequence[PeriodicDevEvalSeedBlockJob],
    eval_device_override: str,
    worker_index: int,
    batched_inference_override: bool | None = None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    evaluation = _validate_periodic_dev_eval_contract(stack)
    batched_inference_enabled = (
        bool(getattr(evaluation, "periodic_dev_eval_batched_inference_enabled", False))
        if batched_inference_override is None
        else bool(batched_inference_override)
    )
    observation_dim, action_dim = _spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    update_dir = run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    worker_artifact_dir = update_dir / "_seed_block_workers" / f"worker_{worker_index}"
    eval_model = _load_checkpoint_eval_model(
        checkpoint_path=checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
    )

    unique_specs: list[PeriodicDevEvalOpponentSpec] = []
    unique_spec_keys: set[tuple[str, str, str, str | None, str | None]] = set()
    for job in jobs:
        key = (
            job.opponent_spec.policy_id,
            job.opponent_spec.display_name,
            job.opponent_spec.kind,
            job.opponent_spec.snapshot_path,
            job.opponent_spec.heuristic_profile,
        )
        if key in unique_spec_keys:
            continue
        unique_spec_keys.add(key)
        unique_specs.append(job.opponent_spec)

    materialized = _materialize_periodic_dev_eval_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=tuple(unique_specs),
        eval_device_override=eval_device_override,
    )
    opponent_by_key = {
        (
            spec.policy_id,
            spec.display_name,
            spec.kind,
            spec.snapshot_path,
            spec.heuristic_profile,
        ): opponent
        for spec, opponent in zip(unique_specs, materialized, strict=True)
    }
    runners: dict[tuple[str, str, str, str | None, str | None], _PeriodicDevEvalRunner] = {}
    results: list[dict[str, Any]] = []
    try:
        for job in jobs:
            key = (
                job.opponent_spec.policy_id,
                job.opponent_spec.display_name,
                job.opponent_spec.kind,
                job.opponent_spec.snapshot_path,
                job.opponent_spec.heuristic_profile,
            )
            opponent_policy_id, display_name, opponent_model, heuristic_policy = opponent_by_key[key]
            runner = runners.get(key)
            if runner is None:
                runner = _PeriodicDevEvalRunner(
                    stack=stack,
                    model=eval_model,
                    opponent_policy_id=opponent_policy_id,
                    opponent_model=opponent_model,
                    heuristic_policy=heuristic_policy,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    pass_action_id=pass_action_id,
                    artifact_dir=worker_artifact_dir / f"opponent_{job.opponent_index}",
                    focal_policy_id=focal_policy_id,
                    require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
                    eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
                )
                runners[key] = runner
            block_started = time.perf_counter()
            scheduled_games = _periodic_dev_eval_schedule_for_seed_items(
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                paired_seed_items=job.paired_seed_items,
            )
            if batched_inference_enabled:
                completed_games = runner.run_scheduled_games_batched(scheduled_games)
                records = tuple(
                    record_completed_game(
                        scheduled_game=scheduled_game,
                        result=result,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    for scheduled_game, result in completed_games
                )
            else:
                records = tuple(
                    record_completed_game(
                        scheduled_game=scheduled_game,
                        result=runner.run_game(scheduled_game),
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    for scheduled_game in scheduled_games
                )
            results.append(
                {
                    "opponent_index": int(job.opponent_index),
                    "block_index": int(job.block_index),
                    "policy_id": opponent_policy_id,
                    "display_name": display_name,
                    "paired_seed_items": tuple(job.paired_seed_items),
                    "records": records,
                    "wall_clock_seconds": max(0.0, time.perf_counter() - block_started),
                    "runner_counters": runner.drain_runtime_counters(),
                    "worker_index": int(worker_index),
                    "worker_device": str(eval_device_override),
                }
            )
    finally:
        for runner in runners.values():
            runner.close()
    return results


def _run_periodic_dev_eval_for_checkpoint(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec] | None = None,
    eval_device_override: torch.device | str | None = None,
    parallel_workers_override: int | None = None,
    parallel_worker_devices_override: Sequence[str] | None = None,
    artifact_dir_name: str = "dev_eval",
    artifact_scope: str = "periodic_dev_eval",
    paired_seeds_override: Sequence[int] | None = None,
    batched_inference_override: bool | None = None,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    evaluation = _validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources, scheduled_paired_seeds, seed_file_sha256 = _periodic_dev_eval_schedule(stack)
    eval_started = time.perf_counter()
    paired_seeds = (
        [int(seed) for seed in paired_seeds_override]
        if paired_seeds_override is not None
        else [int(seed) for seed in scheduled_paired_seeds]
    )
    if not paired_seeds:
        raise RuntimeError("Periodic dev eval requires at least one paired seed")
    batched_inference_enabled = (
        bool(getattr(evaluation, "periodic_dev_eval_batched_inference_enabled", False))
        if batched_inference_override is None
        else bool(batched_inference_override)
    )
    observation_dim, action_dim = _spec_dimensions(contract)
    update_dir = run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    resolved_opponent_specs = opponent_specs
    if resolved_opponent_specs is None:
        resolved_opponent_specs, _ignored_pinned_ids = _resolve_periodic_dev_eval_opponent_specs(
            stack=stack,
            run_dir=run_dir,
        )
    requested_eval_device = evaluation.eval_device if eval_device_override is None else str(eval_device_override)
    configured_parallel_workers = max(
        1,
        int(
            getattr(evaluation, "periodic_dev_eval_parallel_workers", 1)
            if parallel_workers_override is None
            else parallel_workers_override
        ),
    )
    explicit_parallel_devices = tuple(
        getattr(evaluation, "periodic_dev_eval_parallel_worker_devices", ())
        if parallel_worker_devices_override is None
        else parallel_worker_devices_override
    )
    seed_block_jobs = _build_periodic_dev_eval_seed_block_jobs(
        opponent_specs=resolved_opponent_specs,
        paired_seeds=tuple(paired_seeds),
        configured_parallel_workers=configured_parallel_workers,
    )
    effective_parallel_workers = min(configured_parallel_workers, max(1, len(seed_block_jobs)))
    matchup_results: list[dict[str, Any]]
    worker_devices: tuple[str, ...]
    worker_devices = _resolved_periodic_dev_eval_worker_devices(
        stack=stack,
        parallel_workers=max(1, effective_parallel_workers),
        explicit_worker_devices=explicit_parallel_devices,
        eval_device=requested_eval_device,
        learner_device=None,
    )
    if effective_parallel_workers > 1:
        job_shards = _shard_periodic_dev_eval_seed_block_jobs(
            jobs=seed_block_jobs,
            shard_count=effective_parallel_workers,
        )
        executor_context = (
            nullcontext(process_pool_executor)
            if process_pool_executor is not None
            else ProcessPoolExecutor(max_workers=len(job_shards), mp_context=mp.get_context("spawn"))
        )
        with executor_context as executor:
            assert executor is not None
            futures = [
                executor.submit(
                    _run_periodic_dev_eval_seed_block_worker,
                    stack=stack,
                    contract=contract,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    focal_policy_id=focal_policy_id,
                    update_count=update_count,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                    artifact_dir_name=artifact_dir_name,
                    jobs=tuple(shard),
                    eval_device_override=worker_devices[shard_index],
                    worker_index=shard_index,
                    batched_inference_override=batched_inference_enabled,
                )
                for shard_index, shard in enumerate(job_shards)
            ]
            block_results: list[dict[str, Any]] = []
            for future in futures:
                block_results.extend(future.result())

        duplicate_policy_ids = _periodic_dev_eval_duplicate_policy_ids(resolved_opponent_specs)
        block_results_by_opponent: dict[int, list[dict[str, Any]]] = {}
        for block_result in block_results:
            block_results_by_opponent.setdefault(int(block_result["opponent_index"]), []).append(block_result)
        matchup_results = []
        for opponent_index, opponent_spec in enumerate(resolved_opponent_specs):
            opponent_blocks = sorted(
                block_results_by_opponent.get(opponent_index, []),
                key=lambda item: int(item["block_index"]),
            )
            if not opponent_blocks:
                raise RuntimeError(f"Periodic dev eval produced no seed-block results for {opponent_spec.display_name}")
            matchup_dir = _periodic_dev_eval_matchup_dir(
                update_dir=update_dir,
                opponent_spec=opponent_spec,
                duplicate_policy_ids=duplicate_policy_ids,
            )
            records = tuple(
                sorted(
                    (
                        record
                        for block_result in opponent_blocks
                        for record in cast(tuple[EvalGameRecord, ...], block_result["records"])
                    ),
                    key=lambda record: (record.pair_index, record.swap_index),
                )
            )
            seed_usage_payload = {
                "seed_set": "dev_eval",
                "seed_file": {
                    "path": _json_relative_path(seed_file, root=stack.root),
                    "sha256": seed_file_sha256,
                    "validated_sources": dict(validated_sources),
                },
                "artifact_scope": artifact_scope,
                "seed_schedule": {
                    "configured_paired_seed_count": int(len(scheduled_paired_seeds)),
                    "requested_paired_seed_count": len(paired_seeds),
                    "expanded_beyond_seed_file": len(paired_seeds) > int(len(scheduled_paired_seeds)),
                },
                "paired_seed_count": len(paired_seeds),
                "paired_seeds": list(paired_seeds),
                "protocol": {
                    "seat_swap": bool(evaluation.seat_swap),
                    "eval_device": str(requested_eval_device),
                    "eval_inference_mode": bool(evaluation.eval_inference_mode),
                    "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
                    "eval_assert_sorted_legal_ids": bool(evaluation.eval_assert_sorted_legal_ids),
                },
                "focal_policy": {
                    "policy_id": focal_policy_id,
                    "update_count": update_count,
                    "policy_version": policy_version,
                    "checkpoint_path": _json_relative_path(checkpoint_path, root=run_dir),
                },
                "opponent_policy": {
                    "policy_id": opponent_spec.policy_id,
                    "display_name": opponent_spec.display_name,
                },
                "parallel_seed_blocks": [
                    {
                        "block_index": int(block_result["block_index"]),
                        "paired_seed_items": [
                            {"pair_index": int(pair_index), "seed": int(seed)}
                            for pair_index, seed in cast(tuple[tuple[int, int], ...], block_result["paired_seed_items"])
                        ],
                        "worker_index": int(block_result["worker_index"]),
                        "worker_device": str(block_result["worker_device"]),
                    }
                    for block_result in opponent_blocks
                ],
            }
            _write_json(matchup_dir / "seed_usage.json", seed_usage_payload)
            write_episodes_jsonl(matchup_dir / "episodes.jsonl", records)
            matchup_payload = build_matchup_export(
                records,
                stop_rules=evaluation.stop_rules,
                max_paired_seeds=len(paired_seeds),
                scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
                sample_count=1000,
                seed=_periodic_dev_eval_bootstrap_seed(update_count=update_count, policy_version=policy_version),
            )
            seat_diagnostics = build_seat_advantage_diagnostics(records)
            matchup_payload["seat_diagnostics"] = seat_diagnostics
            matchup_payload["evaluation_context"] = {
                "artifact_scope": artifact_scope,
                "update_count": update_count,
                "policy_version": policy_version,
                "checkpoint_path": _json_relative_path(checkpoint_path, root=run_dir),
                "matchup_dir": _json_relative_path(matchup_dir, root=run_dir),
                "episodes_path": _json_relative_path(matchup_dir / "episodes.jsonl", root=run_dir),
                "seed_usage_path": _json_relative_path(matchup_dir / "seed_usage.json", root=run_dir),
                "anchor_display_name": opponent_spec.display_name,
            }
            block_wall_clock_seconds = [float(block_result["wall_clock_seconds"]) for block_result in opponent_blocks]
            matchup_wall_clock_seconds = max(block_wall_clock_seconds) if block_wall_clock_seconds else 0.0
            matchup_payload["evaluation_runtime"] = {
                "wall_clock_seconds": matchup_wall_clock_seconds,
                "games_per_sec": float(len(records) / matchup_wall_clock_seconds)
                if matchup_wall_clock_seconds > 0.0
                else 0.0,
                "game_count": int(len(records)),
                "persistent_env_reuse": True,
                "seed_block_count": int(len(opponent_blocks)),
                "batched_model_inference": bool(
                    batched_inference_enabled
                ),
                "serial_worker_wall_clock_seconds_sum": float(sum(block_wall_clock_seconds)),
                "runner_counters": _sum_periodic_dev_eval_counter_payloads(
                    [cast(Mapping[str, Any], block_result["runner_counters"]) for block_result in opponent_blocks]
                ),
            }
            write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
            write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", matchup_payload)
            write_matchup_diagnostics_json(
                matchup_dir / "diagnostics.json",
                seat_diagnostics,
            )
            matchup_results.append(
                {
                    "policy_id": opponent_spec.policy_id,
                    "display_name": opponent_spec.display_name,
                    "matchup_dir": matchup_dir,
                    "episodes_path": matchup_dir / "episodes.jsonl",
                    "matchup_payload": matchup_payload,
                }
            )
    else:
        matchup_results = _run_periodic_dev_eval_matchups_for_opponents(
            stack=stack,
            contract=contract,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            focal_policy_id=focal_policy_id,
            update_count=update_count,
            policy_version=policy_version,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            artifact_dir_name=artifact_dir_name,
            artifact_scope=artifact_scope,
            paired_seeds=tuple(paired_seeds),
            scheduled_paired_seed_count=len(scheduled_paired_seeds),
            validated_sources=validated_sources,
            seed_file=seed_file,
            seed_file_sha256=seed_file_sha256,
            opponent_specs=resolved_opponent_specs,
            eval_device_override=worker_devices[0],
            batched_inference_override=batched_inference_enabled,
        )

    spec_order = {spec.display_name: index for index, spec in enumerate(resolved_opponent_specs)}
    matchup_results.sort(key=lambda item: spec_order.get(str(item["display_name"]), 10**9))
    anchor_payloads: dict[str, dict[str, Any]] = {}
    anchor_scores: dict[str, float] = {}
    primary_summary: dict[str, Any] | None = None
    for result in matchup_results:
        display_name = str(result["display_name"])
        opponent_policy_id = str(result["policy_id"])
        matchup_payload = dict(cast(dict[str, Any], result["matchup_payload"]))
        anchor_payloads[display_name] = matchup_payload
        anchor_scores[display_name] = float(matchup_payload["uncertainty"]["mean"])
        if primary_summary is None or opponent_policy_id == "b0_randomlegal":
            primary_summary = matchup_payload

    if primary_summary is None:
        raise RuntimeError("Periodic dev eval did not produce any matchup summaries")

    unweighted_aggregate_score = sum(anchor_scores.values()) / max(1, len(anchor_scores))
    anchor_weight_config = _periodic_dev_eval_anchor_weight_map(stack)
    aggregate_score, aggregate_anchor_weights, aggregate_weight_sum = _weighted_dev_eval_aggregate(
        anchor_scores,
        anchor_weights=anchor_weight_config,
    )
    summary_payload = dict(primary_summary)
    total_eval_wall_clock_seconds = max(0.0, time.perf_counter() - eval_started)
    total_eval_games = sum(
        int(cast(dict[str, Any], result["matchup_payload"]).get("evaluation_runtime", {}).get("game_count", 0))
        for result in matchup_results
    )
    summary_payload.update(
        {
            "policy_id": focal_policy_id,
            "update_count": update_count,
            "policy_version": policy_version,
            "aggregate_score": aggregate_score,
            "unweighted_aggregate_score": unweighted_aggregate_score,
            "anchor_scores": anchor_scores,
            "anchor_seat_diagnostics": {
                str(result["display_name"]): cast(dict[str, Any], result["matchup_payload"]).get("seat_diagnostics", {})
                for result in matchup_results
            },
            "aggregate_weighting": {
                "version": "periodic_dev_eval_anchor_weights_v1",
                "anchor_weights": aggregate_anchor_weights,
                "configured_anchor_weights": dict(anchor_weight_config),
                "total_weight": float(aggregate_weight_sum),
                "default_weight": 1.0,
            },
            "anchors": anchor_payloads,
            "periodic_dev_eval_parallel": {
                "enabled": effective_parallel_workers > 1,
                "worker_count": int(max(1, effective_parallel_workers)),
                "worker_devices": list(worker_devices[: max(1, effective_parallel_workers)]),
                "job_count": int(len(seed_block_jobs)),
                "batched_inference_enabled": bool(
                    batched_inference_enabled
                ),
                "seed_block_sharding_enabled": any(
                    int(result["matchup_payload"].get("evaluation_runtime", {}).get("seed_block_count", 1)) > 1
                    for result in matchup_results
                ),
            },
            "periodic_dev_eval_runtime": {
                "wall_clock_seconds": total_eval_wall_clock_seconds,
                "games_per_sec": float(total_eval_games / total_eval_wall_clock_seconds)
                if total_eval_wall_clock_seconds > 0.0
                else 0.0,
                "game_count": int(total_eval_games),
                "persistent_env_reuse": True,
            },
            "evaluation_surface": {
                "kind": "fast_batched_screen" if batched_inference_enabled else "canonical_scalar",
                "authoritative": not batched_inference_enabled,
                "batched_inference_enabled": bool(batched_inference_enabled),
            },
        }
    )
    _write_json(update_dir / "summary.json", summary_payload)
    return summary_payload


def _run_async_periodic_dev_eval_worker(
    request: AsyncPeriodicDevEvalRequest,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    contract = load_verified_simulator_contract(request.stack.root, expected_spec_hash=request.spec_hash256)
    effective_eval_device = request.eval_device_override
    effective_worker_devices = (
        request.parallel_worker_devices
        if request.parallel_worker_devices
        else _resolved_periodic_dev_eval_worker_devices(
            stack=request.stack,
            parallel_workers=max(1, int(request.parallel_workers)),
            explicit_worker_devices=(),
            eval_device=str(effective_eval_device or _evaluation_config_or_raise(request.stack).eval_device),
            learner_device=None,
        )
    )
    return _run_periodic_dev_eval_for_checkpoint(
        stack=request.stack,
        contract=contract,
        run_dir=request.run_dir,
        checkpoint_path=request.checkpoint_path,
        focal_policy_id=request.focal_policy_id,
        update_count=request.update_count,
        policy_version=request.policy_version,
        run_id256=request.run_id256,
        config_hash256=request.config_hash256,
        spec_hash256=request.spec_hash256,
        opponent_specs=request.opponents,
        eval_device_override=effective_eval_device,
        parallel_workers_override=int(request.parallel_workers),
        parallel_worker_devices_override=tuple(effective_worker_devices),
        artifact_dir_name=request.artifact_dir_name,
        artifact_scope=request.artifact_scope,
        paired_seeds_override=request.paired_seeds,
        process_pool_executor=process_pool_executor,
    )


def _run_periodic_dev_eval(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    device: torch.device,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str = "dev_eval",
    artifact_scope: str = "periodic_dev_eval",
    paired_seeds_override: Sequence[int] | None = None,
    persist_summary: bool = True,
    update_stall_monitor: bool = True,
    batched_inference_override: bool | None = None,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Periodic dev eval requires an attached learner model")

    checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=str(stack.config.training.algorithm).strip() if stack.config.training is not None else None,
    )
    summary_payload = _run_periodic_dev_eval_for_checkpoint(
        stack=stack,
        contract=contract,
        run_dir=artifacts.run_dir,
        checkpoint_path=checkpoint_path,
        focal_policy_id=_current_focal_policy_id(learner=learner),
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        artifact_dir_name=artifact_dir_name,
        artifact_scope=artifact_scope,
        paired_seeds_override=paired_seeds_override,
        batched_inference_override=batched_inference_override,
        process_pool_executor=process_pool_executor,
    )
    if update_stall_monitor and _dev_eval_is_authoritative(summary_payload):
        stall_monitor = _update_stall_monitor(
            stack=stack,
            training_paths=training_paths,
            update_count=int(summary_payload["update_count"]),
            summary_payload=summary_payload,
        )
        if stall_monitor is not None:
            summary_payload["stall_monitor"] = stall_monitor
            _write_json(
                artifacts.run_dir / "eval" / artifact_dir_name / f"update_{int(summary_payload['update_count'])}" / "summary.json",
                summary_payload,
            )
            if bool(stall_monitor.get("stall_risk", False)):
                print(
                    "Stall monitor warning: "
                    f"update={int(summary_payload['update_count'])} worst_anchor={stall_monitor['worst_anchor']} "
                    f"stall_rate={float(stall_monitor['worst_stall_rate']):.3f} "
                    f"no_progress_rate={float(stall_monitor['worst_no_progress_timeout_rate']):.3f} "
                    f"truncation_rate={float(stall_monitor['worst_truncation_rate']):.3f} "
                    f"consecutive={int(stall_monitor['consecutive_trigger_count'])}"
                )
    if persist_summary and _dev_eval_is_authoritative(summary_payload):
        _persist_periodic_dev_eval_summary(training_paths=training_paths, payload=summary_payload)
    elif persist_summary:
        _persist_periodic_dev_eval_fast_screen(training_paths=training_paths, payload=summary_payload)
    return summary_payload


def _process_completed_periodic_dev_eval(
    *,
    pending_eval: PendingPeriodicDevEval,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: RunArtifacts,
    training_paths: TrainingPaths,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    device: torch.device,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    last_rollback_update: int | None,
    tensorboard_logger: TensorBoardLogger | None,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        summary_payload = pending_eval.future.result()
        if _dev_eval_is_authoritative(summary_payload):
            stall_monitor = _update_stall_monitor(
                stack=stack,
                training_paths=training_paths,
                update_count=int(summary_payload["update_count"]),
                summary_payload=summary_payload,
            )
            if stall_monitor is not None:
                summary_payload["stall_monitor"] = stall_monitor
                _write_json(
                    artifacts.run_dir / "eval" / "dev_eval" / f"update_{int(summary_payload['update_count'])}" / "summary.json",
                    summary_payload,
                )
                if bool(stall_monitor.get("stall_risk", False)):
                    print(
                        "Stall monitor warning: "
                        f"update={int(summary_payload['update_count'])} worst_anchor={stall_monitor['worst_anchor']} "
                        f"stall_rate={float(stall_monitor['worst_stall_rate']):.3f} "
                        f"no_progress_rate={float(stall_monitor['worst_no_progress_timeout_rate']):.3f} "
                        f"truncation_rate={float(stall_monitor['worst_truncation_rate']):.3f} "
                        f"consecutive={int(stall_monitor['consecutive_trigger_count'])}"
                    )
        effective_summary = summary_payload
        tracker_before_dev_eval = _load_checkpoint_tracker(training_paths)
        existing_best_record = tracker_before_dev_eval.get("best")
        if not isinstance(existing_best_record, Mapping):
            existing_best_record = None
        confirmatory_request = _confirmatory_dev_eval_request(
            stack=stack,
            existing_best_record=cast(Mapping[str, Any] | None, existing_best_record),
            dev_eval_summary=effective_summary,
        )
        if confirmatory_request is not None:
            seed_file, _validated_sources, base_paired_seeds, seed_file_sha256 = _periodic_dev_eval_schedule(stack)
            confirmatory_pairs = _expand_periodic_dev_eval_paired_seeds(
                base_paired_seeds,
                requested_pairs=int(confirmatory_request["target_pairs"]),
                seed_file_sha256=seed_file_sha256,
                update_count=int(summary_payload["update_count"]),
                policy_version=int(summary_payload["policy_version"]),
                scope="periodic_dev_eval_confirmatory",
            )
            effective_summary = _run_periodic_dev_eval_for_checkpoint(
                stack=stack,
                contract=contract,
                run_dir=artifacts.run_dir,
                checkpoint_path=pending_eval.request.checkpoint_path,
                focal_policy_id=str(summary_payload["policy_id"]),
                update_count=int(summary_payload["update_count"]),
                policy_version=int(summary_payload["policy_version"]),
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                opponent_specs=pending_eval.request.opponents,
                eval_device_override=pending_eval.request.eval_device_override,
                artifact_dir_name="dev_eval_confirmatory",
                artifact_scope="periodic_dev_eval_confirmatory",
                paired_seeds_override=confirmatory_pairs,
                batched_inference_override=False,
                process_pool_executor=process_pool_executor,
            )
            print(
                "Confirmatory dev eval: "
                f"update={int(summary_payload['update_count'])} paired_seeds={len(confirmatory_pairs)} "
                f"aggregate={effective_summary['aggregate_score']:.4f} "
                f"reasons={','.join(cast(list[str], confirmatory_request['reasons']))} "
                f"seed_file={seed_file.name}"
            )
            _persist_periodic_dev_eval_summary(training_paths=training_paths, payload=effective_summary)
        elif _dev_eval_is_authoritative(effective_summary):
            _persist_periodic_dev_eval_summary(training_paths=training_paths, payload=effective_summary)
        else:
            _persist_periodic_dev_eval_fast_screen(training_paths=training_paths, payload=effective_summary)

        tracker_payload = _publish_best_checkpoint_from_dev_eval(
            stack=stack,
            training_paths=training_paths,
            artifacts=artifacts,
            checkpoint_path=pending_eval.request.checkpoint_path,
            update_count=int(effective_summary["update_count"]),
            policy_version=int(effective_summary["policy_version"]),
            dev_eval_summary=effective_summary,
        )
        _maybe_log_structured_mainmove_guard(
            training_paths=training_paths,
            learner=learner,
            latest_metrics=pending_eval.latest_metrics,
            dev_eval_summary=effective_summary,
        )
        guard_event = None
        if int(effective_summary["update_count"]) == int(learner.update_count):
            guard_event = _maybe_rollback_to_best_checkpoint(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                runtime=runtime,
                learner=learner,
                model=cast(PolicyValueModel, learner.model),
                device=device,
                spec_hash256=spec_hash256,
                algorithm=str(stack.config.training.algorithm).strip() if stack.config.training is not None else None,
                latest_metrics=pending_eval.latest_metrics,
                dev_eval_summary=effective_summary,
                last_rollback_update=last_rollback_update,
            )
        if tensorboard_logger is not None:
            tensorboard_logger.log_periodic_dev_eval(effective_summary, step=int(effective_summary["update_count"]))
            tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(effective_summary["update_count"]))
        audit_request = _maybe_request_b2_disagreement_audit(
            stack=stack,
            training_paths=training_paths,
            artifacts=artifacts,
            dev_eval_summary=effective_summary,
        )
        if audit_request is not None:
            print(
                "B2 disagreement audit requested: "
                f"update={int(audit_request['update_count'])} "
                f"reasons={','.join(cast(list[str], audit_request['trigger_reasons']))} "
                f"episodes={audit_request['episodes_path']}"
            )
        return effective_summary, guard_event
    finally:
        _unpin_snapshot_ids(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            snapshot_ids=pending_eval.pinned_snapshot_ids,
        )


def _run_async_promotion_gate_worker(request: AsyncPromotionGateRequest) -> dict[str, Any]:
    contract = load_verified_simulator_contract(request.stack.root, expected_spec_hash=request.spec_hash256)
    evaluation = _validate_periodic_dev_eval_contract(request.stack)
    observation_dim, action_dim = _spec_dimensions(contract)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    focal_model = _load_snapshot_eval_model(
        run_dir=request.run_dir,
        snapshot_path=request.candidate_snapshot_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=request.stack,
        eval_device=(evaluation.eval_device if request.eval_device_override is None else request.eval_device_override),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    )
    opponents = _materialize_periodic_dev_eval_opponents(
        stack=request.stack,
        contract=contract,
        run_dir=request.run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=request.anchor_specs,
        eval_device_override=request.eval_device_override,
    )
    anchor_models = {
        policy_id: opponent_model
        for policy_id, _display_name, opponent_model, _heuristic_policy in opponents
        if opponent_model is not None
    }
    heuristic_policies = {
        policy_id: heuristic_policy
        for policy_id, _display_name, _opponent_model, heuristic_policy in opponents
        if heuristic_policy is not None
    }
    runner = _PromotionGateRunner(
        stack=request.stack,
        focal_policy_id=request.candidate_policy_id,
        focal_model=focal_model,
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=request.run_dir / "eval" / "promotion_gate" / f"update_{request.update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        eval_device=(evaluation.eval_device if request.eval_device_override is None else request.eval_device_override),
    )
    try:
        result = run_promotion_gate(
            stack=request.stack,
            run_dir=request.run_dir / "eval" / "promotion_gate" / f"update_{request.update_count}",
            focal_policy_id=request.candidate_policy_id,
            anchor_policy_ids=request.anchor_policy_ids,
            runner=runner,
            run_id256=request.run_id256,
            config_hash256=request.config_hash256,
            spec_hash256=request.spec_hash256,
            bootstrap_seed=_promotion_gate_bootstrap_seed(
                update_count=request.update_count,
                policy_version=request.policy_version,
            ),
        )
    finally:
        close_runner = getattr(runner, "close", None)
        if callable(close_runner):
            close_runner()
    return {
        "candidate_policy_id": request.candidate_policy_id,
        "update_count": int(request.update_count),
        "policy_version": int(request.policy_version),
        "passed": bool(result.passed),
        "ordered_opponents": list(result.ordered_opponents),
        "reasons": [dict(reason) for reason in result.reasons],
        "result": result.to_dict(),
    }


def _run_parallel_promotion_gate_anchor_worker(
    *,
    stack: StackConfig,
    run_dir: Path,
    candidate_policy_id: str,
    candidate_snapshot_path: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    seed_block_jobs: Sequence[PromotionGateSeedBlockJob],
    eval_device_override: str,
) -> list[dict[str, Any]]:
    if not seed_block_jobs:
        return []
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash256)
    evaluation = _validate_periodic_dev_eval_contract(stack)
    observation_dim, action_dim = _spec_dimensions(contract)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    focal_model = _load_snapshot_eval_model(
        run_dir=run_dir,
        snapshot_path=candidate_snapshot_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        eval_device=eval_device_override,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    )
    ordered_specs = list({(job.anchor_index, job.anchor_spec): job.anchor_spec for job in seed_block_jobs}.values())
    opponents = _materialize_periodic_dev_eval_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=ordered_specs,
        eval_device_override=eval_device_override,
    )
    anchor_models = {
        policy_id: opponent_model
        for policy_id, _display_name, opponent_model, _heuristic_policy in opponents
        if opponent_model is not None
    }
    heuristic_policies = {
        policy_id: heuristic_policy
        for policy_id, _display_name, _opponent_model, heuristic_policy in opponents
        if heuristic_policy is not None
    }
    seed_file = resolve_promotion_gate_seed_file(stack)
    paired_seeds = parse_seed_file(seed_file)
    league = stack.config.league
    if league is None:
        raise RuntimeError("Parallel promotion gate requires stack.config.league")
    if len(paired_seeds) != int(league.promotion_gate_paired_seeds):
        raise RuntimeError(
            f"Promotion gate expected {int(league.promotion_gate_paired_seeds)} paired seeds in {seed_file}, "
            f"found {len(paired_seeds)}"
        )
    runner = _PromotionGateRunner(
        stack=stack,
        focal_policy_id=candidate_policy_id,
        focal_model=focal_model,
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        eval_device=eval_device_override,
    )
    worker_payloads: list[dict[str, Any]] = []
    try:
        for job in seed_block_jobs:
            scheduled_games = _periodic_dev_eval_schedule_for_seed_items(
                focal_policy_id=candidate_policy_id,
                opponent_policy_id=job.anchor_spec.policy_id,
                paired_seed_items=job.paired_seed_items,
            )
            records = tuple(
                record_completed_game(
                    scheduled_game=scheduled_game,
                    result=runner.run_game(scheduled_game),
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                )
                for scheduled_game in scheduled_games
            )
            worker_payloads.append(
                {
                    "anchor_index": int(job.anchor_index),
                    "block_index": int(job.block_index),
                    "anchor_policy_id": job.anchor_spec.policy_id,
                    "anchor_display_name": job.anchor_spec.display_name,
                    "paired_seed_items": tuple(job.paired_seed_items),
                    "records": records,
                }
            )
    finally:
        close_runner = getattr(runner, "close", None)
        if callable(close_runner):
            close_runner()
    return worker_payloads


def _run_parallel_snapshot_promotion_gate(
    *,
    stack: StackConfig,
    artifacts: RunArtifacts,
    training_paths: TrainingPaths,
    candidate_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    anchor_policy_ids: Mapping[str, str],
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    candidate_snapshot_path: str,
) -> PromotionGateResult:
    league = stack.config.league
    if league is None:
        raise RuntimeError("Parallel promotion gate requires stack.config.league")
    evaluation = _validate_periodic_dev_eval_contract(stack)
    ordered_anchors = resolve_promotion_gate_anchors(stack, anchor_policy_ids)
    configured_parallel_workers = max(1, int(getattr(league.promotion_gate, "parallel_workers", 1)))
    seed_file = resolve_promotion_gate_seed_file(stack)
    paired_seeds = parse_seed_file(seed_file)
    if len(paired_seeds) != int(league.promotion_gate_paired_seeds):
        raise RuntimeError(
            f"Promotion gate expected {int(league.promotion_gate_paired_seeds)} paired seeds in {seed_file}, "
            f"found {len(paired_seeds)}"
        )
    seed_block_jobs = _build_promotion_gate_seed_block_jobs(
        anchor_specs=anchor_specs,
        paired_seeds=paired_seeds,
        configured_parallel_workers=configured_parallel_workers,
    )
    effective_parallel_workers = min(configured_parallel_workers, max(1, len(seed_block_jobs)))
    worker_devices = _resolved_promotion_gate_worker_devices(
        stack=stack,
        parallel_workers=max(1, effective_parallel_workers),
        explicit_worker_devices=tuple(getattr(league.promotion_gate, "parallel_worker_devices", ())),
        eval_device=str(evaluation.eval_device),
    )
    job_shards = _shard_promotion_gate_seed_block_jobs(
        jobs=seed_block_jobs,
        shard_count=max(1, effective_parallel_workers),
    )
    worker_payloads: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(job_shards), mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _run_parallel_promotion_gate_anchor_worker,
                stack=stack,
                run_dir=artifacts.run_dir,
                candidate_policy_id=candidate_policy_id,
                candidate_snapshot_path=candidate_snapshot_path,
                update_count=update_count,
                policy_version=policy_version,
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                seed_block_jobs=tuple(shard),
                eval_device_override=worker_devices[shard_index],
            )
            for shard_index, shard in enumerate(job_shards)
        ]
        for future in futures:
            worker_payloads.extend(future.result())
    worker_payloads.sort(key=lambda payload: (int(payload["anchor_index"]), int(payload["block_index"])))
    records_by_anchor_index: dict[int, list[EvalGameRecord]] = {index: [] for index in range(len(anchor_specs))}
    for payload in worker_payloads:
        records_by_anchor_index[int(payload["anchor_index"])].extend(
            cast(Sequence[EvalGameRecord], payload["records"])
        )

    episodes_dir = artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}" / "promotion_gate_episodes"
    anchor_results: list[PromotionGateAnchorResult] = []
    all_pair_scores: list[float] = []
    total_truncated_games = 0
    total_games = 0
    for anchor_index, anchor in enumerate(ordered_anchors):
        records = sorted(
            records_by_anchor_index.get(anchor_index, []),
            key=lambda record: (int(record.pair_index), int(record.swap_index), int(record.episode_index)),
        )
        episodes_path = episodes_dir / f"{anchor_index:02d}_{_slug_policy_id(anchor.name)}.jsonl"
        write_episodes_jsonl(episodes_path, records)
        pair_scores = [float(score) for score in paired_seed_scores(records, scheme="S0")]
        truncated_games = sum(1 for record in records if record.truncated)
        anchor_result = PromotionGateAnchorResult(
            anchor_name=anchor.name,
            opponent_policy_id=anchor.policy_id,
            episodes_path=_relative_path_text(episodes_path, root=artifacts.run_dir),
            matchup_summary=summarize_game_records(records),
            truncation=PromotionGateRate(
                numerator=int(truncated_games),
                denominator=int(len(records)),
                rate=(float(truncated_games) / float(len(records))) if records else 0.0,
            ),
            posterior=PromotionGatePosterior.from_scores(
                pair_scores,
                sample_count=1000,
                seed=_promotion_gate_bootstrap_seed(update_count=update_count, policy_version=policy_version),
            ),
        )
        anchor_results.append(anchor_result)
        all_pair_scores.extend(pair_scores)
        total_truncated_games += int(truncated_games)
        total_games += int(len(records))

    result = build_promotion_gate_result(
        stack=stack,
        run_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        focal_policy_id=candidate_policy_id,
        anchors=ordered_anchors,
        anchor_results=tuple(anchor_results),
        all_pair_scores=tuple(all_pair_scores),
        total_truncated_games=total_truncated_games,
        total_games=total_games,
        paired_seed_count=len(paired_seeds),
        sample_count=1000,
        bootstrap_seed=_promotion_gate_bootstrap_seed(
            update_count=update_count,
            policy_version=policy_version,
        ),
    )
    result.write_json(artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}" / league.promotion_gate.record_file)
    return result


def _process_completed_promotion_gate(
    *,
    pending_gate: PendingPromotionGate,
    stack: StackConfig,
    artifacts: RunArtifacts,
    training_paths: TrainingPaths,
) -> bool:
    try:
        payload = pending_gate.future.result()
        if bool(payload["passed"]):
            registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
            registry = SnapshotRegistry.load(registry_path)
            registry.add_champion(str(payload["candidate_policy_id"]))
            _save_snapshot_registry_with_retention(
                stack=stack,
                training_paths=training_paths,
                run_dir=artifacts.run_dir,
                registry=registry,
            )
            print(
                "Promotion gate passed: "
                f"update={int(payload['update_count'])} candidate={str(payload['candidate_policy_id'])} "
                f"anchors={','.join(cast(list[str], payload['ordered_opponents']))}"
            )
            return True

        reason_codes = ",".join(
            str(reason.get("code", "unknown")) for reason in cast(list[dict[str, Any]], payload["reasons"])
        ) or "unknown"
        registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
        registry = SnapshotRegistry.load(registry_path)
        candidate_policy_id = str(payload["candidate_policy_id"])
        if registry.has_snapshot(candidate_policy_id):
            registry.reject_snapshot(candidate_policy_id)
            _save_snapshot_registry_with_retention(
                stack=stack,
                training_paths=training_paths,
                run_dir=artifacts.run_dir,
                registry=registry,
            )
        print(
            "Promotion gate failed: "
            f"update={int(payload['update_count'])} candidate={candidate_policy_id} "
            f"reasons={reason_codes}"
        )
        return False
    finally:
        _unpin_snapshot_ids(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            snapshot_ids=pending_gate.pinned_snapshot_ids,
        )


def _drop_stale_pending_promotion_gate(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    pending_gate: PendingPromotionGate | None,
    rollback_best_update_count: int,
) -> PendingPromotionGate | None:
    if pending_gate is None:
        return None
    if int(pending_gate.request.update_count) <= int(rollback_best_update_count):
        return pending_gate
    print(
        "Promotion gate result discarded after rollback: "
        f"candidate={pending_gate.request.candidate_policy_id} "
        f"candidate_update={int(pending_gate.request.update_count)} "
        f"rollback_best_update={int(rollback_best_update_count)}"
    )
    _unpin_snapshot_ids(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        snapshot_ids=pending_gate.pinned_snapshot_ids,
    )
    return None


def _run_snapshot_promotion_gate(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    candidate_policy_id: str,
    update_count: int,
    league_reference_update: int | None,
    league_eval_warmup_gate_open: bool,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> bool | None:
    league = stack.config.league
    if league is None or not league.enabled or not league.promotion_gate_enabled:
        return None
    reference_update = int(update_count if league_reference_update is None else league_reference_update)
    if reference_update < int(league.warmup.first_updates):
        print(
            "Promotion gate skipped during league warmup: "
            f"update={update_count} effective_update={reference_update} threshold={int(league.warmup.first_updates)} "
            f"candidate={candidate_policy_id}"
        )
        return None
    if bool(getattr(league.warmup, "eval_gate_enabled", False)) and not bool(league_eval_warmup_gate_open):
        print(
            "Promotion gate skipped during league eval warmup gate: "
            f"update={update_count} effective_update={reference_update} candidate={candidate_policy_id}"
        )
        return None
    if learner.model is None:
        raise RuntimeError("Promotion gate requires an attached learner model")

    evaluation = _validate_periodic_dev_eval_contract(stack)
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        print(
            "Promotion gate skipped: "
            f"update={update_count} candidate={candidate_policy_id} "
            f"missing_anchors={','.join(missing_required)}"
        )
        return None

    observation_dim, action_dim = _spec_dimensions(contract)
    snapshot_index = _snapshot_meta_by_policy_id(registry)
    configured_parallel_workers = max(1, int(getattr(league.promotion_gate, "parallel_workers", 1)))
    if configured_parallel_workers > 1:
        candidate_snapshot = snapshot_index.get(candidate_policy_id)
        if candidate_snapshot is None:
            raise RuntimeError(f"Promotion gate could not resolve candidate snapshot {candidate_policy_id!r}")
        anchor_policy_ids_for_parallel, anchor_specs, _pinned_anchor_snapshot_ids = _resolve_promotion_gate_anchor_specs(
            stack=stack,
            training_paths=training_paths,
        )
        result = _run_parallel_snapshot_promotion_gate(
            stack=stack,
            artifacts=artifacts,
            training_paths=training_paths,
            candidate_policy_id=candidate_policy_id,
            update_count=update_count,
            policy_version=policy_version,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            anchor_policy_ids=anchor_policy_ids_for_parallel,
            anchor_specs=anchor_specs,
            candidate_snapshot_path=candidate_snapshot.path,
        )
        if result.passed:
            registry.add_champion(candidate_policy_id)
            _save_snapshot_registry_with_retention(
                stack=stack,
                training_paths=training_paths,
                run_dir=artifacts.run_dir,
                registry=registry,
            )
            print(
                "Promotion gate passed: "
                f"update={update_count} candidate={candidate_policy_id} "
                f"anchors={','.join(result.ordered_opponents)}"
            )
            return True

        reason_codes = ",".join(str(reason.get("code", "unknown")) for reason in result.reasons) or "unknown"
        registry.reject_snapshot(candidate_policy_id)
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            registry=registry,
        )
        print(f"Promotion gate failed: update={update_count} candidate={candidate_policy_id} reasons={reason_codes}")
        return False
    anchor_models = {
        policy_id: _load_snapshot_eval_model(
            run_dir=artifacts.run_dir,
            snapshot_path=snapshot_index[policy_id].path,
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            eval_device=evaluation.eval_device,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        )
        for policy_id in set(anchor_policy_ids.values())
        if policy_id != _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
        and heuristic_public_profile_name_for_policy_id(policy_id) is None
    }
    heuristic_policies: dict[str, HeuristicPublicPolicy] = {}
    heuristic_policy_ids = {
        policy_id
        for policy_id in set(anchor_policy_ids.values())
        if heuristic_public_profile_name_for_policy_id(policy_id) is not None
    }
    if heuristic_policy_ids:
        try:
            heuristic_policies = {
                policy_id: _build_heuristic_public_policy(
                    contract.spec_bundle,
                    scoring_profile=cast(str, heuristic_public_profile_name_for_policy_id(policy_id)),
                )
                for policy_id in heuristic_policy_ids
            }
        except Exception as exc:
            assert league is not None
            missing_required = [
                policy_id for policy_id in heuristic_policy_ids if policy_id in league.promotion_anchor_set_v1.required
            ]
            if missing_required:
                missing_text = ", ".join(missing_required)
                raise RuntimeError(
                    f"Promotion gate requires a heuristic-compatible simulator contract for {missing_text}"
                ) from exc
            anchor_policy_ids = {
                anchor_name: policy_id
                for anchor_name, policy_id in anchor_policy_ids.items()
                if heuristic_public_profile_name_for_policy_id(policy_id) is None
            }
            print(
                "Promotion gate note: skipping optional heuristic-public anchors because the active simulator contract "
                f"does not expose the required public action/observation metadata ({exc})."
            )
    runner = _PromotionGateRunner(
        stack=stack,
        focal_policy_id=candidate_policy_id,
        focal_model=_clone_eval_model(
            learner_model=cast(PolicyValueModel, learner.model),
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            eval_device=evaluation.eval_device,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        ),
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        eval_device=evaluation.eval_device,
    )
    try:
        result = run_promotion_gate(
            stack=stack,
            run_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
            focal_policy_id=candidate_policy_id,
            anchor_policy_ids=anchor_policy_ids,
            runner=runner,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            bootstrap_seed=_promotion_gate_bootstrap_seed(
                update_count=update_count,
                policy_version=policy_version,
            ),
        )
    finally:
        close_runner = getattr(runner, "close", None)
        if callable(close_runner):
            close_runner()
    if result.passed:
        registry.add_champion(candidate_policy_id)
        _save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            registry=registry,
        )
        print(
            "Promotion gate passed: "
            f"update={update_count} candidate={candidate_policy_id} "
            f"anchors={','.join(result.ordered_opponents)}"
        )
        return True

    reason_codes = ",".join(str(reason.get("code", "unknown")) for reason in result.reasons) or "unknown"
    registry.reject_snapshot(candidate_policy_id)
    _save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=artifacts.run_dir,
        registry=registry,
    )
    print(f"Promotion gate failed: update={update_count} candidate={candidate_policy_id} reasons={reason_codes}")
    return False


def _run_minimal_training(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
    max_wall_clock_minutes: float | None,
    profile: str,
    device: torch.device,
    seed: int,
    checkpoint_interval_updates: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    runtime_mode: QueueRuntimeMode,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None = None,
    seed_snapshot_run_dir_auto_inferred: bool = False,
    profile_timers: bool = False,
    torch_profiler: bool = False,
    resume_checkpoint_path: Path | None = None,
    resume_allow_config_mismatch: bool = False,
    resume_reset_optimizer: bool = False,
    tensorboard_logger: TensorBoardLogger | None = None,
    resolved_topology: ResolvedTrainingTopology | None = None,
    distributed_context: DistributedContext | None = None,
) -> dict[str, float]:
    _configure_torch_threads(stack)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    observation_dim, action_dim = _spec_dimensions(contract)
    training_config = stack.config.training
    model_config = stack.config.model
    environment_config = stack.config.environment
    rewards_config = stack.config.rewards
    experiment_role = _experiment_role(stack)
    if training_config is None or model_config is None or environment_config is None or rewards_config is None:
        raise RuntimeError("The locked stack is missing training, model, environment, or rewards config")
    main_residual_policy_enabled = bool(
        getattr(getattr(training_config, "main_residual_policy", None), "enabled", False)
    )

    training_paths = _training_paths(artifacts.run_dir)
    ddp_context = DistributedContext(enabled=False) if distributed_context is None else distributed_context
    rank0 = (not ddp_context.enabled) or ddp_context.is_rank0
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    algorithm = str(training_config.algorithm).strip()
    _validate_algorithm_model_contract(
        algorithm=algorithm,
        recurrent_core=model_config.recurrent_core,
        encoder_kind=model_config.encoder_kind,
    )
    model = _maybe_build_main_residual_model(
        stack=stack,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
        device=device,
    )
    if model is None:
        model = build_policy_value_model(
            observation_dim=observation_dim,
            config=model_config,
            action_dim=action_dim,
            observation_spec=contract.spec_bundle.get("observation"),
            spec_bundle=contract.spec_bundle,
        ).to(device)
    compiled_model = _maybe_compile_learner_model(
        model=model,
        training_config=training_config,
        device=device,
    )
    learner = _build_training_learner(
        algorithm=algorithm,
        model=model,
        compiled_model=compiled_model,
        training_config=training_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=checkpoint_interval_updates,
        gradient_sync=(None if not ddp_context.enabled else lambda: average_gradients(model, context=ddp_context)),
        artifact_writes_enabled=rank0,
    )
    resume_state = None
    resume_dev_eval_summary: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        resume_state = _restore_learner_from_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
            restore_optimizer_state=not bool(resume_reset_optimizer),
            allow_config_hash_mismatch=resume_allow_config_mismatch,
        )
        print(
            "Resumed learner state: "
            f"checkpoint={resume_state.checkpoint_path} "
            f"update={resume_state.update_count} "
            f"policy_version={resume_state.policy_version}"
        )
        if rank0:
            seeded_best_record = _seed_checkpoint_tracker_from_resume_best(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                resume_checkpoint_path=resume_state.checkpoint_path,
            )
            if seeded_best_record is not None:
                print(
                    "Seeded checkpoint best alias from resumed dev-eval best: "
                    f"update={int(seeded_best_record['update_count'])} "
                    f"metric={float(seeded_best_record['metric_value']):.4f}"
                )
        resume_dev_eval_summary = _load_resume_checkpoint_dev_eval_summary(
            stack=stack,
            resume_checkpoint_path=resume_state.checkpoint_path,
            update_count=int(resume_state.update_count),
            allow_config_hash_mismatch=resume_allow_config_mismatch,
        )
        if rank0 and resume_dev_eval_summary is not None:
            print(
                "Seeded resume dev-eval summary: "
                f"update={int(resume_state.update_count)} "
                f"aggregate={float(_dev_eval_aggregate_score(resume_dev_eval_summary) or 0.0):.4f}"
            )

    config_hash256 = compute_config_hash256(stack)
    if rank0:
        _ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            learner=learner,
            device=device,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            baseline_run_dir=b1_baseline_run_dir,
            permit_current_run_alias=_is_noleague_baseline_role(experiment_role),
            update=int(learner.update_count),
        )
    distributed_barrier(ddp_context)
    _attach_reference_policy_model_if_configured(
        learner=learner,
        training_config=training_config,
        training_paths=training_paths,
        model_config=model_config,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any], contract.spec_bundle),
        device=device,
    )
    imported_resume_league_policy_ids: tuple[str, ...] = ()
    if rank0 and resume_state is not None:
        imported_resume_league_policy_ids = tuple(
            _import_resume_league_snapshot_pool(
                stack=stack,
                training_paths=training_paths,
                run_dir=artifacts.run_dir,
                resume_checkpoint_path=resume_state.checkpoint_path,
                max_update=int(resume_state.update_count),
                expected_model_state_dict=learner.model.state_dict(),
            )
        )
    seed_snapshot_max_update = _seed_snapshot_import_max_update(
        resume_state=resume_state,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        seed_snapshot_run_dir_auto_inferred=seed_snapshot_run_dir_auto_inferred,
    )
    if rank0 and seed_snapshot_run_dir is not None:
        _import_seed_snapshot_pool(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            seed_snapshot_run_dir=seed_snapshot_run_dir,
            max_update=seed_snapshot_max_update,
            exclude_source_policy_ids=imported_resume_league_policy_ids,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
    if seed_snapshot_run_dir is not None or resume_state is not None:
        distributed_barrier(ddp_context)
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=shard_env_count(global_num_envs=num_envs, world_size=ddp_context.world_size, rank=ddp_context.rank)
        if ddp_context.enabled
        else num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=rank_seed(seed, rank=ddp_context.rank) if ddp_context.enabled else seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
        resolved_actor_count=None if resolved_topology is None else max(1, int(resolved_topology.actor_count) // max(1, int(ddp_context.world_size))),
        resolved_envs_per_actor=None if resolved_topology is None else int(resolved_topology.envs_per_actor),
        resolved_batch_unrolls_per_update=(
            None
            if resolved_topology is None
            else max(1, int(resolved_topology.batch_unrolls_per_update) // max(1, int(ddp_context.world_size)))
        ),
        resolved_queue_capacity_unrolls=(
            None
            if resolved_topology is None
            else max(1, int(resolved_topology.queue_capacity_unrolls) // max(1, int(ddp_context.world_size)))
        ),
    )
    runtime = QueueRuntime(
        stack=stack,
        config=runtime_config,
        model=model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any], contract.spec_bundle),
        run_dir=artifacts.run_dir,
        performance_log_path=training_paths.performance_log_path,
        learner_device=device,
        initial_learner_update=int(learner.update_count),
    )
    if int(learner.update_count) > 0:
        runtime.maybe_publish_snapshot(
            learner_model=model,
            learner_update_count=int(learner.update_count),
            force=True,
        )
    actor_torch_threads = _central_runtime_actor_torch_threads(stack, runtime)
    learner_torch_threads = None if stack.config.system is None else int(stack.config.system.learner_torch_threads)
    latest_metrics: dict[str, float] = {}
    last_checkpoint_guard_rollback_update: int | None = None
    last_dev_eval_summary: Mapping[str, Any] | None = resume_dev_eval_summary
    last_dev_eval_update_count: int | None = (
        int(resume_state.update_count) if resume_state is not None and resume_dev_eval_summary is not None else None
    )
    league_eval_warmup_gate_status = _sync_runtime_league_eval_warmup_gate(
        runtime=runtime,
        stack=stack,
        dev_eval_summary=last_dev_eval_summary,
    )
    league_eval_warmup_gate_open = bool(league_eval_warmup_gate_status["open"])
    collect_batch_prefetch_enabled = bool(getattr(training_config, "collect_batch_prefetch_enabled", False))
    start_time = time.time()
    max_wall_clock_seconds = _wall_clock_budget_seconds(max_wall_clock_minutes)
    profiler, profiler_context, profiler_trace_dir = _build_training_profiler(
        enabled=bool(torch_profiler),
        run_dir=artifacts.run_dir,
        device=device,
    )
    prefetch_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="collect-batch-prefetch")
        if collect_batch_prefetch_enabled
        else None
    )
    async_periodic_dev_eval_enabled = bool(
        rank0
        and
        stack.config.evaluation is not None
        and getattr(stack.config.evaluation, "async_periodic_dev_eval_enabled", False)
    )
    async_periodic_dev_eval_executor: ThreadPoolExecutor | None = None
    if async_periodic_dev_eval_enabled:
        async_periodic_dev_eval_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="async-periodic-dev-eval",
        )
    async_promotion_gate_enabled = bool(
        rank0
        and
        stack.config.league is not None
        and stack.config.league.promotion_gate_enabled
        and bool(getattr(stack.config.league.promotion_gate, "async_enabled", False))
    )
    async_promotion_gate_executor: ProcessPoolExecutor | None = None
    if async_promotion_gate_enabled:
        async_promotion_gate_executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    pending_promotion_gate: PendingPromotionGate | None = None
    pending_periodic_dev_eval: PendingPeriodicDevEval | None = None
    periodic_dev_eval_process_pool: ProcessPoolExecutor | None = None
    prefetched_runtime_batch: Any | None = None
    early_cutoff_payload: Mapping[str, Any] | None = None
    with profiler_context:
        if int(learner.update_count) == 0:
            latest_metrics = _run_structured_warmstart(
                learner=learner,
                runtime=runtime,
                algorithm=algorithm,
                training_config=training_config,
                rewards_config=rewards_config,
                training_paths=training_paths,
                tensorboard_logger=tensorboard_logger,
                start_time=start_time,
                profile_timers=bool(profile_timers),
                actor_torch_threads=actor_torch_threads,
                learner_torch_threads=learner_torch_threads,
            )
        if int(learner.update_count) >= max_updates:
            raise RuntimeError(
                f"Resume checkpoint is already at update {learner.update_count}, which is >= --max-updates {max_updates}"
            )
        try:
            for _update_index in range(int(learner.update_count), max_updates):
                stop_requested = False
                if pending_promotion_gate is not None and pending_promotion_gate.future.done():
                    promotion_passed = _process_completed_promotion_gate(
                        pending_gate=pending_promotion_gate,
                        stack=stack,
                        artifacts=artifacts,
                        training_paths=training_paths,
                    )
                    pending_promotion_gate = None
                    if promotion_passed:
                        runtime.refresh_opponent_pool()
                if pending_periodic_dev_eval is not None and pending_periodic_dev_eval.future.done():
                    completed_summary, guard_event = _process_completed_periodic_dev_eval(
                        pending_eval=pending_periodic_dev_eval,
                        stack=stack,
                        contract=contract,
                        artifacts=artifacts,
                        training_paths=training_paths,
                        runtime=runtime,
                        learner=learner,
                        device=device,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                        last_rollback_update=last_checkpoint_guard_rollback_update,
                        tensorboard_logger=tensorboard_logger,
                        process_pool_executor=periodic_dev_eval_process_pool,
                    )
                    pending_periodic_dev_eval = None
                    last_dev_eval_summary = completed_summary
                    last_dev_eval_update_count = int(completed_summary["update_count"])
                    league_eval_warmup_gate_status = _sync_runtime_league_eval_warmup_gate(
                        runtime=runtime,
                        stack=stack,
                        dev_eval_summary=completed_summary,
                    )
                    league_eval_warmup_gate_open = bool(league_eval_warmup_gate_status["open"])
                    anchor_keys = sorted(cast(dict[str, Any], completed_summary["anchor_scores"]).keys())
                    opponent_fragment = f" opponent={_slug_policy_id(anchor_keys[0])}" if anchor_keys else ""
                    print(
                        "Periodic dev eval complete: "
                        f"update={int(completed_summary['update_count'])}{opponent_fragment} "
                        f"aggregate={completed_summary['aggregate_score']:.4f} "
                        f"anchors={','.join(anchor_keys)}"
                    )
                    if bool(league_eval_warmup_gate_status.get("enabled", False)):
                        print(
                            "League eval warmup gate: "
                            f"open={league_eval_warmup_gate_open} "
                            f"reasons={','.join(cast(list[str], league_eval_warmup_gate_status.get('reasons', [])))}"
                        )
                    if guard_event is not None:
                        last_checkpoint_guard_rollback_update = int(learner.update_count)
                        pending_promotion_gate = _drop_stale_pending_promotion_gate(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            pending_gate=pending_promotion_gate,
                            rollback_best_update_count=int(guard_event["best_update_count"]),
                        )
                        prefetched_runtime_batch = None
                        print(
                            "Checkpoint guard rollback: "
                            f"update={guard_event['update_count']} "
                            f"best_update={guard_event['best_update_count']} "
                            f"current_score={float(guard_event['current_score']):.4f} "
                            f"best_score={float(guard_event['best_score']):.4f} "
                            f"reasons={','.join(cast(list[str], guard_event['reasons']))}"
                        )
                    early_cutoff_payload = _update_early_cutoff(
                        stack=stack,
                        training_paths=training_paths,
                        update_count=int(completed_summary["update_count"]),
                        summary_payload=completed_summary,
                    )
                    if early_cutoff_payload is not None and bool(early_cutoff_payload.get("should_stop", False)):
                        latest_metrics.update(
                            {
                                "early_cutoff_triggered": 1.0,
                                "early_cutoff_best_score": float(early_cutoff_payload["best_score"]),
                                "early_cutoff_current_score": float(early_cutoff_payload["current_score"]),
                                "early_cutoff_no_improvement_updates": float(
                                    early_cutoff_payload["no_improvement_updates"]
                                ),
                                "early_cutoff_consecutive_stall_evals": float(
                                    early_cutoff_payload["consecutive_stall_evals"]
                                ),
                            }
                        )
                        print(
                            "Early cutoff triggered: "
                            f"update={int(completed_summary['update_count'])} "
                            f"best_update={int(early_cutoff_payload['best_update_count'])} "
                            f"best_score={float(early_cutoff_payload['best_score']):.4f} "
                            f"current_score={float(early_cutoff_payload['current_score']):.4f} "
                            f"reasons={','.join(cast(list[str], early_cutoff_payload['reasons']))}"
                        )
                        stop_requested = True
                if stop_requested:
                    break
                if _wall_clock_budget_reached(start_time=start_time, max_wall_clock_seconds=max_wall_clock_seconds):
                    elapsed_seconds = time.time() - start_time
                    latest_metrics.update(
                        {
                            "wall_clock_budget_reached": 1.0,
                            "wall_clock_budget_seconds": float(max_wall_clock_seconds),
                            "wall_clock_budget_elapsed_seconds": float(elapsed_seconds),
                        }
                    )
                    print(
                        "Wall clock budget reached: "
                        f"elapsed={elapsed_seconds:.2f}s "
                        f"budget={float(max_wall_clock_seconds):.2f}s"
                    )
                    break
                guidance_schedule_metrics = _apply_guidance_schedule_for_next_update(
                    learner=learner,
                    model=model,
                    stack=stack,
                    update_count=int(learner.update_count) + 1,
                )
                learner.set_entropy_coef(
                    _entropy_coef_for_next_update(training_config, update_count=int(learner.update_count) + 1)
                )
                upcoming_update_count = int(learner.update_count) + 1
                batch_wait_started = time.perf_counter()
                if prefetched_runtime_batch is not None:
                    runtime_batch = prefetched_runtime_batch
                    prefetched_runtime_batch = None
                else:
                    with (
                        _profile_block(profile_timers, "collect_update_batch"),
                        _torch_num_threads_scope(actor_torch_threads),
                    ):
                        runtime_batch = _collect_training_batch(
                            runtime=runtime,
                            algorithm=algorithm,
                            training_config=training_config,
                            rewards_config=rewards_config,
                        )
                learner_idle_wait_for_batch_ms = (time.perf_counter() - batch_wait_started) * 1000.0
                prefetch_future: Future[Any] | None = None
                should_prefetch_next_batch = (
                    prefetch_executor is not None
                    and upcoming_update_count < max_updates
                    and upcoming_update_count % checkpoint_interval_updates != 0
                    and not _should_run_periodic_dev_eval(stack, update_count=upcoming_update_count)
                )
                if should_prefetch_next_batch:
                    prefetch_future = prefetch_executor.submit(
                        _collect_training_batch_prefetch,
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                        actor_torch_threads=actor_torch_threads,
                    )
                learner_update_started = time.perf_counter()
                with _profile_block(profile_timers, "learner_update"), _torch_num_threads_scope(learner_torch_threads):
                    latest_metrics = learner.update(runtime_batch.learner_batch)
                learner_update_ms = (time.perf_counter() - learner_update_started) * 1000.0
                learner_idle_wait_for_prefetch_ms = 0.0
                if prefetch_future is not None:
                    prefetch_wait_started = time.perf_counter()
                    with _profile_block(profile_timers, "collect_update_batch_prefetch_join"):
                        prefetched_runtime_batch = prefetch_future.result()
                    learner_idle_wait_for_prefetch_ms = (time.perf_counter() - prefetch_wait_started) * 1000.0
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(guidance_schedule_metrics)
                latest_metrics.update(
                    {
                        "learner_idle_wait_for_batch_ms": learner_idle_wait_for_batch_ms,
                        "learner_idle_wait_for_prefetch_ms": learner_idle_wait_for_prefetch_ms,
                        "learner_update_ms": learner_update_ms,
                        "distributed_rank": float(ddp_context.rank),
                        "distributed_world_size": float(ddp_context.world_size),
                        "async_periodic_dev_eval_overlap_active": 1.0
                        if pending_periodic_dev_eval is not None
                        else 0.0,
                        "async_promotion_gate_overlap_active": 1.0
                        if pending_promotion_gate is not None
                        else 0.0,
                    }
                )
                if ddp_context.enabled:
                    local_batch_env_steps = float(latest_metrics.get("batch_env_steps", 0.0))
                    global_batch_env_steps = all_reduce_float(local_batch_env_steps, context=ddp_context, op="sum")
                    global_total_samples = all_reduce_float(
                        float(getattr(learner, "total_samples_processed", 0)),
                        context=ddp_context,
                        op="sum",
                    )
                    elapsed_for_global = max(time.time() - start_time, 1e-6)
                    latest_metrics.update(
                        {
                            "distributed_global_batch_env_steps": global_batch_env_steps,
                            "distributed_global_total_samples_processed": global_total_samples,
                            "distributed_global_samples_per_sec": global_total_samples / elapsed_for_global,
                            "distributed_local_batch_env_steps": local_batch_env_steps,
                        }
                    )
                with _profile_block(profile_timers, "runtime_snapshot_publish"):
                    latest_metrics.update(
                        runtime.maybe_publish_snapshot(
                            learner_model=model,
                            learner_update_count=int(learner.update_count),
                        )
                    )
                latest_metrics["snapshot_publish_reload_ms"] = float(
                    latest_metrics.get("snapshot_publish_latency_ms", 0.0)
                    + latest_metrics.get("snapshot_apply_latency_ms", 0.0)
                )
                if rank0:
                    _write_scalars_record(
                        scalars_path=training_paths.scalars_path,
                        learner=learner,
                        metrics=latest_metrics,
                        start_time=start_time,
                    )
                if rank0 and tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
                if rank0 and learner.update_count % checkpoint_interval_updates == 0:
                    ckpt_path = training_paths.checkpoints_dir / f"checkpoint_{learner.update_count}.pt"
                    _write_checkpoint(
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    tracker_payload = _publish_checkpoint_aliases(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        latest_metrics=latest_metrics,
                    )
                    _maybe_log_structured_mainmove_guard(
                        training_paths=training_paths,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=last_dev_eval_summary,
                    )
                    if tensorboard_logger is not None:
                        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))

                    if main_residual_policy_enabled:
                        latest_metrics["main_residual_snapshot_registry_skipped"] = 1.0
                    else:
                        if learner.model is None:
                            raise RuntimeError("Cannot persist a snapshot registry entry without a learner model")
                        candidate_policy_id = _persist_snapshot_registry_entry(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            checkpoint_path=ckpt_path,
                            model_state_dict=learner.model.state_dict(),
                            config_hash256=config_hash256,
                            device=device,
                            update=int(learner.update_count),
                            policy_version=int(learner.get_policy_version()),
                            model=learner.model,
                        )
                        defer_noleague_baseline_alias_refresh = _should_defer_noleague_baseline_alias_refresh(
                            stack=stack,
                            experiment_role=experiment_role,
                            update_count=int(learner.update_count),
                        )
                        if _is_noleague_baseline_role(experiment_role) and not defer_noleague_baseline_alias_refresh:
                            _ensure_noleague_baseline_anchor(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                learner=learner,
                                device=device,
                                config_hash256=config_hash256,
                                permit_current_run_alias=True,
                                source_checkpoint_path=ckpt_path,
                                update=int(learner.update_count),
                            )
                        runtime.refresh_opponent_pool()
                    if main_residual_policy_enabled:
                        continue
                    if async_promotion_gate_enabled:
                        if async_promotion_gate_executor is None:
                            raise RuntimeError("async promotion gate is enabled but the worker pool was not created")
                        if pending_promotion_gate is not None:
                            promotion_passed = _process_completed_promotion_gate(
                                pending_gate=pending_promotion_gate,
                                stack=stack,
                                artifacts=artifacts,
                                training_paths=training_paths,
                            )
                            pending_promotion_gate = None
                            if promotion_passed:
                                runtime.refresh_opponent_pool()
                        league_reference_update = (
                            None
                            if "league_effective_update" not in latest_metrics
                            else int(latest_metrics["league_effective_update"])
                        )
                        league = stack.config.league
                        reference_update = int(
                            int(learner.update_count) if league_reference_update is None else league_reference_update
                        )
                        if (
                            league is not None
                            and league.enabled
                            and league.promotion_gate_enabled
                            and reference_update >= int(league.warmup.first_updates)
                            and (
                                not bool(getattr(league.warmup, "eval_gate_enabled", False))
                                or bool(league_eval_warmup_gate_open)
                            )
                        ):
                            registry = SnapshotRegistry.load(training_paths.snapshots_dir / REGISTRY_FILENAME)
                            anchor_policy_ids, anchor_specs, pinned_anchor_snapshot_ids = _resolve_promotion_gate_anchor_specs(
                                stack=stack,
                                training_paths=training_paths,
                            )
                            snapshot_index = _snapshot_meta_by_policy_id(registry)
                            candidate_snapshot = snapshot_index.get(candidate_policy_id)
                            if candidate_snapshot is None:
                                raise RuntimeError(
                                    f"Could not resolve persisted candidate snapshot for promotion gate: {candidate_policy_id}"
                                )
                            newly_pinned_snapshot_ids = _pin_snapshot_ids(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                snapshot_ids=(candidate_policy_id, *pinned_anchor_snapshot_ids),
                            )
                            request = AsyncPromotionGateRequest(
                                stack=stack,
                                run_dir=artifacts.run_dir,
                                candidate_policy_id=candidate_policy_id,
                                candidate_snapshot_path=candidate_snapshot.path,
                                update_count=int(learner.update_count),
                                policy_version=int(learner.get_policy_version()),
                                run_id256=run_id256,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                anchor_policy_ids=anchor_policy_ids,
                                anchor_specs=anchor_specs,
                                eval_device_override=_resolve_async_promotion_gate_device(
                                    stack=stack,
                                    learner_device=device,
                                ),
                            )
                            pending_promotion_gate = PendingPromotionGate(
                                future=async_promotion_gate_executor.submit(_run_async_promotion_gate_worker, request),
                                request=request,
                                pinned_snapshot_ids=newly_pinned_snapshot_ids,
                            )
                            print(
                                "Scheduled async promotion gate: "
                                f"update={int(learner.update_count)} candidate={candidate_policy_id} "
                                f"anchors={','.join(anchor_policy_ids.keys())}"
                            )
                        elif league is not None and league.enabled and league.promotion_gate_enabled:
                            if reference_update < int(league.warmup.first_updates):
                                print(
                                    "Promotion gate skipped during league warmup: "
                                    f"update={int(learner.update_count)} effective_update={reference_update} "
                                    f"threshold={int(league.warmup.first_updates)} candidate={candidate_policy_id}"
                                )
                            else:
                                print(
                                    "Promotion gate skipped during league eval warmup gate: "
                                    f"update={int(learner.update_count)} effective_update={reference_update} "
                                    f"candidate={candidate_policy_id}"
                                )
                    else:
                        promotion_passed = _run_snapshot_promotion_gate(
                            stack=stack,
                            contract=contract,
                            artifacts=artifacts,
                            training_paths=training_paths,
                            learner=learner,
                            candidate_policy_id=candidate_policy_id,
                            update_count=int(learner.update_count),
                            league_reference_update=(
                                None
                                if "league_effective_update" not in latest_metrics
                                else int(latest_metrics["league_effective_update"])
                            ),
                            league_eval_warmup_gate_open=league_eval_warmup_gate_open,
                            policy_version=int(learner.get_policy_version()),
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                        )
                        if promotion_passed:
                            runtime.refresh_opponent_pool()

                if rank0 and _should_run_periodic_dev_eval(stack, update_count=int(learner.update_count)):
                    defer_noleague_baseline_alias_refresh = _should_defer_noleague_baseline_alias_refresh(
                        stack=stack,
                        experiment_role=experiment_role,
                        update_count=int(learner.update_count),
                    )
                    checkpoint_path = _ensure_current_checkpoint(
                        training_paths=training_paths,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    if async_periodic_dev_eval_enabled:
                        if async_periodic_dev_eval_executor is None:
                            raise RuntimeError("async periodic dev eval is enabled but the worker pool was not created")
                        if pending_periodic_dev_eval is not None:
                            completed_summary, guard_event = _process_completed_periodic_dev_eval(
                                pending_eval=pending_periodic_dev_eval,
                                stack=stack,
                                contract=contract,
                                artifacts=artifacts,
                                training_paths=training_paths,
                                runtime=runtime,
                                learner=learner,
                                device=device,
                                run_id256=run_id256,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                last_rollback_update=last_checkpoint_guard_rollback_update,
                                tensorboard_logger=tensorboard_logger,
                                process_pool_executor=periodic_dev_eval_process_pool,
                            )
                            pending_periodic_dev_eval = None
                            last_dev_eval_summary = completed_summary
                            last_dev_eval_update_count = int(completed_summary["update_count"])
                            if guard_event is not None:
                                last_checkpoint_guard_rollback_update = int(learner.update_count)
                                pending_promotion_gate = _drop_stale_pending_promotion_gate(
                                    stack=stack,
                                    training_paths=training_paths,
                                    run_dir=artifacts.run_dir,
                                    pending_gate=pending_promotion_gate,
                                    rollback_best_update_count=int(guard_event["best_update_count"]),
                                )
                                prefetched_runtime_batch = None
                                print(
                                    "Checkpoint guard rollback: "
                                    f"update={guard_event['update_count']} "
                                    f"best_update={guard_event['best_update_count']} "
                                    f"current_score={float(guard_event['current_score']):.4f} "
                                    f"best_score={float(guard_event['best_score']):.4f} "
                                    f"reasons={','.join(cast(list[str], guard_event['reasons']))}"
                                )
                        opponent_specs, pinned_snapshot_ids = _resolve_periodic_dev_eval_opponent_specs(
                            stack=stack,
                            run_dir=artifacts.run_dir,
                        )
                        newly_pinned_snapshot_ids = _pin_snapshot_ids(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            snapshot_ids=pinned_snapshot_ids,
                        )
                        async_eval_device = _resolve_async_periodic_dev_eval_device(
                            stack=stack,
                            learner_device=device,
                        )
                        request = AsyncPeriodicDevEvalRequest(
                            stack=stack,
                            checkpoint_path=checkpoint_path,
                            focal_policy_id=_current_focal_policy_id(learner=learner),
                            update_count=int(learner.update_count),
                            policy_version=int(learner.get_policy_version()),
                            run_dir=artifacts.run_dir,
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            artifact_dir_name="dev_eval",
                            artifact_scope="periodic_dev_eval",
                            paired_seeds=tuple(_periodic_dev_eval_schedule(stack)[2]),
                            opponents=tuple(opponent_specs),
                            eval_device_override=async_eval_device,
                            parallel_workers=max(
                                1,
                                int(
                                    getattr(
                                        stack.config.evaluation,
                                        "periodic_dev_eval_parallel_workers",
                                        1,
                                    )
                                ),
                            ),
                            parallel_worker_devices=_resolved_periodic_dev_eval_worker_devices(
                                stack=stack,
                                parallel_workers=max(
                                    1,
                                    int(
                                        getattr(
                                            stack.config.evaluation,
                                            "periodic_dev_eval_parallel_workers",
                                            1,
                                        )
                                    ),
                                ),
                                explicit_worker_devices=tuple(
                                    getattr(
                                        stack.config.evaluation,
                                        "periodic_dev_eval_parallel_worker_devices",
                                        (),
                                    )
                                ),
                                eval_device=str(
                                    getattr(
                                        stack.config.evaluation,
                                        "eval_device",
                                        "cpu",
                                    )
                                ),
                                learner_device=device,
                            ),
                        )
                        if (
                            periodic_dev_eval_process_pool is None
                            and int(request.parallel_workers) > 1
                        ):
                            periodic_dev_eval_process_pool = ProcessPoolExecutor(
                                max_workers=int(request.parallel_workers),
                                mp_context=mp.get_context("spawn"),
                            )
                        pending_periodic_dev_eval = PendingPeriodicDevEval(
                            future=async_periodic_dev_eval_executor.submit(
                                _run_async_periodic_dev_eval_worker,
                                request,
                                periodic_dev_eval_process_pool,
                            ),
                            request=request,
                            pinned_snapshot_ids=tuple(newly_pinned_snapshot_ids),
                            latest_metrics=dict(latest_metrics),
                        )
                        print(
                            "Periodic dev eval scheduled: "
                            f"update={int(learner.update_count)} "
                            f"devices={','.join(request.parallel_worker_devices) or str(stack.config.evaluation.eval_device)} "
                            f"anchors={','.join(spec.display_name for spec in request.opponents)}"
                        )
                        if defer_noleague_baseline_alias_refresh:
                            _ensure_noleague_baseline_anchor(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                learner=learner,
                                device=device,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                permit_current_run_alias=True,
                                source_checkpoint_path=checkpoint_path,
                                update=int(learner.update_count),
                            )
                    else:
                        if (
                            periodic_dev_eval_process_pool is None
                            and stack.config.evaluation is not None
                            and int(getattr(stack.config.evaluation, "periodic_dev_eval_parallel_workers", 1)) > 1
                        ):
                            periodic_dev_eval_process_pool = ProcessPoolExecutor(
                                max_workers=int(stack.config.evaluation.periodic_dev_eval_parallel_workers),
                                mp_context=mp.get_context("spawn"),
                            )
                        summary_payload = _run_periodic_dev_eval(
                            stack=stack,
                            contract=contract,
                            artifacts=artifacts,
                            training_paths=training_paths,
                            learner=learner,
                            device=device,
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            process_pool_executor=periodic_dev_eval_process_pool,
                        )
                        anchor_keys = sorted(cast(dict[str, Any], summary_payload["anchor_scores"]).keys())
                        opponent_fragment = f" opponent={_slug_policy_id(anchor_keys[0])}" if anchor_keys else ""
                        print(
                            "Periodic dev eval: "
                            f"update={learner.update_count}{opponent_fragment} "
                            f"aggregate={summary_payload['aggregate_score']:.4f} "
                            f"anchors={','.join(anchor_keys)}"
                        )
                        effective_summary = summary_payload
                        tracker_before_dev_eval = _load_checkpoint_tracker(training_paths)
                        existing_best_record = tracker_before_dev_eval.get("best")
                        if not isinstance(existing_best_record, Mapping):
                            existing_best_record = None
                        confirmatory_request = _confirmatory_dev_eval_request(
                            stack=stack,
                            existing_best_record=cast(Mapping[str, Any] | None, existing_best_record),
                            dev_eval_summary=effective_summary,
                        )
                        if confirmatory_request is not None:
                            seed_file, _validated_sources, base_paired_seeds, seed_file_sha256 = (
                                _periodic_dev_eval_schedule(stack)
                            )
                            confirmatory_pairs = _expand_periodic_dev_eval_paired_seeds(
                                base_paired_seeds,
                                requested_pairs=int(confirmatory_request["target_pairs"]),
                                seed_file_sha256=seed_file_sha256,
                                update_count=int(learner.update_count),
                                policy_version=int(learner.get_policy_version()),
                                scope="periodic_dev_eval_confirmatory",
                            )
                            effective_summary = _run_periodic_dev_eval(
                                stack=stack,
                                contract=contract,
                                artifacts=artifacts,
                                training_paths=training_paths,
                                learner=learner,
                                device=device,
                                run_id256=run_id256,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                artifact_dir_name="dev_eval_confirmatory",
                                artifact_scope="periodic_dev_eval_confirmatory",
                                paired_seeds_override=confirmatory_pairs,
                                persist_summary=False,
                                update_stall_monitor=False,
                                batched_inference_override=False,
                                process_pool_executor=periodic_dev_eval_process_pool,
                            )
                            print(
                                "Confirmatory dev eval: "
                                f"update={learner.update_count} paired_seeds={len(confirmatory_pairs)} "
                                f"aggregate={effective_summary['aggregate_score']:.4f} "
                                f"reasons={','.join(cast(list[str], confirmatory_request['reasons']))} "
                                f"seed_file={seed_file.name}"
                            )
                        if _dev_eval_is_authoritative(effective_summary):
                            _persist_periodic_dev_eval_summary(training_paths=training_paths, payload=effective_summary)
                        else:
                            _persist_periodic_dev_eval_fast_screen(training_paths=training_paths, payload=effective_summary)
                        last_dev_eval_summary = effective_summary
                        last_dev_eval_update_count = int(learner.update_count)
                        league_eval_warmup_gate_status = _sync_runtime_league_eval_warmup_gate(
                            runtime=runtime,
                            stack=stack,
                            dev_eval_summary=effective_summary,
                        )
                        league_eval_warmup_gate_open = bool(league_eval_warmup_gate_status["open"])
                        if bool(league_eval_warmup_gate_status.get("enabled", False)):
                            print(
                                "League eval warmup gate: "
                                f"open={league_eval_warmup_gate_open} "
                                f"reasons={','.join(cast(list[str], league_eval_warmup_gate_status.get('reasons', [])))}"
                            )
                        ckpt_path = _ensure_current_checkpoint(
                            training_paths=training_paths,
                            learner=learner,
                            stack=stack,
                            device=device,
                            spec_hash256=spec_hash256,
                            algorithm=algorithm,
                        )
                        tracker_payload = _publish_checkpoint_aliases(
                            stack=stack,
                            training_paths=training_paths,
                            artifacts=artifacts,
                            checkpoint_path=ckpt_path,
                            learner=learner,
                            latest_metrics=latest_metrics,
                            dev_eval_summary=effective_summary,
                        )
                        _maybe_log_structured_mainmove_guard(
                            training_paths=training_paths,
                            learner=learner,
                            latest_metrics=latest_metrics,
                            dev_eval_summary=effective_summary,
                        )
                        guard_event = _maybe_rollback_to_best_checkpoint(
                            stack=stack,
                            training_paths=training_paths,
                            artifacts=artifacts,
                            runtime=runtime,
                            learner=learner,
                            model=model,
                            device=device,
                            spec_hash256=spec_hash256,
                            algorithm=algorithm,
                            latest_metrics=latest_metrics,
                            dev_eval_summary=effective_summary,
                            last_rollback_update=last_checkpoint_guard_rollback_update,
                        )
                        if guard_event is not None:
                            last_checkpoint_guard_rollback_update = int(learner.update_count)
                            pending_promotion_gate = _drop_stale_pending_promotion_gate(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                pending_gate=pending_promotion_gate,
                                rollback_best_update_count=int(guard_event["best_update_count"]),
                            )
                            prefetched_runtime_batch = None
                            print(
                                "Checkpoint guard rollback: "
                                f"update={guard_event['update_count']} "
                                f"best_update={guard_event['best_update_count']} "
                                f"current_score={float(guard_event['current_score']):.4f} "
                                f"best_score={float(guard_event['best_score']):.4f} "
                                f"reasons={','.join(cast(list[str], guard_event['reasons']))}"
                            )
                        if defer_noleague_baseline_alias_refresh:
                            _ensure_noleague_baseline_anchor(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                learner=learner,
                                device=device,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                permit_current_run_alias=True,
                                update=int(learner.update_count),
                            )
                        if tensorboard_logger is not None:
                            tensorboard_logger.log_periodic_dev_eval(effective_summary, step=int(learner.update_count))
                            tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
                        audit_request = _maybe_request_b2_disagreement_audit(
                            stack=stack,
                            training_paths=training_paths,
                            artifacts=artifacts,
                            dev_eval_summary=effective_summary,
                        )
                        if audit_request is not None:
                            latest_metrics["b2_disagreement_audit_requested"] = 1.0
                            print(
                                "B2 disagreement audit requested: "
                                f"update={int(audit_request['update_count'])} "
                                f"reasons={','.join(cast(list[str], audit_request['trigger_reasons']))} "
                                f"episodes={audit_request['episodes_path']}"
                            )
                        early_cutoff_payload = _update_early_cutoff(
                            stack=stack,
                            training_paths=training_paths,
                            update_count=int(learner.update_count),
                            summary_payload=effective_summary,
                        )
                        if early_cutoff_payload is not None and bool(early_cutoff_payload.get("should_stop", False)):
                            latest_metrics.update(
                                {
                                    "early_cutoff_triggered": 1.0,
                                    "early_cutoff_best_score": float(early_cutoff_payload["best_score"]),
                                    "early_cutoff_current_score": float(early_cutoff_payload["current_score"]),
                                    "early_cutoff_no_improvement_updates": float(
                                        early_cutoff_payload["no_improvement_updates"]
                                    ),
                                    "early_cutoff_consecutive_stall_evals": float(
                                        early_cutoff_payload["consecutive_stall_evals"]
                                    ),
                                }
                            )
                            print(
                                "Early cutoff triggered: "
                                f"update={int(learner.update_count)} "
                                f"best_update={int(early_cutoff_payload['best_update_count'])} "
                                f"best_score={float(early_cutoff_payload['best_score']):.4f} "
                                f"current_score={float(early_cutoff_payload['current_score']):.4f} "
                                f"reasons={','.join(cast(list[str], early_cutoff_payload['reasons']))}"
                            )
                            break
        finally:
            if pending_promotion_gate is not None:
                promotion_passed = _process_completed_promotion_gate(
                    pending_gate=pending_promotion_gate,
                    stack=stack,
                    artifacts=artifacts,
                    training_paths=training_paths,
                )
                pending_promotion_gate = None
                if promotion_passed:
                    runtime.refresh_opponent_pool()
            if pending_periodic_dev_eval is not None:
                completed_summary, guard_event = _process_completed_periodic_dev_eval(
                    pending_eval=pending_periodic_dev_eval,
                    stack=stack,
                    contract=contract,
                    artifacts=artifacts,
                    training_paths=training_paths,
                    runtime=runtime,
                    learner=learner,
                    device=device,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                    last_rollback_update=last_checkpoint_guard_rollback_update,
                    tensorboard_logger=tensorboard_logger,
                    process_pool_executor=periodic_dev_eval_process_pool,
                )
                pending_periodic_dev_eval = None
                last_dev_eval_summary = completed_summary
                last_dev_eval_update_count = int(completed_summary["update_count"])
                if guard_event is not None:
                    last_checkpoint_guard_rollback_update = int(learner.update_count)
                    pending_promotion_gate = _drop_stale_pending_promotion_gate(
                        stack=stack,
                        training_paths=training_paths,
                        run_dir=artifacts.run_dir,
                        pending_gate=pending_promotion_gate,
                        rollback_best_update_count=int(guard_event["best_update_count"]),
                    )
                    prefetched_runtime_batch = None
            if prefetch_executor is not None:
                prefetch_executor.shutdown(wait=False, cancel_futures=True)
            if async_promotion_gate_executor is not None:
                async_promotion_gate_executor.shutdown(wait=True, cancel_futures=False)
            if async_periodic_dev_eval_executor is not None:
                async_periodic_dev_eval_executor.shutdown(wait=True, cancel_futures=False)
            if periodic_dev_eval_process_pool is not None:
                periodic_dev_eval_process_pool.shutdown(wait=True, cancel_futures=False)
            runtime.close()

    if profiler is not None and profiler_trace_dir is not None:
        trace_path = profiler_trace_dir / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        print(f"Wrote torch profiler trace: {trace_path}")

    if rank0 and _is_noleague_baseline_role(experiment_role):
        _ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            learner=learner,
            device=device,
            config_hash256=config_hash256,
            permit_current_run_alias=True,
            update=int(learner.update_count),
        )

    if not latest_metrics:
        raise RuntimeError("The canonical single-node run finished without producing learner metrics")
    if not rank0:
        return latest_metrics
    final_checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    final_dev_eval_summary = last_dev_eval_summary if last_dev_eval_update_count == int(learner.update_count) else None
    tracker_payload = _publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=final_checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    finalize_guard_event = _maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    if finalize_guard_event is not None:
        print(
            "Checkpoint guard final selection: "
            f"update={finalize_guard_event['update_count']} "
            f"best_update={finalize_guard_event['best_update_count']} "
            f"current_score={float(finalize_guard_event['current_score']):.4f} "
            f"best_score={float(finalize_guard_event['best_score']):.4f}"
        )
        tracker_payload = _load_checkpoint_tracker(training_paths)
    if tensorboard_logger is not None:
        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
    if early_cutoff_payload is not None and bool(early_cutoff_payload.get("should_stop", False)):
        latest_metrics.setdefault("early_cutoff_triggered", 1.0)
    return latest_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical single-node thesis training entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Expected spec hash or spec bundle SHA-256")
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Stage the built-in public-safe toy catalog/policy bundle instead of probing weiss_sim.",
    )
    parser.add_argument(
        "--config-hash",
        type=str,
        default="",
        help="Expected config_hash256 for contract validation",
    )
    parser.add_argument("--run-label", type=str, default="", help="Optional run directory label override")
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--override",
        "--config-override",
        dest="config_override",
        action="append",
        default=None,
        help="Deterministic config override in KEY=JSON_VALUE form, e.g. training.optimizer.learning_rate=0.0001",
    )
    parser.add_argument("--num-envs", type=int, default=2, help="Env count for the single-node training run")
    parser.add_argument("--unroll-length", type=int, default=4, help="Tiny rollout length for the smoke run")
    parser.add_argument("--max-updates", type=int, default=1, help="Number of learner updates to run")
    parser.add_argument(
        "--autoscale",
        action="store_true",
        help="Resolve training topology from training.scaling and the selected hardware profile",
    )
    parser.add_argument(
        "--autoscale-dry-run",
        action="store_true",
        help="Print resolved scaling topology and exit before writing run artifacts",
    )
    parser.add_argument(
        "--hardware-profile",
        type=str,
        default="local",
        help="Autoscale hardware profile: local, uc1-l40-3, uc1-l40-4, 8gpu-l40, or gpu<N>",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable torch.distributed learner gradient averaging; intended for torchrun",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="auto",
        choices=("auto", "nccl", "gloo"),
        help="torch.distributed backend for --ddp or WORLD_SIZE>1",
    )
    parser.add_argument(
        "--max-wall-clock-minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget in minutes; the run stops cleanly between updates when the budget is reached",
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="train_ordered",
        choices=("train_ordered", "train_async_fast"),
        help="Queue runtime mode: deterministic ordered collection or throughput-oriented async-fast collection",
    )
    parser.add_argument(
        "--profile-timers",
        action="store_true",
        help="Enable cheap runtime/learner timers and record_function ranges without emitting a torch profiler trace",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Emit a torch profiler trace under profiling/torch_profiler/trace.json",
    )
    parser.add_argument("--profile", type=str, default="", help="Optional simulator profile override")
    parser.add_argument("--device", type=str, default="", help="Optional learner device override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument(
        "--checkpoint-interval-updates",
        type=int,
        default=None,
        help="Optional checkpoint cadence override for the single-node training run",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--dev-eval-summaries-json",
        type=Path,
        default=None,
        help="Optional dev-eval summaries JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help="Completed baseline_noleague run directory used to import the canonical B1 baseline anchor",
    )
    parser.add_argument(
        "--seed-snapshot-run-dir",
        type=Path,
        default=None,
        help="Optional completed run directory whose snapshot registry should be imported into the current training league before update 1",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume training in-place inside an existing run directory",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Checkpoint path or alias (`latest`/`best`) to restore before continuing training",
    )
    parser.add_argument(
        "--resume-allow-config-mismatch",
        action="store_true",
        help=(
            "Allow loading a checkpoint whose config hash differs from the current stack. "
            "Spec hash and algorithm are still checked; use only for explicit research continuations."
        ),
    )
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="Load checkpoint model weights and counters but start with a fresh optimizer and grad scaler state.",
    )
    args = parser.parse_args()
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    num_envs = _require_positive_int("--num-envs", args.num_envs)
    unroll_length = _require_positive_int("--unroll-length", args.unroll_length)
    max_updates = _require_positive_int("--max-updates", args.max_updates)
    max_wall_clock_minutes = _require_positive_optional_float(
        "--max-wall-clock-minutes",
        args.max_wall_clock_minutes,
    )
    stack = load_stack_config(args.stack_config)
    stack = apply_stack_overrides(stack, parse_override_tokens(args.config_override))
    stack = _apply_training_flag_overrides(
        stack,
        enable_profile_timers=bool(args.profile_timers),
        enable_torch_profiler=bool(args.torch_profiler),
    )
    training_config = stack.config.training
    manifest_only_reason = _manifest_scaffold_only_reason(stack)
    if training_config is None and manifest_only_reason is None:
        parser.error("stack config is missing training")

    public_demo_enabled = bool(args.public_demo)
    resume_run_dir = None if args.resume_run_dir is None else args.resume_run_dir.resolve()
    resume_checkpoint_path = _resolve_resume_checkpoint_path(
        resume_from=str(args.resume_from),
        resume_run_dir=resume_run_dir,
    )
    seed_snapshot_run_dir = (
        args.seed_snapshot_run_dir.resolve()
        if args.seed_snapshot_run_dir is not None
        else _infer_seed_snapshot_run_dir_from_resume_checkpoint(
            stack=stack,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_run_dir=resume_run_dir,
        )
    )
    seed_snapshot_run_dir_auto_inferred = args.seed_snapshot_run_dir is None and seed_snapshot_run_dir is not None
    if public_demo_enabled and (resume_run_dir is not None or resume_checkpoint_path is not None):
        parser.error("Public demo mode does not support checkpoint resume")
    if public_demo_enabled:
        public_demo_bundle = public_demo_spec_bundle()
        assert_spec_bundle_contract(args.spec_hash, public_demo_bundle)
        spec_bundle = public_demo_bundle
        spec_hash256 = public_demo_spec_hash256()
        simulator_info = public_demo_simulator_info()
    else:
        simulator_contract = load_verified_simulator_contract(stack.root, expected_spec_hash=args.spec_hash)
        spec_bundle = simulator_contract.spec_bundle
        spec_hash256 = simulator_contract.spec_hash256
        simulator_info = simulator_contract.simulator
    config_hash256 = compute_config_hash256(stack)
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    ddp_context = distributed_context_from_env(force=bool(args.ddp), backend=str(args.ddp_backend))
    if ddp_context.enabled and torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
        torch.cuda.set_device(int(ddp_context.local_rank) % int(torch.cuda.device_count()))
    ddp_context = init_process_group_if_needed(ddp_context)
    rank0 = (not ddp_context.enabled) or ddp_context.is_rank0
    resolved_topology: ResolvedTrainingTopology | None = None
    if bool(args.autoscale or args.autoscale_dry_run):
        resolved_topology = _resolve_autoscale_topology(
            stack=stack,
            hardware_profile_name=str(args.hardware_profile),
            runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
        )
        if not ddp_context.enabled and str(resolved_topology.resolved_learner_parallelism) == "ddp":
            if args.autoscale_dry_run:
                pass
            else:
                parser.error(
                    "autoscale resolved a multi-GPU DDP topology; launch with torchrun/--ddp or use --autoscale-dry-run"
                )
        if args.autoscale_dry_run:
            if rank0:
                print(
                    json.dumps(
                        {
                            "format": "autoscale_training_topology_v1",
                            "hardware_profile": str(args.hardware_profile),
                            "runtime_mode": str(args.runtime_mode),
                            "scaling_request": _scaling_request_from_config(training_config).to_dict()
                            if hasattr(_scaling_request_from_config(training_config), "to_dict")
                            else _scaling_request_from_config(training_config).__dict__,
                            "resolved_topology": resolved_topology.to_dict(),
                            "distributed": ddp_context.to_dict(),
                        },
                        sort_keys=True,
                        indent=2,
                    )
                )
            destroy_process_group_if_initialized()
            return
        num_envs = int(resolved_topology.total_envs)

    git_commit = _git_commit()
    start_nonce = int(broadcast_object(_start_nonce() if rank0 else None, context=ddp_context))
    manifest_dict: dict[str, Any] | None = None
    if resume_run_dir is None:
        run_id256 = compute_run_id256(spec_hash256, config_hash256, git_commit or None, start_nonce)
        run_id64 = f"{compute_run_id64(spec_hash256, config_hash256, git_commit or None, start_nonce):016x}"
        run_dir_name = run_label or default_run_dir_name(run_id64)
    else:
        artifacts = _run_artifacts_from_existing_run_dir(resume_run_dir)
        manifest_dict = _load_json_object(artifacts.manifest_path, label="resume manifest")
        run_id256 = str(manifest_dict.get("run_id256", "")).strip().lower()
        run_id64 = str(manifest_dict.get("run_id64", "")).strip().lower()
        run_dir_name = artifacts.run_dir_name
        existing_spec_hash = str(manifest_dict.get("spec_hash256", "")).strip().lower()
        existing_config_hash = str(manifest_dict.get("config_hash256", "")).strip().lower()
        if existing_spec_hash != spec_hash256:
            raise RuntimeError(
                f"resume run spec hash mismatch: expected {spec_hash256}, found {existing_spec_hash} in {artifacts.manifest_path}"
            )
        if existing_config_hash != config_hash256:
            raise RuntimeError(
                f"resume run config hash mismatch: expected {config_hash256}, found {existing_config_hash} in {artifacts.manifest_path}"
            )

    print_startup_banner(
        spec_hash256,
        config_hash256,
        run_id64=run_id64,
        run_id256=run_id256,
        run_label=run_label or ("" if resume_run_dir is None else run_dir_name),
        run_dir_name=run_dir_name,
        spec_mismatch_policy=_spec_mismatch_policy(stack),
    )
    spec_bundle_message = (
        "Loaded synthetic public-demo spec bundle: " if public_demo_enabled else "Verified runtime spec bundle: "
    )
    print(spec_bundle_message + f"compat={simulator_info.get('compatibility_hash', '')} sha256={spec_hash256}")
    print(f"Loaded stack config with {len(stack.components)} components")

    if ddp_context.enabled:
        device_override = str(args.device).strip().lower()
        if device_override and device_override != "auto":
            device = torch.device(device_override)
        elif torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
            device = torch.device(f"cuda:{int(ddp_context.local_rank) % int(torch.cuda.device_count())}")
        else:
            device = torch.device("cpu")
    else:
        device = _resolve_device(stack, args.device)
    profile = _resolve_runtime_profile(stack, args.profile)
    seed = _resolve_seed(stack, args.seed)
    actor_device_layout = _manifest_actor_device_layout(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=int(spec_bundle["action"]["pass_action_id"]),
        runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
        learner_device=device,
        resolved_topology=resolved_topology,
    )
    policy_set_selection, policy_set_selection_details = _resolve_policy_set_selection(
        stack,
        snapshot_registry_path=args.snapshot_registry_json,
        dev_eval_summaries_path=args.dev_eval_summaries_json,
    )
    manifest = RunManifest(
        run_id256=run_id256,
        run_id64=run_id64,
        start_nonce=start_nonce,
        git_commit=git_commit,
        git_dirty=_git_dirty(),
        spec_hash256=spec_hash256,
        config_hash256=config_hash256,
        simulator=simulator_info,
        spec_bundle=spec_bundle,
        config_canonical=canonical_config_dict(stack),
        seed_files=build_seed_file_manifest(stack.seed_sets, root=stack.root),
        hardware=_hardware_summary(
            device,
            actor_device=("cpu" if stack.config.system is None else stack.config.system.actor_device),
            actor_device_layout=actor_device_layout,
        ),
        evaluation_pinning=_evaluation_pinning(stack),
        policy_set_selection=policy_set_selection,
        policy_set_selection_details=policy_set_selection_details,
    )
    if resume_run_dir is None:
        if rank0:
            artifacts = write_run_artifacts(
                stack.root / "runs",
                manifest,
                run_label=run_label or None,
            )
            run_dir_text = artifacts.run_dir.as_posix()
        else:
            run_dir_text = ""
        run_dir_text = str(broadcast_object(run_dir_text, context=ddp_context))
        if not rank0:
            artifacts = _run_artifacts_from_existing_run_dir(Path(run_dir_text))
        distributed_barrier(ddp_context)
    else:
        artifacts = _run_artifacts_from_existing_run_dir(resume_run_dir)
    tensorboard_logger: TensorBoardLogger | None = None
    if rank0:
        run_summary_payload = _load_json_object(artifacts.run_summary_path, label="run summary")
        run_summary_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
        run_summary_payload["policy_set_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
        run_summary_payload["distributed"] = ddp_context.to_dict()
        if resolved_topology is not None:
            run_summary_payload["autoscale_topology"] = resolved_topology.to_dict()
        if training_config is not None:
            run_summary_payload["training_controls"] = {
                "profile_timers": bool(training_config.profile_timers),
                "torch_profiler": bool(training_config.torch_profiler),
                "structured_metrics_mode": str(training_config.structured_metrics_mode),
                "teacher_aux_mode": str(training_config.teacher_aux_mode),
                "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
                "heuristic_native_rollout_enabled": bool(training_config.heuristic_native_rollout_enabled),
                "heuristic_native_rollout_profile": str(training_config.heuristic_native_rollout_profile),
                "heuristic_native_rollout_profiles": list(training_config.heuristic_native_rollout_profiles),
                "heuristic_native_rollout_profile_mode": str(training_config.heuristic_native_rollout_profile_mode),
                "max_wall_clock_minutes": None if max_wall_clock_minutes is None else float(max_wall_clock_minutes),
            }
        if args.b1_baseline_run_dir is not None:
            run_summary_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
        if seed_snapshot_run_dir is not None:
            run_summary_payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.as_posix()
            run_summary_payload["seed_snapshot_run_dir_auto_inferred"] = seed_snapshot_run_dir_auto_inferred
        run_summary_payload["stack_config_path"] = args.stack_config.resolve().as_posix()
        if resume_checkpoint_path is not None:
            run_summary_payload["resume"] = {
                "enabled": True,
                "resume_run_dir": None if resume_run_dir is None else resume_run_dir.as_posix(),
                "resume_checkpoint_path": resume_checkpoint_path.as_posix(),
                "reset_optimizer": bool(args.resume_reset_optimizer),
            }
        _write_json(artifacts.run_summary_path, run_summary_payload)

        determinism_payload = _load_json_object(artifacts.determinism_report_path, label="determinism report")
        determinism_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
        determinism_payload["policy_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
        determinism_payload["distributed"] = ddp_context.to_dict()
        if resolved_topology is not None:
            determinism_payload["autoscale_topology"] = resolved_topology.to_dict()
        if training_config is not None:
            determinism_payload["training_controls"] = {
                "profile_timers": bool(training_config.profile_timers),
                "torch_profiler": bool(training_config.torch_profiler),
                "structured_metrics_mode": str(training_config.structured_metrics_mode),
                "teacher_aux_mode": str(training_config.teacher_aux_mode),
                "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
                "heuristic_native_rollout_enabled": bool(training_config.heuristic_native_rollout_enabled),
                "heuristic_native_rollout_profile": str(training_config.heuristic_native_rollout_profile),
                "heuristic_native_rollout_profiles": list(training_config.heuristic_native_rollout_profiles),
                "heuristic_native_rollout_profile_mode": str(training_config.heuristic_native_rollout_profile_mode),
            }
        if args.b1_baseline_run_dir is not None:
            determinism_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
        if seed_snapshot_run_dir is not None:
            determinism_payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.as_posix()
            determinism_payload["seed_snapshot_run_dir_auto_inferred"] = seed_snapshot_run_dir_auto_inferred
        if resume_checkpoint_path is not None:
            determinism_payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
            determinism_payload["resume_reset_optimizer"] = bool(args.resume_reset_optimizer)
        _write_json(artifacts.determinism_report_path, determinism_payload)

        environment_payload = _load_json_object(artifacts.environment_path, label="environment manifest")
        environment_payload["cwd"] = stack.root.as_posix()
        environment_payload["argv"] = sys.argv
        environment_payload["hardware"] = manifest.hardware
        environment_payload["distributed"] = ddp_context.to_dict()
        if resolved_topology is not None:
            environment_payload["autoscale_topology"] = resolved_topology.to_dict()
        if resume_checkpoint_path is not None:
            environment_payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
            environment_payload["resume_reset_optimizer"] = bool(args.resume_reset_optimizer)
        _write_json(artifacts.environment_path, environment_payload)
        tensorboard_logger = TensorBoardLogger(artifacts.layout.tensorboard_dir)
        if not tensorboard_logger.enabled:
            unavailable_reason = tensorboard_unavailable_reason()
            print(
                "TensorBoard logging is disabled: "
                + ("SummaryWriter unavailable" if unavailable_reason is None else unavailable_reason),
                file=sys.stderr,
            )
        else:
            tensorboard_logger.log_run_context(
                manifest=manifest.to_dict(),
                environment=environment_payload,
                run_summary=run_summary_payload,
                determinism_report=determinism_payload,
            )
        if resume_run_dir is None:
            print(f"Wrote manifest: {artifacts.manifest_path}")
        else:
            print(f"Resuming existing run directory: {artifacts.run_dir}")

    try:
        if public_demo_enabled:
            staged = stage_public_demo_run(artifacts.run_dir)
            print(
                "Staged public-demo toy catalog and policy bundle: "
                f"mode={PUBLIC_DEMO_MODE} policy_count={len(staged.policy_ids)} "
                f"catalog={staged.catalog_path}"
            )
            print(
                "Public demo mode is intentionally synthetic and demo-only. "
                "It does not execute simulator training or claim thesis-grade results."
            )
            return

        if manifest_only_reason is not None:
            _print_manifest_only_message(manifest_only_reason)
            return

        runtime_prerequisite_failure = _runtime_training_prerequisite_failure(stack)
        if runtime_prerequisite_failure is not None:
            _raise_runtime_prerequisite_failure(runtime_prerequisite_failure)

        assert training_config is not None
        checkpoint_interval_updates = _require_positive_int(
            "--checkpoint-interval-updates",
            args.checkpoint_interval_updates
            if args.checkpoint_interval_updates is not None
            else int(training_config.checkpoint_interval_updates),
        )

        profile_timers = bool(training_config.profile_timers)
        torch_profiler = bool(training_config.torch_profiler)
        if profile_timers or torch_profiler:
            print(
                "Structured profiling enabled: "
                f"profile_timers={profile_timers} "
                f"torch_profiler={torch_profiler} "
                f"structured_metrics_mode={training_config.structured_metrics_mode} "
                f"teacher_aux_mode={training_config.teacher_aux_mode} "
                f"fixed_opponent_backend={training_config.fixed_opponent_backend}"
            )

        metrics = _run_minimal_training(
            stack=stack,
            contract=simulator_contract,
            artifacts=artifacts,
            num_envs=num_envs,
            unroll_length=unroll_length,
            max_updates=max_updates,
            max_wall_clock_minutes=max_wall_clock_minutes,
            profile=profile,
            device=device,
            seed=seed,
            checkpoint_interval_updates=checkpoint_interval_updates,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
            b1_baseline_run_dir=None if args.b1_baseline_run_dir is None else args.b1_baseline_run_dir.resolve(),
            seed_snapshot_run_dir=seed_snapshot_run_dir,
            seed_snapshot_run_dir_auto_inferred=seed_snapshot_run_dir_auto_inferred,
            profile_timers=profile_timers,
            torch_profiler=torch_profiler,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_allow_config_mismatch=bool(args.resume_allow_config_mismatch),
            resume_reset_optimizer=bool(args.resume_reset_optimizer),
            tensorboard_logger=tensorboard_logger,
            resolved_topology=resolved_topology,
            distributed_context=ddp_context,
        )
        print(
            "Completed canonical single-node training run: "
            f"loss={metrics.get('loss', 0.0):.6f} "
            f"policy_loss={metrics.get('policy_loss', 0.0):.6f} "
            f"value_loss={metrics.get('value_loss', 0.0):.6f} "
            f"entropy={metrics.get('entropy', 0.0):.6f}"
        )
        if float(metrics.get("early_cutoff_triggered", 0.0)) >= 0.5:
            print(
                "Training stopped by early cutoff: "
                f"best_score={metrics.get('early_cutoff_best_score', 0.0):.4f} "
                f"current_score={metrics.get('early_cutoff_current_score', 0.0):.4f} "
                f"no_improvement_updates={int(metrics.get('early_cutoff_no_improvement_updates', 0.0))} "
                f"consecutive_stall_evals={int(metrics.get('early_cutoff_consecutive_stall_evals', 0.0))}"
            )
    finally:
        if tensorboard_logger is not None:
            tensorboard_logger.close()
        destroy_process_group_if_initialized()


if __name__ == "__main__":
    main()
