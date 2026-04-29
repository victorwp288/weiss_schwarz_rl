"""Periodic dev-eval runner used by the training entrypoint."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.eval import Pcg32XshRrV1, game_result_from_step, sample_action_pinned
from weiss_rl.eval.harness import ScheduledGame, abort_on_engine_fault_eval
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.simulator_runner import _resolve_eval_device
from weiss_rl.model import PolicyValueModel
from weiss_rl.training.eval_schedule import AsyncPeriodicDevEvalRequest, AsyncPromotionGateRequest

BuildIdsEvalEnvFn = Callable[[StackConfig], DecisionBoundaryEnv]
LegalIdsForEnvRowFn = Callable[..., np.ndarray]
PeriodicDevEvalSeedFn = Callable[..., int]


@dataclass(slots=True)
class PeriodicDevEvalActiveGame:
    env: DecisionBoundaryEnv
    scheduled_game: ScheduledGame
    batch: DecisionBoundaryBatch
    focal_hidden: torch.Tensor
    opponent_hidden: torch.Tensor | None
    seat_rngs: dict[int, Pcg32XshRrV1]
    last_acting_seat: int | None = None
    completed: bool = False


@dataclass(slots=True)
class PendingPeriodicDevEval:
    future: Future[dict[str, Any]]
    request: AsyncPeriodicDevEvalRequest
    pinned_snapshot_ids: tuple[str, ...]
    latest_metrics: dict[str, float]


@dataclass(slots=True)
class PendingPromotionGate:
    future: Future[dict[str, Any]]
    request: AsyncPromotionGateRequest
    pinned_snapshot_ids: tuple[str, ...]


class PeriodicDevEvalRunner:
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
        build_ids_eval_env_fn: Callable[..., DecisionBoundaryEnv],
        legal_ids_for_env_row_fn: LegalIdsForEnvRowFn,
        periodic_dev_eval_rng_seed_fn: PeriodicDevEvalSeedFn,
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
        self._build_ids_eval_env = build_ids_eval_env_fn
        self._legal_ids_for_env_row = legal_ids_for_env_row_fn
        self._periodic_dev_eval_rng_seed = periodic_dev_eval_rng_seed_fn
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
        active_games = [
            self._start_active_game(scheduled_game, env_seed=env_seed) for scheduled_game in scheduled_games
        ]
        completed: list[tuple[ScheduledGame, Any]] = []
        remaining = len(active_games)
        while remaining > 0:
            ready_slots: list[tuple[int, PeriodicDevEvalActiveGame]] = []
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

    def run_game(self, scheduled_game: ScheduledGame) -> Any:
        self._add_count("games")
        env = self._env_for_game(seed=scheduled_game.episode_seed)
        focal_hidden = self.model.initial_seat_hidden(1, device=self._device)
        opponent_hidden = (
            None if self.opponent_model is None else self.opponent_model.initial_seat_hidden(1, device=self._device)
        )
        seat_rngs = {
            seat: Pcg32XshRrV1(self._periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=seat))
            for seat in (0, 1)
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
            self._persistent_env = self._build_ids_eval_env(
                self.stack,
                seed=seed,
                pass_action_id=self.pass_action_id,
            )
        return self._persistent_env

    def _start_active_game(self, scheduled_game: ScheduledGame, *, env_seed: int) -> PeriodicDevEvalActiveGame:
        self._add_count("games")
        env = self._build_ids_eval_env(
            self.stack,
            seed=env_seed,
            pass_action_id=self.pass_action_id,
        )
        self._batched_envs.append(env)
        started = time.perf_counter()
        batch = env.reset(seed=scheduled_game.episode_seed)
        self._add_seconds("env_reset", time.perf_counter() - started)
        self._abort_on_fault(batch)
        return PeriodicDevEvalActiveGame(
            env=env,
            scheduled_game=scheduled_game,
            batch=batch,
            focal_hidden=self.model.initial_seat_hidden(1, device=self._device),
            opponent_hidden=(
                None if self.opponent_model is None else self.opponent_model.initial_seat_hidden(1, device=self._device)
            ),
            seat_rngs={
                seat: Pcg32XshRrV1(self._periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=seat))
                for seat in (0, 1)
            },
        )

    def _select_batched_actions(
        self,
        ready_slots: Sequence[tuple[int, PeriodicDevEvalActiveGame]],
    ) -> dict[int, int]:
        actions: dict[int, int] = {}
        focal_requests: list[tuple[int, PeriodicDevEvalActiveGame, np.ndarray]] = []
        opponent_requests: list[tuple[int, PeriodicDevEvalActiveGame, np.ndarray]] = []

        for index, active_game in ready_slots:
            batch = active_game.batch
            current_seat = int(batch.actor[0])
            self._add_count("decisions")
            legal_ids = self._legal_ids_for_env_row(
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
        requests: Sequence[tuple[int, PeriodicDevEvalActiveGame, np.ndarray]],
        hidden_name: str,
    ) -> dict[int, int]:
        if not requests:
            return {}
        started = time.perf_counter()
        obs = torch.as_tensor(
            np.stack(
                [np.asarray(active_game.batch.obs[0], dtype=np.float32) for _index, active_game, _legal in requests]
            ),
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
        legal_ids = self._legal_ids_for_env_row(
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
