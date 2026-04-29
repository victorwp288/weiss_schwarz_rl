"""Promotion-gate runner implementation used by the training entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.eval import Pcg32XshRrV1, game_result_from_step, sample_action_pinned
from weiss_rl.eval.harness import ScheduledGame, abort_on_engine_fault_eval
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.simulator_runner import _resolve_eval_device
from weiss_rl.model import PolicyValueModel

LegalIdsForEnvRow = Callable[[DecisionBoundaryBatch, int, bool], np.ndarray]
BuildIdsEvalEnv = Callable[[int], DecisionBoundaryEnv]
PromotionGateRngSeed = Callable[[ScheduledGame, int], int]


class PromotionGateRunnerCore:
    def __init__(
        self,
        *,
        focal_policy_id: str,
        focal_model: PolicyValueModel,
        anchor_models: dict[str, PolicyValueModel],
        heuristic_policies: dict[str, HeuristicPublicPolicy],
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        require_sorted_legal_ids: bool,
        eval_device: torch.device | str,
        randomlegal_policy_id: str,
        build_ids_eval_env: BuildIdsEvalEnv,
        legal_ids_for_env_row: LegalIdsForEnvRow,
        promotion_gate_rng_seed: PromotionGateRngSeed,
    ) -> None:
        self.focal_policy_id = focal_policy_id
        self.action_dim = action_dim
        self.pass_action_id = pass_action_id
        self.artifact_dir = artifact_dir
        self.require_sorted_legal_ids = require_sorted_legal_ids
        self._policy_models = {focal_policy_id: focal_model, **anchor_models}
        self._heuristic_policies = dict(heuristic_policies)
        self._baseline_logits = np.zeros((action_dim,), dtype=np.float32)
        self._device = torch.device(eval_device)
        self._randomlegal_policy_id = str(randomlegal_policy_id)
        self._build_ids_eval_env = build_ids_eval_env
        self._legal_ids_for_env_row = legal_ids_for_env_row
        self._promotion_gate_rng_seed = promotion_gate_rng_seed
        self._persistent_env: DecisionBoundaryEnv | None = None

    @classmethod
    def from_stack(
        cls,
        *,
        stack: Any,
        eval_device: torch.device | str | None,
        **kwargs: Any,
    ) -> PromotionGateRunnerCore:
        return cls(eval_device=_resolve_eval_device(stack, eval_device=eval_device), **kwargs)

    def close(self) -> None:
        env = self._persistent_env
        self._persistent_env = None
        if env is not None:
            env.close()

    def run_game(self, scheduled_game: ScheduledGame) -> Any:
        env = self._env_for_game(seed=scheduled_game.episode_seed)
        seat_hidden = {
            seat: self._initial_hidden(scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id)
            for seat in (0, 1)
        }
        seat_rngs = {
            seat: Pcg32XshRrV1(self._promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=seat))
            for seat in (0, 1)
        }
        last_acting_seat: int | None = None

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
            current_policy_id = scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
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

    def _env_for_game(self, *, seed: int) -> DecisionBoundaryEnv:
        if self._persistent_env is None:
            self._persistent_env = self._build_ids_eval_env(seed)
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
        legal_ids = self._legal_ids_for_env_row(batch, 0, self.require_sorted_legal_ids)
        heuristic_policy = self._heuristic_policies.get(current_policy_id)
        if heuristic_policy is not None:
            action = heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), seat_hidden
        model = self._policy_models.get(current_policy_id)
        if model is None:
            if current_policy_id != self._randomlegal_policy_id:
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
