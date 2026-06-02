"""Promotion-gate runner used by the training entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.diagnostics.action_diagnostics import (
    ActionSummaryCounters,
    make_action_sequence_state,
    update_eval_action_counters,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval import Pcg32XshRrV1, game_result_from_step, sample_action_pinned, select_action_argmax_pinned
from weiss_rl.eval.harness import ScheduledGame, abort_on_engine_fault_eval
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.model_sampling import model_eval_logits_for_legal_ids
from weiss_rl.model import PolicyValueModel
from weiss_rl.models.observation_contract import header_field_index
from weiss_rl.runtime.components.action_surface import (
    filter_batch_main_move_only_rows_to_pass,
    filter_batch_mulligan_select_after_select,
    filter_batch_pass_when_attack_available,
)
from weiss_rl.runtime.components.legal_meta import action_catalog_indices
from weiss_rl.training.dev_eval import legal_ids_for_env_row, promotion_gate_rng_seed


class PromotionGateRunner:
    def __init__(
        self,
        *,
        stack: Any,
        focal_policy_id: str,
        focal_model: PolicyValueModel,
        anchor_models: dict[str, PolicyValueModel],
        heuristic_policies: dict[str, HeuristicPublicPolicy],
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        require_sorted_legal_ids: bool,
        build_eval_env: Callable[..., Any],
        random_legal_policy_id: str,
    ) -> None:
        self.stack = stack
        self.focal_policy_id = focal_policy_id
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pass_action_id = pass_action_id
        self.artifact_dir = artifact_dir
        self.require_sorted_legal_ids = require_sorted_legal_ids
        self.build_eval_env = build_eval_env
        self.random_legal_policy_id = str(random_legal_policy_id)
        self._policy_models = {focal_policy_id: focal_model, **anchor_models}
        self._heuristic_policies = dict(heuristic_policies)
        self._baseline_logits = np.zeros((action_dim,), dtype=np.float32)
        self._device = torch.device("cpu")
        evaluation_config = getattr(getattr(self.stack, "config", None), "evaluation", None)
        self._eval_sampling_algorithm = str(
            getattr(evaluation_config, "eval_sampling_algorithm", "pinned_cdf_pcg_v1") or "pinned_cdf_pcg_v1"
        ).strip()
        self._model_sampling_temperature = float(getattr(evaluation_config, "model_sampling_temperature", 1.0) or 1.0)
        training_config = getattr(getattr(self.stack, "config", None), "training", None)
        self._mulligan_force_confirm_after_select = bool(
            getattr(training_config, "mulligan_force_confirm_after_select", False)
        )
        self._force_pass_over_main_move_only = bool(getattr(training_config, "force_pass_over_main_move_only", False))
        self._main_move_only_max_consecutive = int(getattr(training_config, "main_move_only_max_consecutive", 0))
        self._force_attack_over_pass_when_attack_legal = bool(
            getattr(training_config, "force_attack_over_pass_when_attack_legal", False)
        )

    def run_game(self, scheduled_game: ScheduledGame):
        env = self.build_eval_env(
            self.stack,
            seed=scheduled_game.episode_seed,
            pass_action_id=self.pass_action_id,
        )
        seat_hidden = {
            seat: self._initial_hidden(scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id)
            for seat in (0, 1)
        }
        seat_rngs = {
            seat: Pcg32XshRrV1(promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=seat)) for seat in (0, 1)
        }
        last_acting_seat: int | None = None
        action_counters = ActionSummaryCounters()
        action_sequence_state = make_action_sequence_state(1)

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
                legal_ids = legal_ids_for_env_row(
                    batch=batch,
                    env_index=0,
                    require_sorted=self.require_sorted_legal_ids,
                )
                action, next_hidden = self._select_action(
                    batch=batch,
                    current_seat=current_seat,
                    current_policy_id=current_policy_id,
                    seat_hidden=seat_hidden[current_seat],
                    rng=seat_rngs[current_seat],
                    legal_ids=legal_ids,
                    action_sequence_state=action_sequence_state,
                )
                update_eval_action_counters(
                    counters=action_counters,
                    state=action_sequence_state,
                    action=int(action),
                    legal_ids=legal_ids,
                    pass_action_id=self.pass_action_id,
                )
                seat_hidden[current_seat] = next_hidden
                last_acting_seat = current_seat
                batch = env.step(np.asarray([action], dtype=np.uint32))
                self._abort_on_fault(batch)
        finally:
            env.close()

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
        legal_ids: np.ndarray,
        action_sequence_state: Any | None = None,
    ) -> tuple[int, torch.Tensor | None]:
        heuristic_policy = self._heuristic_policies.get(current_policy_id)
        if heuristic_policy is not None:
            action = heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), seat_hidden
        model = self._policy_models.get(current_policy_id)
        if model is None:
            if current_policy_id != self.random_legal_policy_id:
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

        batch_for_model, legal_ids_for_model = self._model_action_surface_batch_and_ids(
            model=model,
            batch=batch,
            legal_ids=legal_ids,
            action_sequence_state=action_sequence_state,
        )
        with torch.inference_mode():
            logits, next_seat_hidden = model_eval_logits_for_legal_ids(
                model=model,
                batch=batch_for_model,
                current_seat=int(current_seat),
                seat_hidden=seat_hidden,
                legal_ids=legal_ids_for_model,
                action_dim=int(self.action_dim),
                device=self._device,
            )
        if self._eval_sampling_algorithm == "model_argmax_pinned_v1":
            action, _ = select_action_argmax_pinned(
                logits,
                legal_ids_for_model,
                pass_action_id=self.pass_action_id,
            )
        else:
            action, _ = sample_action_pinned(
                logits,
                legal_ids_for_model,
                rng=rng,
                pass_action_id=self.pass_action_id,
                temperature=self._model_sampling_temperature,
            )
        return action, next_seat_hidden

    def _model_action_surface_batch_and_ids(
        self,
        *,
        model: PolicyValueModel,
        batch: DecisionBoundaryBatch,
        legal_ids: np.ndarray,
        action_sequence_state: Any | None = None,
    ) -> tuple[DecisionBoundaryBatch, np.ndarray]:
        if (
            not self._mulligan_force_confirm_after_select
            and not self._force_pass_over_main_move_only
            and not self._force_attack_over_pass_when_attack_legal
        ):
            return batch, legal_ids
        action_catalog = getattr(model, "action_catalog", None)
        if action_catalog is None:
            return batch, legal_ids
        filtered_batch = batch
        contract = getattr(model, "_structured_observation_contract", None)
        layout = getattr(contract, "layout", None)
        field_index = None if layout is None else header_field_index(layout, "last_action_arg0")
        last_action_arg0_index = -1 if field_index is None else int(field_index)
        family_index, _attack_type_index = action_catalog_indices(action_catalog)
        if self._mulligan_force_confirm_after_select:
            filtered_batch, _result = filter_batch_mulligan_select_after_select(
                filtered_batch,
                last_action_arg0_index=last_action_arg0_index,
                mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
                mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
            )
        if self._force_pass_over_main_move_only:
            allow_main_move_only_rows = None
            if self._main_move_only_max_consecutive > 0 and action_sequence_state is not None:
                consecutive = np.asarray(action_sequence_state.consecutive_main_moves_by_env, dtype=np.int32)
                if consecutive.shape == (1,):
                    allow_main_move_only_rows = consecutive < self._main_move_only_max_consecutive
            filtered_batch, _result = filter_batch_main_move_only_rows_to_pass(
                filtered_batch,
                pass_action_id=int(self.pass_action_id),
                main_move_family_id=int(family_index.get("main_move", -1)),
                allow_main_move_only_rows=allow_main_move_only_rows,
            )
        if self._force_attack_over_pass_when_attack_legal:
            filtered_batch, _result = filter_batch_pass_when_attack_available(
                filtered_batch,
                pass_action_id=int(self.pass_action_id),
                attack_family_id=int(family_index.get("attack", -1)),
            )
        if filtered_batch.ids_offsets is None:
            return batch, legal_ids
        filtered_ids, filtered_offsets = filtered_batch.ids_offsets
        return (
            filtered_batch,
            np.asarray(filtered_ids[int(filtered_offsets[0]) : int(filtered_offsets[1])], dtype=np.uint32),
        )

    def _abort_on_fault(self, batch: DecisionBoundaryBatch) -> None:
        abort_on_engine_fault_eval(
            run_dir=self.artifact_dir,
            engine_status=batch.engine_status,
            decision_id=batch.decision_id,
            episode_key=batch.episode_key,
            note="engine_status!=0 during promotion gate",
        )
