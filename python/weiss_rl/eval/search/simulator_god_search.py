"""God-search rollout helpers for simulator-backed evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.diagnostics.probes.action_diagnostics import (
    ActionSummaryCounters,
    update_eval_action_counters,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.policies.resolution import ResolvedEvalPolicy
from weiss_rl.eval.sampling.rng_pcg32 import Pcg32XshRrV1
from weiss_rl.eval.search.god_search import GodSearchConfig, GodSearchStats, top_k_legal_actions
from weiss_rl.eval.search.simulator_god_search_outcomes import (
    build_prefix_replay_failure_detail,
    score_terminal_rollout,
)
from weiss_rl.eval.search.simulator_god_search_rollouts import (
    clone_seat_hidden_map,
    copy_action_sequence_state,
    god_search_rollout_rng_seed,
    god_search_rollout_sampling_algorithm,
)
from weiss_rl.eval.search.simulator_god_search_selection import (
    build_god_search_trace,
    mean_candidate_scores,
    root_logit_by_action,
    select_god_search_candidate,
)
from weiss_rl.eval.simulator.harness import ScheduledGame


class SimulatorGodSearchMixin:
    """Private god-search behavior shared by the simulator eval runner."""

    if TYPE_CHECKING:
        _eval_sampling_algorithm: str
        _god_search_config: GodSearchConfig
        _god_search_stats: GodSearchStats
        pass_action_id: int

        def _build_ids_eval_env(self, *, seed: int, scheduled_game: ScheduledGame | None = None) -> Any: ...

        def _legal_ids_for_env_row(self, *, batch: DecisionBoundaryBatch) -> np.ndarray: ...

        def _select_action_without_god_search(
            self,
            *,
            batch: DecisionBoundaryBatch,
            current_seat: int,
            current_policy_id: str,
            opponent_policy_id: str,
            seat_hidden: torch.Tensor | None,
            rng: Pcg32XshRrV1,
            legal_ids: np.ndarray,
            action_sequence_state: Any | None = None,
            sampling_algorithm: str | None = None,
        ) -> tuple[int, torch.Tensor | None]: ...

    def _should_run_god_search(
        self,
        *,
        policy: ResolvedEvalPolicy,
        current_policy_id: str,
        scheduled_game: ScheduledGame | None,
        legal_ids_for_model: np.ndarray,
        game_search_state: dict[str, int] | None,
    ) -> bool:
        config = self._god_search_config
        if not config.enabled:
            return False
        if policy.model is None:
            self._god_search_stats.skipped_no_model += 1
            return False
        if scheduled_game is None:
            return False
        if config.apply_to_focal_only and current_policy_id != scheduled_game.focal_policy_id:
            self._god_search_stats.skipped_non_focal += 1
            return False
        if np.asarray(legal_ids_for_model).size <= 1:
            self._god_search_stats.skipped_single_candidate += 1
            return False
        if game_search_state is not None and config.max_search_decisions_per_game > 0:
            if int(game_search_state.get("searched", 0)) >= int(config.max_search_decisions_per_game):
                return False
        return True

    def _select_action_with_god_search(
        self,
        *,
        scheduled_game: ScheduledGame | None,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        current_policy_id: str,
        opponent_policy_id: str,
        root_seat_hidden: torch.Tensor,
        root_next_seat_hidden: torch.Tensor | None,
        seat_hidden_by_seat: Mapping[int, torch.Tensor | None] | None,
        action_sequence_state: Any | None,
        action_history: Sequence[int],
        root_logits: np.ndarray,
        legal_ids: np.ndarray,
        legal_ids_for_model: np.ndarray,
        base_action: int,
        game_search_state: dict[str, int] | None,
    ) -> int:
        if scheduled_game is None or seat_hidden_by_seat is None:
            return int(base_action)
        config = self._god_search_config
        if config.mode != "same_world_prefix_rollout":
            raise ValueError(f"unsupported god-search mode: {config.mode!r}")
        candidates = top_k_legal_actions(root_logits, legal_ids_for_model, top_k=int(config.top_k))
        if len(candidates) <= 1:
            self._god_search_stats.skipped_single_candidate += 1
            return int(base_action)
        if game_search_state is not None:
            game_search_state["searched"] = int(game_search_state.get("searched", 0)) + 1
        self._god_search_stats.search_decisions += 1

        root_logits_by_action = root_logit_by_action(root_logits=root_logits, candidates=candidates)
        candidate_scores: dict[int, list[float]] = {int(action): [] for action in candidates}
        rollout_details: dict[int, list[dict[str, Any]]] = {int(action): [] for action in candidates}
        decision_id = int(np.asarray(batch.decision_id, dtype=np.int64)[0])
        for action in candidates:
            for rollout_index in range(int(config.rollouts_per_action)):
                score, detail = self._run_same_world_prefix_rollout(
                    scheduled_game=scheduled_game,
                    action_history=action_history,
                    candidate_action=int(action),
                    current_seat=current_seat,
                    current_policy_id=current_policy_id,
                    opponent_policy_id=opponent_policy_id,
                    root_seat_hidden=root_seat_hidden,
                    root_next_seat_hidden=root_next_seat_hidden,
                    seat_hidden_by_seat=seat_hidden_by_seat,
                    action_sequence_state=action_sequence_state,
                    root_legal_ids=legal_ids,
                    root_decision_id=decision_id,
                    rollout_index=rollout_index,
                )
                candidate_scores[int(action)].append(float(score))
                rollout_details[int(action)].append(detail)

        averaged = mean_candidate_scores(candidate_scores)
        selected_action = select_god_search_candidate(
            candidates=candidates,
            mean_scores=averaged,
            root_logits=root_logits_by_action,
        )
        if int(selected_action) != int(base_action):
            self._god_search_stats.changed_decisions += 1
        self._god_search_stats.add_trace(
            build_god_search_trace(
                scheduled_game=scheduled_game,
                decision_id=decision_id,
                current_seat=current_seat,
                current_policy_id=current_policy_id,
                opponent_policy_id=opponent_policy_id,
                base_action=base_action,
                selected_action=selected_action,
                candidates=candidates,
                mean_scores=averaged,
                root_logits=root_logits_by_action,
                rollout_details=rollout_details,
            )
        )
        return int(selected_action)

    def _run_same_world_prefix_rollout(
        self,
        *,
        scheduled_game: ScheduledGame,
        action_history: Sequence[int],
        candidate_action: int,
        current_seat: int,
        current_policy_id: str,
        opponent_policy_id: str,
        root_seat_hidden: torch.Tensor,
        root_next_seat_hidden: torch.Tensor | None,
        seat_hidden_by_seat: Mapping[int, torch.Tensor | None],
        action_sequence_state: Any | None,
        root_legal_ids: np.ndarray,
        root_decision_id: int,
        rollout_index: int,
    ) -> tuple[float, dict[str, Any]]:
        self._god_search_stats.candidate_evaluations += 1
        self._god_search_stats.rollout_games += 1
        env = self._build_ids_eval_env(seed=scheduled_game.episode_seed, scheduled_game=scheduled_game)
        rollout_decisions = 0
        last_acting_seat: int | None = None
        try:
            batch = env.reset(seed=scheduled_game.episode_seed)
            for prefix_action in action_history:
                last_acting_seat = int(batch.actor[0])
                batch = env.step(np.asarray([int(prefix_action)], dtype=np.uint32))
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    return self._prefix_replay_failed(
                        reason="prefix_terminated_before_root",
                        scheduled_game=scheduled_game,
                        root_decision_id=root_decision_id,
                    )

            if self._god_search_config.verify_prefix_replay:
                observed_decision_id = int(np.asarray(batch.decision_id, dtype=np.int64)[0])
                observed_legal_ids = self._legal_ids_for_env_row(batch=batch)
                if observed_decision_id != int(root_decision_id) or not np.array_equal(
                    np.asarray(observed_legal_ids, dtype=np.uint32),
                    np.asarray(root_legal_ids, dtype=np.uint32),
                ):
                    return self._prefix_replay_failed(
                        reason="prefix_root_mismatch",
                        scheduled_game=scheduled_game,
                        root_decision_id=root_decision_id,
                        observed_decision_id=observed_decision_id,
                        expected_legal_count=int(np.asarray(root_legal_ids).size),
                        observed_legal_count=int(np.asarray(observed_legal_ids).size),
                    )

            if int(candidate_action) not in set(int(action) for action in np.asarray(root_legal_ids).tolist()):
                return self._prefix_replay_failed(
                    reason="candidate_not_legal_at_root",
                    scheduled_game=scheduled_game,
                    root_decision_id=root_decision_id,
                    candidate_action=int(candidate_action),
                )

            rollout_hidden = clone_seat_hidden_map(
                seat_hidden_by_seat,
                current_seat=current_seat,
                root_next_seat_hidden=root_next_seat_hidden,
            )
            rollout_sequence_state = copy_action_sequence_state(action_sequence_state)
            update_eval_action_counters(
                counters=ActionSummaryCounters(),
                state=rollout_sequence_state,
                action=int(candidate_action),
                legal_ids=np.asarray(root_legal_ids, dtype=np.uint32),
                pass_action_id=self.pass_action_id,
            )
            last_acting_seat = int(current_seat)
            batch = env.step(np.asarray([int(candidate_action)], dtype=np.uint32))
            if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                return self._score_rollout_terminal(
                    batch=batch,
                    scheduled_game=scheduled_game,
                    last_acting_seat=last_acting_seat,
                    rollout_decisions=rollout_decisions,
                    cutoff=False,
                )

            max_rollout_decisions = int(self._god_search_config.max_rollout_decisions)
            while True:
                if max_rollout_decisions > 0 and rollout_decisions >= max_rollout_decisions:
                    self._god_search_stats.horizon_cutoffs += 1
                    return 0.0, {
                        "score": 0.0,
                        "status": "horizon_cutoff",
                        "rollout_decisions": int(rollout_decisions),
                    }
                rollout_decisions += 1
                branch_seat = int(batch.actor[0])
                branch_policy_id = (
                    scheduled_game.seat0_policy_id if branch_seat == 0 else scheduled_game.seat1_policy_id
                )
                branch_opponent_id = (
                    scheduled_game.seat1_policy_id if branch_seat == 0 else scheduled_game.seat0_policy_id
                )
                branch_legal_ids = self._legal_ids_for_env_row(batch=batch)
                branch_rng = Pcg32XshRrV1(
                    god_search_rollout_rng_seed(
                        scheduled_game=scheduled_game,
                        seat=branch_seat,
                        candidate_action=int(candidate_action),
                        rollout_index=int(rollout_index),
                        decision_id=int(np.asarray(batch.decision_id, dtype=np.int64)[0]),
                    )
                )
                branch_action, branch_next_hidden = self._select_action_without_god_search(
                    batch=batch,
                    current_seat=branch_seat,
                    current_policy_id=branch_policy_id,
                    opponent_policy_id=branch_opponent_id,
                    seat_hidden=rollout_hidden.get(branch_seat),
                    rng=branch_rng,
                    legal_ids=branch_legal_ids,
                    action_sequence_state=rollout_sequence_state,
                    sampling_algorithm=god_search_rollout_sampling_algorithm(
                        rollout_policy=self._god_search_config.rollout_policy,
                        eval_sampling_algorithm=self._eval_sampling_algorithm,
                    ),
                )
                update_eval_action_counters(
                    counters=ActionSummaryCounters(),
                    state=rollout_sequence_state,
                    action=int(branch_action),
                    legal_ids=branch_legal_ids,
                    pass_action_id=self.pass_action_id,
                )
                last_acting_seat = branch_seat
                batch = env.step(np.asarray([int(branch_action)], dtype=np.uint32))
                rollout_hidden[branch_seat] = branch_next_hidden
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    return self._score_rollout_terminal(
                        batch=batch,
                        scheduled_game=scheduled_game,
                        last_acting_seat=last_acting_seat,
                        rollout_decisions=rollout_decisions,
                        cutoff=False,
                    )
        finally:
            env.close()

    def _prefix_replay_failed(
        self,
        *,
        reason: str,
        scheduled_game: ScheduledGame,
        root_decision_id: int,
        **extra: Any,
    ) -> tuple[float, dict[str, Any]]:
        self._god_search_stats.prefix_replay_failures += 1
        detail = build_prefix_replay_failure_detail(
            reason=reason,
            scheduled_game=scheduled_game,
            root_decision_id=root_decision_id,
            extra=extra,
        )
        if self._god_search_config.fail_on_prefix_mismatch:
            raise RuntimeError(f"god-search prefix replay failed: {json.dumps(detail, sort_keys=True)}")
        return 0.0, detail

    def _score_rollout_terminal(
        self,
        *,
        batch: DecisionBoundaryBatch,
        scheduled_game: ScheduledGame,
        last_acting_seat: int | None,
        rollout_decisions: int,
        cutoff: bool,
    ) -> tuple[float, dict[str, Any]]:
        scored = score_terminal_rollout(
            batch=batch,
            scheduled_game=scheduled_game,
            last_acting_seat=last_acting_seat,
            rollout_decisions=rollout_decisions,
            cutoff=cutoff,
        )
        if scored.stat_bucket == "truncated":
            self._god_search_stats.truncated_rollouts += 1
            return scored.score, scored.detail
        self._god_search_stats.terminal_rollouts += 1
        return scored.score, scored.detail

    def god_search_diagnostics(self) -> dict[str, Any] | None:
        if not self._god_search_config.enabled:
            return None
        return self._god_search_stats.to_json_dict(config=self._god_search_config)
