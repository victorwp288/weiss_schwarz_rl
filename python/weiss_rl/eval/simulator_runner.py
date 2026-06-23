"""Simulator-backed deterministic evaluation runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.config import StackConfig
from weiss_rl.core.masking import assert_strictly_increasing_legal_ids
from weiss_rl.diagnostics.action_diagnostics import (
    ActionSummaryCounters,
    make_action_sequence_state,
    summarize_eval_action_counters,
    update_eval_action_counters,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval.god_search import GodSearchConfig, GodSearchStats
from weiss_rl.eval.harness import (
    EvalGameRunner,
    GameResult,
    ScheduledGame,
    abort_on_engine_fault_eval,
    game_result_from_step,
    sample_action_pinned,
    select_action_argmax_pinned,
)
from weiss_rl.eval.model_action_surface import (
    ModelActionSurfaceSettings,
    model_action_surface_batch_and_ids,
)
from weiss_rl.eval.model_sampling import model_eval_logits_for_legal_ids
from weiss_rl.eval.policies.resolution import (
    ResolvedEvalPolicy,
    resolve_eval_policies,
)
from weiss_rl.eval.rng_pcg32 import Pcg32XshRrV1
from weiss_rl.eval.simulator_god_search import SimulatorGodSearchMixin
from weiss_rl.eval.simulator_replay import SimulatorReplayRecorder
from weiss_rl.model import PolicyValueModel
from weiss_rl.models.loading import (
    restore_model_guidance_from_payload as _shared_restore_model_guidance_from_payload,
)
from weiss_rl.runtime.components.opponent_context import (
    eval_policy_uses_opponent_context,
    initial_seat_hidden_for_opponents,
)

__all__ = [
    "ResolvedEvalPolicy",
    "SimulatorEvalRunner",
    "resolve_eval_policies",
]


def _restore_model_guidance_from_payload(model: PolicyValueModel | None, payload: Mapping[str, object]) -> None:
    _shared_restore_model_guidance_from_payload(model, payload)


class SimulatorEvalRunner(SimulatorGodSearchMixin, EvalGameRunner):
    def __init__(
        self,
        *,
        stack: StackConfig,
        policies: Mapping[str, ResolvedEvalPolicy],
        artifact_layout: ArtifactLayout,
        run_id256: str,
        spec_hash256: str,
        action_dim: int,
        pass_action_id: int,
        require_sorted_legal_ids: bool,
        replay_capture_rate: float,
        regression_capture_count: int,
        god_search_config: GodSearchConfig | None = None,
    ) -> None:
        self.stack = stack
        self.policies = dict(policies)
        self.artifact_layout = artifact_layout
        self.run_id256_bytes = bytes.fromhex(run_id256)
        self.spec_hash256_bytes = bytes.fromhex(spec_hash256)
        self.action_dim = int(action_dim)
        self.pass_action_id = int(pass_action_id)
        self.require_sorted_legal_ids = bool(require_sorted_legal_ids)
        self._replay_recorder = SimulatorReplayRecorder(
            stack=self.stack,
            artifact_layout=self.artifact_layout,
            run_id256_bytes=self.run_id256_bytes,
            spec_hash256_bytes=self.spec_hash256_bytes,
            capture_rate=replay_capture_rate,
            capture_limit=regression_capture_count,
        )
        self._god_search_config = god_search_config or GodSearchConfig()
        self._god_search_stats = GodSearchStats(trace_limit=int(self._god_search_config.trace_limit))
        evaluation_config = getattr(self.stack.config, "evaluation", None)
        self._eval_sampling_algorithm = str(
            getattr(evaluation_config, "eval_sampling_algorithm", "pinned_cdf_pcg_v1") or "pinned_cdf_pcg_v1"
        ).strip()
        self._model_sampling_temperature = float(getattr(evaluation_config, "model_sampling_temperature", 1.0) or 1.0)
        if self._eval_sampling_algorithm not in {"pinned_cdf_pcg_v1", "model_argmax_pinned_v1"}:
            raise ValueError(
                "evaluation.eval_sampling_algorithm must be 'pinned_cdf_pcg_v1' or "
                f"'model_argmax_pinned_v1', got {self._eval_sampling_algorithm!r}"
            )
        requested_device = str(getattr(evaluation_config, "eval_device", "cpu") or "cpu").strip().lower()
        if requested_device in {"", "auto", "cuda:auto"}:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self._device = torch.device(requested_device)
        for policy in self.policies.values():
            if policy.model is not None:
                if hasattr(policy.model, "to"):
                    policy.model.to(self._device)
                if hasattr(policy.model, "eval"):
                    policy.model.eval()
        self._baseline_logits = np.zeros((self.action_dim,), dtype=np.float32)
        training_config = getattr(getattr(self.stack, "config", None), "training", None)
        self._model_action_surface = ModelActionSurfaceSettings.from_training_config(
            training_config,
            pass_action_id=self.pass_action_id,
        )

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        env = self._build_ids_eval_env(seed=scheduled_game.episode_seed, scheduled_game=scheduled_game)
        replay_capture = self._replay_recorder.maybe_start(env=env, scheduled_game=scheduled_game)
        seat_hidden = {
            seat: self._initial_hidden(
                scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
                opponent_policy_id=scheduled_game.seat1_policy_id if seat == 0 else scheduled_game.seat0_policy_id,
            )
            for seat in (0, 1)
        }
        seat_rngs = {seat: Pcg32XshRrV1(self._rng_seed(scheduled_game=scheduled_game, seat=seat)) for seat in (0, 1)}
        action_counters = ActionSummaryCounters()
        action_sequence_state = make_action_sequence_state(1)
        action_history: list[int] = []
        game_search_state = {"searched": 0}
        last_acting_seat: int | None = None

        try:
            batch = env.reset(seed=scheduled_game.episode_seed)
            self._abort_on_fault(batch=batch, scheduled_game=scheduled_game)
            if replay_capture is not None:
                self._replay_recorder.record_initial_batch(replay_capture, batch=batch)
            while True:
                if bool(batch.terminated[0]) or bool(batch.truncated[0]):
                    result = game_result_from_step(
                        batch,
                        env_index=0,
                        acting_seat=last_acting_seat,
                        episode_seed=scheduled_game.episode_seed,
                        max_decisions=getattr(env, "max_decisions", None),
                        max_ticks=getattr(env, "max_ticks", None),
                        max_no_progress_decisions=getattr(env, "max_no_progress_decisions", None),
                    )
                    action_summary = summarize_eval_action_counters(action_counters)
                    return self._finalize_game_result(
                        result=result,
                        action_summary=action_summary,
                        scheduled_game=scheduled_game,
                        replay_capture=replay_capture,
                    )

                current_seat = int(batch.actor[0])
                current_policy_id = (
                    scheduled_game.seat0_policy_id if current_seat == 0 else scheduled_game.seat1_policy_id
                )
                legal_ids = self._legal_ids_for_env_row(batch=batch)
                action, next_hidden = self._select_action(
                    batch=batch,
                    current_seat=current_seat,
                    current_policy_id=current_policy_id,
                    opponent_policy_id=(
                        scheduled_game.seat1_policy_id if current_seat == 0 else scheduled_game.seat0_policy_id
                    ),
                    seat_hidden=seat_hidden[current_seat],
                    rng=seat_rngs[current_seat],
                    legal_ids=legal_ids,
                    action_sequence_state=action_sequence_state,
                    scheduled_game=scheduled_game,
                    action_history=action_history,
                    seat_hidden_by_seat=seat_hidden,
                    game_search_state=game_search_state,
                )
                update_eval_action_counters(
                    counters=action_counters,
                    state=action_sequence_state,
                    action=int(action),
                    legal_ids=legal_ids,
                    pass_action_id=self.pass_action_id,
                )
                decision_id = int(np.asarray(batch.decision_id, dtype=np.int64)[0])
                last_acting_seat = current_seat
                next_batch = env.step(np.asarray([action], dtype=np.uint32))
                self._abort_on_fault(batch=next_batch, scheduled_game=scheduled_game)
                if replay_capture is not None:
                    self._replay_recorder.record_step(
                        replay_capture,
                        decision_id=decision_id,
                        actor=current_seat,
                        action=int(action),
                        next_batch=next_batch,
                        legal_ids=legal_ids,
                    )
                seat_hidden[current_seat] = next_hidden
                action_history.append(int(action))
                batch = next_batch
        finally:
            env.close()

    def _finalize_game_result(
        self,
        *,
        result: GameResult,
        action_summary: Mapping[str, int],
        scheduled_game: ScheduledGame,
        replay_capture: Any | None,
    ) -> GameResult:
        replay_sample = (
            None
            if replay_capture is None
            else self._replay_recorder.finish(
                scheduled_game=scheduled_game,
                capture=replay_capture,
            )
        )
        return GameResult(
            episode_seed=result.episode_seed,
            terminated=result.terminated,
            truncated=result.truncated,
            winner_seat=result.winner_seat,
            engine_status=result.engine_status,
            decision_count=result.decision_count,
            tick_count=result.tick_count,
            no_progress_count=result.no_progress_count,
            termination_reason=result.termination_reason,
            simulator_episode_key=result.simulator_episode_key,
            total_actions=action_summary["total_actions"],
            pass_actions=action_summary["pass_actions"],
            main_move_actions=action_summary["main_move_actions"],
            pass_with_nonpass_available=action_summary["pass_with_nonpass_available"],
            max_consecutive_main_moves=action_summary["max_consecutive_main_moves"],
            replay_sample=replay_sample,
        )

    def _build_ids_eval_env(self, *, seed: int, scheduled_game: ScheduledGame | None = None) -> DecisionBoundaryEnv:
        env_config = build_env_config_from_stack(
            self.stack,
            seed=int(seed),
            deck=None if scheduled_game is None else scheduled_game.seat0_deck,
            opponent_deck=None if scheduled_game is None else scheduled_game.seat1_deck,
        )
        pool, layout_name = make_env_pool_from_config(
            env_config,
            profile="fast",
            num_envs=1,
        )
        if layout_name != "i16_legal_ids":
            raise RuntimeError(
                f"Pinned evaluation requires ids-based legality for deterministic CPU sampling, got {layout_name!r}."
            )
        max_no_progress_decisions = None
        curriculum = self.stack.config.curriculum
        if curriculum is not None:
            raw_limit = curriculum.simulator.get("max_no_progress_decisions")
            if raw_limit is not None:
                max_no_progress_decisions = int(raw_limit)
        return DecisionBoundaryEnv(
            pool,
            legality="ids_offsets",
            pass_action_id=self.pass_action_id,
            engine_status_policy="hard_fail",
            max_decisions=int(env_config["max_decisions"]),
            max_ticks=int(env_config["max_ticks"]),
            max_no_progress_decisions=max_no_progress_decisions,
        )

    def _select_action(
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
        scheduled_game: ScheduledGame | None = None,
        action_history: Sequence[int] = (),
        seat_hidden_by_seat: Mapping[int, torch.Tensor | None] | None = None,
        game_search_state: dict[str, int] | None = None,
    ) -> tuple[int, torch.Tensor | None]:
        policy = self.policies.get(current_policy_id)
        if policy is None:
            raise RuntimeError(f"Missing resolved eval policy for {current_policy_id!r}")
        if policy.heuristic_policy is not None:
            action = policy.heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), seat_hidden
        if policy.model is None:
            action, _logp = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=rng,
            )
            return action, seat_hidden
        if seat_hidden is None:
            raise RuntimeError(f"Missing hidden state for eval policy {current_policy_id!r}")
        logits, next_seat_hidden, legal_ids_for_model = self._model_logits_for_eval(
            policy=policy,
            current_policy_id=current_policy_id,
            opponent_policy_id=opponent_policy_id,
            batch=batch,
            current_seat=current_seat,
            seat_hidden=seat_hidden,
            legal_ids=legal_ids,
            action_sequence_state=action_sequence_state,
        )
        action, _logp = self._select_model_action_from_logits(
            logits=logits,
            legal_ids=legal_ids_for_model,
            rng=rng,
            sampling_algorithm=self._eval_sampling_algorithm,
        )
        if self._should_run_god_search(
            policy=policy,
            current_policy_id=current_policy_id,
            scheduled_game=scheduled_game,
            legal_ids_for_model=legal_ids_for_model,
            game_search_state=game_search_state,
        ):
            action = self._select_action_with_god_search(
                scheduled_game=scheduled_game,
                batch=batch,
                current_seat=current_seat,
                current_policy_id=current_policy_id,
                opponent_policy_id=opponent_policy_id,
                root_seat_hidden=seat_hidden,
                root_next_seat_hidden=next_seat_hidden,
                seat_hidden_by_seat=seat_hidden_by_seat,
                action_sequence_state=action_sequence_state,
                action_history=action_history,
                root_logits=logits,
                legal_ids=legal_ids,
                legal_ids_for_model=legal_ids_for_model,
                base_action=action,
                game_search_state=game_search_state,
            )
        return action, next_seat_hidden

    def _model_logits_for_eval(
        self,
        *,
        policy: ResolvedEvalPolicy,
        current_policy_id: str,
        opponent_policy_id: str,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        seat_hidden: torch.Tensor,
        legal_ids: np.ndarray,
        action_sequence_state: Any | None = None,
    ) -> tuple[np.ndarray, torch.Tensor | None, np.ndarray]:
        batch_for_model, legal_ids_for_model = model_action_surface_batch_and_ids(
            model=policy.model,
            batch=batch,
            legal_ids=legal_ids,
            settings=self._model_action_surface,
            action_sequence_state=action_sequence_state,
        )
        if policy.model is None:
            raise RuntimeError(f"Missing model for eval policy {current_policy_id!r}")
        with torch.inference_mode():
            logits, next_seat_hidden = model_eval_logits_for_legal_ids(
                model=policy.model,
                batch=batch_for_model,
                current_seat=int(current_seat),
                seat_hidden=seat_hidden,
                legal_ids=legal_ids_for_model,
                action_dim=int(self.action_dim),
                device=self._device,
                opponent_context_index=self._opponent_context_index_for_eval(
                    policy=policy,
                    policy_id=current_policy_id,
                    opponent_policy_id=opponent_policy_id,
                ),
            )
        return logits, next_seat_hidden, legal_ids_for_model

    def _select_model_action_from_logits(
        self,
        *,
        logits: np.ndarray,
        legal_ids: np.ndarray,
        rng: Pcg32XshRrV1,
        sampling_algorithm: str,
    ) -> tuple[int, np.float32]:
        if sampling_algorithm == "model_argmax_pinned_v1":
            return select_action_argmax_pinned(
                logits,
                legal_ids,
                pass_action_id=self.pass_action_id,
            )
        if sampling_algorithm == "pinned_cdf_pcg_v1":
            return sample_action_pinned(
                logits,
                legal_ids,
                rng=rng,
                pass_action_id=self.pass_action_id,
                temperature=self._model_sampling_temperature,
            )
        raise ValueError(f"unsupported eval sampling algorithm: {sampling_algorithm!r}")

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
    ) -> tuple[int, torch.Tensor | None]:
        policy = self.policies.get(current_policy_id)
        if policy is None:
            raise RuntimeError(f"Missing resolved eval policy for {current_policy_id!r}")
        if policy.heuristic_policy is not None:
            action = policy.heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return int(action), seat_hidden
        if policy.model is None:
            action, _logp = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=rng,
            )
            return action, seat_hidden
        if seat_hidden is None:
            raise RuntimeError(f"Missing hidden state for eval policy {current_policy_id!r}")
        logits, next_seat_hidden, legal_ids_for_model = self._model_logits_for_eval(
            policy=policy,
            current_policy_id=current_policy_id,
            opponent_policy_id=opponent_policy_id,
            batch=batch,
            current_seat=current_seat,
            seat_hidden=seat_hidden,
            legal_ids=legal_ids,
            action_sequence_state=action_sequence_state,
        )
        action, _logp = self._select_model_action_from_logits(
            logits=logits,
            legal_ids=legal_ids_for_model,
            rng=rng,
            sampling_algorithm=sampling_algorithm or self._eval_sampling_algorithm,
        )
        return action, next_seat_hidden

    def _initial_hidden(self, policy_id: str, *, opponent_policy_id: str | None = None) -> torch.Tensor | None:
        policy = self.policies.get(policy_id)
        if policy is None or policy.model is None:
            return None
        if opponent_policy_id is not None and eval_policy_uses_opponent_context(policy.model, policy_id):
            return initial_seat_hidden_for_opponents(
                policy.model,
                1,
                device=self._device,
                opponent_policy_ids=[opponent_policy_id],
            )
        return initial_seat_hidden_for_opponents(policy.model, 1, device=self._device)

    def _opponent_context_index_for_eval(
        self,
        *,
        policy: ResolvedEvalPolicy,
        policy_id: str,
        opponent_policy_id: str,
    ) -> int | None:
        if policy.model is None or not eval_policy_uses_opponent_context(policy.model, policy_id):
            return None
        index_fn = getattr(policy.model, "opponent_context_indices_for_policy_ids", None)
        if not callable(index_fn):
            return None
        indices = index_fn([opponent_policy_id], batch_size=1)
        if not indices:
            return None
        return int(indices[0])

    def _abort_on_fault(self, *, batch: DecisionBoundaryBatch, scheduled_game: ScheduledGame) -> None:
        matchup_dir = (
            self.artifact_layout.final_eval_matchups_dir
            / f"{scheduled_game.pair_index:04d}_{scheduled_game.swap_index:01d}_{scheduled_game.episode_seed:016x}"
        )
        abort_on_engine_fault_eval(
            run_dir=matchup_dir,
            engine_status=batch.engine_status,
            decision_id=batch.decision_id,
            episode_key=batch.episode_key,
            note="engine_status!=0 during canonical final eval",
        )

    def _legal_ids_for_env_row(self, *, batch: DecisionBoundaryBatch) -> np.ndarray:
        if batch.ids_offsets is None:
            raise RuntimeError("Pinned evaluation requires ids_offsets legality")
        legal_ids, legal_offsets = batch.ids_offsets
        row = np.asarray(legal_ids[int(legal_offsets[0]) : int(legal_offsets[1])], dtype=np.uint32)
        if self.require_sorted_legal_ids:
            assert_strictly_increasing_legal_ids(row)
        return row

    def _rng_seed(self, *, scheduled_game: ScheduledGame, seat: int) -> int:
        payload = canonical_json_bytes(
            {
                "kind": "simulator_eval_rng_v1",
                "pair_index": int(scheduled_game.pair_index),
                "swap_index": int(scheduled_game.swap_index),
                "episode_seed": int(scheduled_game.episode_seed),
                "seat": int(seat),
                "seat_policy_id": (scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id),
            }
        )
        return stable_hash64(payload)
