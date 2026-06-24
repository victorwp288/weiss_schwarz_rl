"""Simulator-backed deterministic evaluation runner."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.diagnostics.probes.action_diagnostics import (
    ActionSummaryCounters,
    make_action_sequence_state,
    summarize_eval_action_counters,
    update_eval_action_counters,
)
from weiss_rl.eval.policies.resolution import (
    ResolvedEvalPolicy,
    resolve_eval_policies,
)
from weiss_rl.eval.replay.simulator_replay import SimulatorReplayRecorder
from weiss_rl.eval.sampling.model_action_surface import (
    ModelActionSurfaceSettings,
)
from weiss_rl.eval.sampling.rng_pcg32 import Pcg32XshRrV1
from weiss_rl.eval.search.god_search import GodSearchConfig, GodSearchStats
from weiss_rl.eval.search.simulator_god_search import SimulatorGodSearchMixin
from weiss_rl.eval.simulator.harness import (
    EvalGameRunner,
    GameResult,
    ScheduledGame,
    game_result_from_step,
)
from weiss_rl.eval.simulator.simulator_action_selection import SimulatorActionSelectionMixin
from weiss_rl.eval.simulator.simulator_game_lifecycle import SimulatorGameLifecycleMixin
from weiss_rl.model import PolicyValueModel
from weiss_rl.models.policy.loading import (
    restore_model_guidance_from_payload as _shared_restore_model_guidance_from_payload,
)

__all__ = [
    "ResolvedEvalPolicy",
    "SimulatorEvalRunner",
    "resolve_eval_policies",
]


def _restore_model_guidance_from_payload(model: PolicyValueModel | None, payload: Mapping[str, object]) -> None:
    _shared_restore_model_guidance_from_payload(model, payload)


class SimulatorEvalRunner(
    SimulatorActionSelectionMixin,
    SimulatorGodSearchMixin,
    SimulatorGameLifecycleMixin,
    EvalGameRunner,
):
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

