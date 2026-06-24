"""Simulator eval game setup, teardown, and deterministic seed helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.env_config import build_env_config_from_stack
from weiss_rl.envs.pool_factory import make_env_pool_from_config
from weiss_rl.eval.simulator.harness import GameResult, ScheduledGame, abort_on_engine_fault_eval


class SimulatorGameLifecycleMixin:
    """Own simulator env setup, fault handling, replay finalization, and RNG seeds."""

    def _finalize_game_result(
        self: Any,
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

    def _build_ids_eval_env(
        self: Any, *, seed: int, scheduled_game: ScheduledGame | None = None
    ) -> DecisionBoundaryEnv:
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

    def _abort_on_fault(self: Any, *, batch: DecisionBoundaryBatch, scheduled_game: ScheduledGame) -> None:
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

    def _rng_seed(self, *, scheduled_game: ScheduledGame, seat: int) -> int:
        payload = canonical_json_bytes(
            {
                "kind": "simulator_eval_rng_v1",
                "pair_index": int(scheduled_game.pair_index),
                "swap_index": int(scheduled_game.swap_index),
                "episode_seed": int(scheduled_game.episode_seed),
                "seat": int(seat),
                "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
            }
        )
        return stable_hash64(payload)


__all__ = ["SimulatorGameLifecycleMixin"]
