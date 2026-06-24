from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

from weiss_rl.eval.simulator.harness import GameResult, MatchupSummary, ScheduledGame
from weiss_rl.league import PromotionGateAnchorResult, PromotionGatePosterior, PromotionGateRate

RUN_ID256 = "ab" * 32
CONFIG_HASH256 = "cd" * 32
SPEC_HASH256 = "ef" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class OutcomeRunner:
    def __init__(self, outcome_for_game: Callable[[ScheduledGame], OutcomeToken]) -> None:
        self._outcome_for_game = outcome_for_game

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        return game_result(scheduled_game, self._outcome_for_game(scheduled_game))


def game_result(scheduled_game: ScheduledGame, outcome: OutcomeToken) -> GameResult:
    if outcome == "W":
        return GameResult(
            episode_seed=scheduled_game.episode_seed,
            terminated=True,
            truncated=False,
            winner_seat=scheduled_game.focal_seat,
        )
    if outcome == "L":
        return GameResult(
            episode_seed=scheduled_game.episode_seed,
            terminated=True,
            truncated=False,
            winner_seat=1 - scheduled_game.focal_seat,
        )
    if outcome == "D":
        return GameResult(
            episode_seed=scheduled_game.episode_seed,
            terminated=True,
            truncated=False,
            winner_seat=None,
        )
    return GameResult(
        episode_seed=scheduled_game.episode_seed,
        terminated=False,
        truncated=True,
        winner_seat=None,
    )


def stack_with_promotion_gate_override(stack, **overrides):
    assert stack.config.league is not None
    promotion_gate = replace(stack.config.league.promotion.gate, **overrides)
    promotion = replace(stack.config.league.promotion, gate=promotion_gate)
    league = replace(stack.config.league, promotion=promotion)
    return replace(stack, config=replace(stack.config, league=league))


def posterior(*, prob_gt_target: float, prob_lt_guardrail: float = 0.0) -> PromotionGatePosterior:
    return PromotionGatePosterior(
        mean=0.6,
        ci_low=0.55,
        ci_high=0.65,
        ci_half_width=0.05,
        prob_gt_half=1.0,
        prob_lt_half=0.0,
        prob_gt_target=prob_gt_target,
        prob_lt_guardrail=prob_lt_guardrail,
        paired_seed_count=10,
        sample_count=128,
    )


def anchor_result(anchor_name: str, *, prob_lt_guardrail: float) -> PromotionGateAnchorResult:
    return PromotionGateAnchorResult(
        anchor_name=anchor_name,
        opponent_policy_id=f"{anchor_name}_policy",
        episodes_path="promotion_gate_episodes/test.jsonl",
        matchup_summary=MatchupSummary(games=20),
        truncation=PromotionGateRate(numerator=0, denominator=20, rate=0.0),
        posterior=posterior(prob_gt_target=1.0, prob_lt_guardrail=prob_lt_guardrail),
    )
