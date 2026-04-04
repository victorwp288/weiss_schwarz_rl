from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import pytest

from weiss_rl.config import load_stack_config
from weiss_rl.eval.harness import GameResult, ScheduledGame
from weiss_rl.league import run_promotion_gate
from weiss_rl.repro import hash_seed_file

_RUN_ID256 = "ab" * 32
_CONFIG_HASH256 = "cd" * 32
_SPEC_HASH256 = "ef" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _OutcomeRunner:
    def __init__(self, outcome_for_game: Callable[[ScheduledGame], OutcomeToken]) -> None:
        self._outcome_for_game = outcome_for_game

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        return _game_result(scheduled_game, self._outcome_for_game(scheduled_game))


def _game_result(scheduled_game: ScheduledGame, outcome: OutcomeToken) -> GameResult:
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


def test_run_promotion_gate_writes_pass_record_in_anchor_order(tmp_path: Path) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/rl_stack_locked.yaml")

    result = run_promotion_gate(
        stack=stack,
        run_dir=tmp_path,
        focal_policy_id="candidate_snapshot",
        anchor_policy_ids={
            "B0 RandomLegal": "random_legal",
            "B1 NoLeague baseline": "b1_snapshot",
            "B2 HeuristicPublic": "heuristic_public",
        },
        runner=_OutcomeRunner(lambda _: "W"),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        sample_count=64,
        bootstrap_seed=7,
    )

    assert result.passed is True
    assert result.ordered_opponents == (
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
    )
    assert result.seed_file_sha256 == hash_seed_file(repo_root / "configs/seeds/promotion_eval_seeds.txt")
    assert result.overall_posterior.prob_gt_target == 1.0
    assert result.truncation.rate == 0.0

    record_path = tmp_path / "promotion_gate.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["decision"] == {"passed": True, "reasons": []}
    assert payload["ordered_opponents"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
    ]
    assert payload["seed_file"] == {
        "path": "configs/seeds/promotion_eval_seeds.txt",
        "sha256": result.seed_file_sha256,
    }
    assert [anchor["episodes_path"] for anchor in payload["anchors"]] == [
        "promotion_gate_episodes/00_b0_randomlegal.jsonl",
        "promotion_gate_episodes/01_b1_noleague_baseline.jsonl",
        "promotion_gate_episodes/02_b2_heuristicpublic.jsonl",
    ]


def test_run_promotion_gate_records_machine_readable_failure_reasons(tmp_path: Path) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/rl_stack_locked.yaml")

    def outcome_for_game(game: ScheduledGame) -> OutcomeToken:
        return "L" if game.opponent_policy_id == "b1_snapshot" else "W"

    result = run_promotion_gate(
        stack=stack,
        run_dir=tmp_path,
        focal_policy_id="candidate_snapshot",
        anchor_policy_ids={
            "B0 RandomLegal": "random_legal",
            "B1 NoLeague baseline": "b1_snapshot",
        },
        runner=_OutcomeRunner(outcome_for_game),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        sample_count=128,
        bootstrap_seed=11,
    )

    assert result.passed is False
    assert {reason["code"] for reason in result.reasons} == {
        "anchor_loss_guardrail_exceeded",
        "overall_posterior_below_threshold",
    }
    assert any(
        reason["code"] == "anchor_loss_guardrail_exceeded" and reason["anchor_name"] == "B1 NoLeague baseline"
        for reason in result.reasons
    )
    assert result.truncation.rate == 0.0

    payload = json.loads((tmp_path / "promotion_gate.json").read_text(encoding="utf-8"))
    assert payload["decision"]["passed"] is False
    assert {reason["code"] for reason in payload["decision"]["reasons"]} == {
        "anchor_loss_guardrail_exceeded",
        "overall_posterior_below_threshold",
    }


def test_run_promotion_gate_rejects_missing_required_anchor(tmp_path: Path) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/rl_stack_locked.yaml")

    with pytest.raises(ValueError, match="missing required promotion anchors: B1 NoLeague baseline"):
        run_promotion_gate(
            stack=stack,
            run_dir=tmp_path,
            focal_policy_id="candidate_snapshot",
            anchor_policy_ids={"B0 RandomLegal": "random_legal"},
            runner=_OutcomeRunner(lambda _: "W"),
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
            sample_count=32,
            bootstrap_seed=5,
        )
