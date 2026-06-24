from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from weiss_rl.config import load_stack_config
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.simulator.harness import GameResult, ReplaySampleResult, ScheduledGame

from ._config_paths import canonical_stack_config_path

_RUN_ID256 = "ab" * 32
_CONFIG_HASH256 = "cd" * 32
_SPEC_HASH256 = "ef" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


class _FakeMatrixRunner:
    def __init__(self, outcomes: dict[tuple[str, str, int, int], OutcomeToken]) -> None:
        self._outcomes = outcomes
        self.calls: list[ScheduledGame] = []

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        self.calls.append(scheduled_game)
        key = (
            scheduled_game.focal_policy_id,
            scheduled_game.opponent_policy_id,
            scheduled_game.pair_index,
            scheduled_game.swap_index,
        )
        outcome = self._outcomes[key]
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


class _AlwaysWinRunner:
    def __init__(self) -> None:
        self.calls: list[ScheduledGame] = []

    def run_game(self, scheduled: ScheduledGame) -> GameResult:
        replay_sample = None
        if not self.calls:
            replay_sample = ReplaySampleResult(
                pair_index=scheduled.pair_index,
                swap_index=scheduled.swap_index,
                episode_index=scheduled.episode_index,
                focal_policy_id=scheduled.focal_policy_id,
                opponent_policy_id=scheduled.opponent_policy_id,
                raw_replay_path="raw/replay.json",
                bundle_path="bundles/replay.zip",
                verification_report_path="verification/replay.json",
                verification_status="success",
                replay_key64="abcd1234",
                matched=True,
            )
        self.calls.append(scheduled)
        return GameResult(
            episode_seed=scheduled.episode_seed,
            terminated=True,
            truncated=False,
            winner_seat=scheduled.focal_seat,
            replay_sample=replay_sample,
        )

    def god_search_diagnostics(self) -> dict[str, Any]:
        return {"enabled": True, "checked": len(self.calls)}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _selection_config(**overrides: Any):
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.evaluation is not None
    return replace(stack.config.evaluation.final_policy_set_selection, **overrides)


def _final_eval_stop_rules() -> StopRulesConfig:
    return StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95)


def _write_policy_set_inputs(tmp_path: Path) -> tuple[Path, Path]:
    snapshot_registry_path = tmp_path / "snapshot_registry.json"
    dev_eval_summaries_path = tmp_path / "dev_eval_summaries.json"
    snapshot_registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "snapshots": [
                    {
                        "policy_id": "policy_000100",
                        "update": 100,
                        "weights_sha256": "1" * 64,
                        "path": "training/snapshots/policy_000100/weights.pt",
                        "created_utc": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "policy_id": "policy_000200",
                        "update": 200,
                        "weights_sha256": "2" * 64,
                        "path": "training/snapshots/policy_000200/weights.pt",
                        "created_utc": "2026-01-01T00:00:01+00:00",
                    },
                    {
                        "policy_id": "policy_000300",
                        "update": 300,
                        "weights_sha256": "3" * 64,
                        "path": "training/snapshots/policy_000300/weights.pt",
                        "created_utc": "2026-01-01T00:00:02+00:00",
                    },
                ],
                "champion_snapshots": ["policy_000300"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    dev_eval_summaries_path.write_text(
        json.dumps(
            {
                "policy_000150": {
                    "aggregate_score": 0.95,
                    "anchor_scores": {
                        "B0 RandomLegal": 0.8,
                        "B1 NoLeague baseline": 0.8,
                    },
                },
                "policy_000250": {
                    "aggregate_score": 0.85,
                    "anchor_scores": {
                        "B0 RandomLegal": 0.7,
                        "B1 NoLeague baseline": 0.7,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot_registry_path, dev_eval_summaries_path
