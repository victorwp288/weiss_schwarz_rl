from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from weiss_rl.config import load_stack_config
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import resolve_final_policy_set, run_final_eval
from weiss_rl.eval.harness import GameResult, ScheduledGame

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _selection_config(**overrides: Any):
    stack = load_stack_config(_repo_root() / "configs/rl_stack_locked.yaml")
    assert stack.config.evaluation is not None
    return replace(stack.config.evaluation.final_policy_set_selection, **overrides)


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


def test_resolve_final_policy_set_uses_deterministic_order_from_artifacts(tmp_path: Path) -> None:
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    config = _selection_config(
        include_random_legal_baseline_b0=True,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=True,
        include_spaced_snapshots_near_percent_updates=(50,),
    )

    selected = resolve_final_policy_set(
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        config=config,
        final_policy_set_size=4,
    )

    assert selected == [
        "B0 RandomLegal",
        "policy_000300",
        "policy_000100",
        "policy_000150",
    ]


def test_run_final_eval_writes_matrix_exports_and_posterior_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    policies = ["policy_beta", "policy_alpha"]
    seed_file = tmp_path / "report_eval_seeds.txt"
    seed_file.write_text("11\n22\n33\n", encoding="utf-8")
    outcomes: dict[tuple[str, str, int, int], OutcomeToken] = {
        ("policy_beta", "policy_beta", 0, 0): "D",
        ("policy_beta", "policy_beta", 0, 1): "D",
        ("policy_beta", "policy_beta", 1, 0): "D",
        ("policy_beta", "policy_beta", 1, 1): "D",
        ("policy_beta", "policy_alpha", 0, 0): "W",
        ("policy_beta", "policy_alpha", 0, 1): "W",
        ("policy_beta", "policy_alpha", 1, 0): "L",
        ("policy_beta", "policy_alpha", 1, 1): "L",
        ("policy_beta", "policy_alpha", 2, 0): "W",
        ("policy_beta", "policy_alpha", 2, 1): "L",
        ("policy_alpha", "policy_alpha", 0, 0): "D",
        ("policy_alpha", "policy_alpha", 0, 1): "D",
        ("policy_alpha", "policy_alpha", 1, 0): "D",
        ("policy_alpha", "policy_alpha", 1, 1): "D",
    }
    runner = _FakeMatrixRunner(outcomes)

    payload = run_final_eval(
        output_dir=output_dir,
        runner=runner,
        policy_ids=policies,
        paired_seeds=[11, 22, 33],
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        sample_count=16,
        seed_file_path=seed_file,
        metadata={"paper_tag": "final_results_v1"},
    )

    assert payload["policy_ids"] == policies
    assert payload["metadata"]["paper_tag"] == "final_results_v1"
    assert payload["metadata"]["seed_file"]["path"] == seed_file.as_posix()
    assert payload["matrices"]["stop_reason"]["values"] == [
        ["precision", "budget"],
        ["budget", "precision"],
    ]
    assert payload["matrices"]["paired_seed_count"]["values"] == [
        [2, 3],
        [3, 2],
    ]
    assert payload["matrices"]["mean"]["values"][1][0] == pytest.approx(
        1.0 - payload["matrices"]["mean"]["values"][0][1]
    )
    assert payload["matrices"]["ci_low"]["values"][1][0] == pytest.approx(
        1.0 - payload["matrices"]["ci_high"]["values"][0][1]
    )
    assert payload["matrices"]["ci_high"]["values"][1][0] == pytest.approx(
        1.0 - payload["matrices"]["ci_low"]["values"][0][1]
    )
    assert len(payload["posterior_samples"]["values"]) == 2
    assert len(payload["posterior_samples"]["values"][0][1]) == 16
    assert len(payload["posterior_samples"]["values"][1][0]) == 16
    assert all(
        forward + reverse == pytest.approx(1.0)
        for forward, reverse in zip(
            payload["posterior_samples"]["values"][0][1],
            payload["posterior_samples"]["values"][1][0],
            strict=True,
        )
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    posterior_payload = json.loads((output_dir / "posterior_samples.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    policy_set = json.loads((output_dir / "policy_set.json").read_text(encoding="utf-8"))

    assert summary["policy_ids"] == policies
    assert posterior_payload["policy_ids"] == policies
    assert metadata["selection"] == {"mode": "explicit", "policy_count": 2}
    assert metadata["matchup_artifacts"]["kind"] == "canonical_unordered_pairs_v1"
    assert policy_set == {"policy_ids": policies}
    assert len(summary["matchups"]) == 3
    assert summary["matchups"][1]["matrix_cells"] == [
        {"focal_policy_index": 0, "opponent_policy_index": 1},
        {"focal_policy_index": 1, "opponent_policy_index": 0},
    ]

    with (output_dir / "matrices" / "stop_reason.csv").open("r", encoding="utf-8") as handle:
        stop_reason_rows = list(csv.reader(handle))
    assert stop_reason_rows == [
        ["focal_policy_id", "policy_beta", "policy_alpha"],
        ["policy_beta", "precision", "budget"],
        ["policy_alpha", "budget", "precision"],
    ]

    with (output_dir / "matchups.csv").open("r", encoding="utf-8") as handle:
        matchup_rows = list(csv.DictReader(handle))
    assert len(matchup_rows) == 3
    assert matchup_rows[0]["matchup_dir"] == "matchups/00_policy_beta__vs__00_policy_beta"
    assert matchup_rows[1]["matchup_dir"] == "matchups/00_policy_beta__vs__01_policy_alpha"
    assert matchup_rows[1]["paired_seed_count"] == "3"
    assert matchup_rows[1]["observed_paired_seed_count"] == "3"
    assert matchup_rows[1]["excluded_paired_seed_count"] == "0"
    assert matchup_rows[1]["has_payoff_samples"] == "True"
    assert matchup_rows[2]["stop_reason"] == "precision"

    budget_summary = json.loads(
        (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "matchup_summary.json").read_text(
            encoding="utf-8"
        )
    )
    matchup_samples = json.loads(
        (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "posterior_samples.json").read_text(
            encoding="utf-8"
        )
    )

    assert budget_summary["stop_reason"] == "budget"
    assert budget_summary["observed_paired_seeds"] == 3
    assert budget_summary["excluded_paired_seeds"] == 0
    assert budget_summary["has_payoff_samples"] is True
    assert budget_summary["evaluation_context"]["used_paired_seeds"] == [11, 22, 33]
    assert matchup_samples["sample_count"] == 16
    assert len(matchup_samples["samples"]) == 16
    assert summary["matrices"]["observed_paired_seeds"]["values"] == [[2, 3], [3, 2]]
    assert summary["matrices"]["wins"]["values"] == [[0, 3], [3, 0]]
    assert summary["matrices"]["losses"]["values"] == [[0, 3], [3, 0]]
    assert not (output_dir / "matchups" / "01_policy_alpha__vs__00_policy_beta").exists()
    assert (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "diagnostics.json").is_file()
    assert (output_dir / "matchups" / "00_policy_beta__vs__01_policy_alpha" / "episodes.jsonl").is_file()
    assert {(call.focal_policy_id, call.opponent_policy_id) for call in runner.calls} == {
        ("policy_beta", "policy_beta"),
        ("policy_beta", "policy_alpha"),
        ("policy_alpha", "policy_alpha"),
    }
    assert len(runner.calls) == 14


def test_run_final_eval_rejects_duplicate_explicit_policy_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy_ids must be unique"):
        run_final_eval(
            output_dir=tmp_path / "final_eval",
            runner=_FakeMatrixRunner({}),
            policy_ids=["policy_dup", "policy_dup"],
            paired_seeds=[11],
            stage1_paired_seeds=1,
            max_paired_seeds=1,
            stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
        )


def test_run_final_eval_rejects_underfilled_deterministic_selection(tmp_path: Path) -> None:
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    config = _selection_config(
        include_random_legal_baseline_b0=False,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(),
    )

    with pytest.raises(ValueError, match="underfilled"):
        run_final_eval(
            output_dir=tmp_path / "final_eval",
            runner=_FakeMatrixRunner({}),
            paired_seeds=[11],
            stage1_paired_seeds=1,
            max_paired_seeds=1,
            stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
            snapshot_registry_path=snapshot_registry_path,
            dev_eval_summaries_path=dev_eval_summaries_path,
            selection_config=config,
            final_policy_set_size=3,
        )


def test_run_final_eval_emits_explicit_s2_no_included_pair_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    policies = ["policy_alpha", "policy_beta"]
    outcomes: dict[tuple[str, str, int, int], OutcomeToken] = {
        (focal_policy_id, opponent_policy_id, pair_index, swap_index): "T"
        for focal_policy_id in policies
        for opponent_policy_id in policies
        for pair_index in range(2)
        for swap_index in (0, 1)
    }
    runner = _FakeMatrixRunner(outcomes)

    payload = run_final_eval(
        output_dir=output_dir,
        runner=runner,
        policy_ids=policies,
        paired_seeds=[11, 22],
        stage1_paired_seeds=1,
        max_paired_seeds=2,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        scheme="S2",
        sample_count=16,
    )

    assert payload["matrices"]["paired_seed_count"]["values"] == [[0, 0], [0, 0]]
    assert payload["matrices"]["observed_paired_seeds"]["values"] == [[2, 2], [2, 2]]
    assert payload["matrices"]["excluded_paired_seeds"]["values"] == [[2, 2], [2, 2]]
    assert payload["matrices"]["has_payoff_samples"]["values"] == [[False, False], [False, False]]
    assert payload["matrices"]["stop_reason"]["values"] == [
        ["no_included_pairs", "no_included_pairs"],
        ["no_included_pairs", "no_included_pairs"],
    ]
    assert payload["posterior_samples"]["values"] == [[[], []], [[], []]]

    matchup_summary = json.loads(
        (output_dir / "matchups" / "00_policy_alpha__vs__01_policy_beta" / "matchup_summary.json").read_text(
            encoding="utf-8"
        )
    )
    matchup_samples = json.loads(
        (output_dir / "matchups" / "00_policy_alpha__vs__01_policy_beta" / "posterior_samples.json").read_text(
            encoding="utf-8"
        )
    )

    assert matchup_summary["summary"]["truncations"] == 4
    assert matchup_summary["paired_seeds"] == 0
    assert matchup_summary["observed_paired_seeds"] == 2
    assert matchup_summary["excluded_paired_seeds"] == 2
    assert matchup_summary["has_payoff_samples"] is False
    assert matchup_summary["stop_reason"] == "no_included_pairs"
    assert matchup_summary["uncertainty"] == {
        "mean": None,
        "ci_low": None,
        "ci_high": None,
        "ci_half_width": None,
        "prob_gt_half": None,
        "prob_lt_half": None,
        "paired_seed_count": 0,
        "sample_count": 0,
    }
    assert matchup_samples["requested_sample_count"] == 16
    assert matchup_samples["sample_count"] == 0
    assert matchup_samples["has_payoff_samples"] is False
    assert matchup_samples["samples"] == []
