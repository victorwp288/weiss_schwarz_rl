from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import load_stack_config
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import final_eval as final_eval_module
from weiss_rl.eval import resolve_final_policy_set, run_final_eval
from weiss_rl.eval.final_eval_artifacts import (
    final_eval_matchup_manifest_rows,
    write_final_eval_artifacts,
)
from weiss_rl.eval.final_eval_matchups import (
    matchup_dir_name,
    run_final_eval_matchup,
    scheduled_game,
)
from weiss_rl.eval.final_eval_payload import build_final_eval_payload
from weiss_rl.eval.final_eval_policy_selection import (
    load_dev_eval_summaries,
    resolve_final_eval_policy_ids,
    validate_final_eval_seed_budget,
)
from weiss_rl.eval.harness import GameResult, ReplaySampleResult, ScheduledGame
from weiss_rl.eval.policy_set import DevEvalPolicySummary
from weiss_rl.tests._config_paths import canonical_stack_config_path

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
    return Path(__file__).resolve().parents[3]


def _selection_config(**overrides: Any):
    stack = load_stack_config(canonical_stack_config_path())
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


def test_final_eval_policy_selection_helpers_preserve_payload_shapes(tmp_path: Path) -> None:
    summaries_path = tmp_path / "dev_eval_summaries.json"
    summaries_path.write_text(
        json.dumps(
            {
                "policy_scalar": 0.25,
                "policy_structured": {
                    "aggregate_score": 0.75,
                    "anchor_scores": {"B0 RandomLegal": 0.9},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = load_dev_eval_summaries(summaries_path)
    resolved, selection_payload = resolve_final_eval_policy_ids(
        policy_ids=["B0 RandomLegal", 123],
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        selection_config=None,
        final_policy_set_size=None,
    )
    validate_final_eval_seed_budget(paired_seeds=[11, 22], stage1_paired_seeds=1, max_paired_seeds=2)

    assert summaries["policy_scalar"] == 0.25
    assert isinstance(summaries["policy_structured"], DevEvalPolicySummary)
    assert summaries["policy_structured"].anchor_scores == {"B0 RandomLegal": 0.9}
    assert resolved == ["B0 RandomLegal", "123"]
    assert selection_payload == {"mode": "explicit", "policy_count": 2}


def test_final_eval_policy_selection_helpers_preserve_errors() -> None:
    with pytest.raises(ValueError, match="policy_ids must contain at least one policy"):
        resolve_final_eval_policy_ids(
            policy_ids=[],
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            selection_config=None,
            final_policy_set_size=None,
        )
    with pytest.raises(ValueError, match="policy_ids must be unique"):
        resolve_final_eval_policy_ids(
            policy_ids=["policy_dup", "policy_dup"],
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            selection_config=None,
            final_policy_set_size=None,
        )
    with pytest.raises(ValueError, match="missing: snapshot_registry_path, dev_eval_summaries_path"):
        resolve_final_eval_policy_ids(
            policy_ids=None,
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            selection_config=_selection_config(),
            final_policy_set_size=2,
        )
    with pytest.raises(ValueError, match="stage1_paired_seeds must be positive"):
        validate_final_eval_seed_budget(paired_seeds=[11], stage1_paired_seeds=0, max_paired_seeds=1)
    with pytest.raises(ValueError, match="max_paired_seeds must be >= stage1_paired_seeds"):
        validate_final_eval_seed_budget(paired_seeds=[11, 22], stage1_paired_seeds=2, max_paired_seeds=1)
    with pytest.raises(ValueError, match="final eval requires at least 3 paired seeds, found 2"):
        validate_final_eval_seed_budget(paired_seeds=[11, 22], stage1_paired_seeds=1, max_paired_seeds=3)


def test_final_eval_payload_builder_preserves_metadata_paths_and_reverse_cells(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    seed_file = tmp_path / "report_eval_seeds.txt"
    seed_file.write_text("11\n22\n33\n", encoding="utf-8")
    policy_ids = ["policy_a", "policy_b"]

    def result(
        *,
        focal_index: int,
        opponent_index: int,
        mean: float,
        posterior_samples: tuple[float, ...],
    ) -> dict[str, Any]:
        matchup_dir = output_dir / "matchups" / f"{focal_index:02d}_vs_{opponent_index:02d}"
        return {
            "focal_policy_id": policy_ids[focal_index],
            "opponent_policy_id": policy_ids[opponent_index],
            "focal_index": focal_index,
            "opponent_index": opponent_index,
            "matchup_dir": matchup_dir,
            "episodes_path": matchup_dir / "episodes.jsonl",
            "posterior_samples": posterior_samples,
            "summary": {
                "paired_seeds": 3,
                "observed_paired_seeds": 3,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "precision",
                "should_stop": True,
                "summary": {
                    "games": 6,
                    "wins": int(round(mean * 6)),
                    "losses": 6 - int(round(mean * 6)),
                    "draws": 0,
                    "truncations": 0,
                    "engine_errors": 0,
                },
                "uncertainty": {
                    "mean": mean,
                    "ci_low": max(0.0, mean - 0.1),
                    "ci_high": min(1.0, mean + 0.1),
                    "ci_half_width": 0.1,
                    "prob_gt_half": 0.8,
                    "prob_lt_half": 0.2,
                    "paired_seed_count": 3,
                },
            },
        }

    payload = build_final_eval_payload(
        output_dir=output_dir,
        policy_ids=policy_ids,
        matchup_results=[
            result(focal_index=0, opponent_index=0, mean=0.5, posterior_samples=(0.5,)),
            result(focal_index=0, opponent_index=1, mean=0.75, posterior_samples=(0.2, 0.8)),
            result(focal_index=1, opponent_index=1, mean=0.5, posterior_samples=(0.5,)),
        ],
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        paired_seeds=[11, 22, 33],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        scheme="S0",
        sample_count=16,
        selection_payload={"mode": "explicit", "policy_count": 2},
        metadata={"paper_tag": "final_results_v1"},
        seed_file_path=seed_file,
    )

    assert final_eval_module._build_final_eval_payload is build_final_eval_payload
    assert payload["metadata"]["paper_tag"] == "final_results_v1"
    assert payload["metadata"]["seed_file"]["path"] == seed_file.as_posix()
    assert payload["metadata"]["seed_file"]["sha256"]
    assert payload["matrices"]["mean"]["values"] == [[0.5, 0.75], [0.25, 0.5]]
    assert payload["matrices"]["wins"]["values"] == [[3, 4], [2, 3]]
    assert payload["posterior_samples"]["values"] == [[[0.5], [0.2, 0.8]], [[0.8, 0.19999999999999996], [0.5]]]
    assert payload["matchups"][1]["matchup_dir"] == "matchups/00_vs_01"
    assert payload["matchups"][1]["summary_path"] == "matchups/00_vs_01/matchup_summary.json"
    assert payload["matchups"][1]["matrix_cells"] == [
        {"focal_policy_index": 0, "opponent_policy_index": 1},
        {"focal_policy_index": 1, "opponent_policy_index": 0},
    ]


def test_final_eval_run_split_preserves_facade_and_upper_triangle_jobs(tmp_path: Path) -> None:
    from weiss_rl.eval import final_eval_run

    jobs = final_eval_run.build_final_eval_matchup_jobs(["policy_a", "policy_b", "policy_c"])

    assert final_eval_module.run_final_eval is final_eval_run.run_final_eval
    assert final_eval_module._build_matchup_jobs is final_eval_run.build_final_eval_matchup_jobs
    assert final_eval_module._build_run_payload is final_eval_run.build_final_eval_run_payload
    assert final_eval_module._resolve_run_policy_ids is final_eval_run.resolve_final_eval_run_policy_ids
    assert final_eval_module._run_matchup_jobs is final_eval_run.run_final_eval_matchup_jobs
    assert final_eval_module._validate_run_seed_budget is final_eval_run.validate_final_eval_run_seed_budget
    assert final_eval_module._write_run_artifacts is final_eval_run.write_final_eval_run_artifacts
    assert [(job.focal_index, job.opponent_index, job.focal_policy_id, job.opponent_policy_id) for job in jobs] == [
        (0, 0, "policy_a", "policy_a"),
        (0, 1, "policy_a", "policy_b"),
        (0, 2, "policy_a", "policy_c"),
        (1, 1, "policy_b", "policy_b"),
        (1, 2, "policy_b", "policy_c"),
        (2, 2, "policy_c", "policy_c"),
    ]

    calls: list[dict[str, Any]] = []

    def fake_run_matchup(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "focal_index": kwargs["focal_index"],
            "opponent_index": kwargs["opponent_index"],
            "focal_policy_id": kwargs["focal_policy_id"],
            "opponent_policy_id": kwargs["opponent_policy_id"],
            "matchup_dir": tmp_path / "matchup",
            "summary": {},
            "posterior_samples": (),
            "records": (),
            "replay_samples": (),
        }

    results = final_eval_run.run_final_eval_matchup_jobs(
        output_dir=tmp_path / "final_eval",
        jobs=jobs[:2],
        runner=_FakeMatrixRunner({}),
        paired_seeds=[11, 22],
        stage1_paired_seeds=1,
        max_paired_seeds=2,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        scheme="S1",
        sample_count=7,
        run_matchup_fn=fake_run_matchup,
    )

    assert (tmp_path / "final_eval").is_dir()
    assert [(result["focal_index"], result["opponent_index"]) for result in results] == [(0, 0), (0, 1)]
    assert [(call["focal_policy_id"], call["opponent_policy_id"]) for call in calls] == [
        ("policy_a", "policy_a"),
        ("policy_a", "policy_b"),
    ]
    assert calls[0]["paired_seeds"] == [11, 22]
    assert calls[0]["stage1_paired_seeds"] == 1
    assert calls[0]["max_paired_seeds"] == 2
    assert calls[0]["scheme"] == "S1"
    assert calls[0]["sample_count"] == 7


def test_final_eval_matchup_runner_preserves_schedule_artifacts_and_early_stop(tmp_path: Path) -> None:
    runner = _AlwaysWinRunner()
    output_dir = tmp_path / "final_eval"
    first_swap = scheduled_game(
        pair_index=2,
        swap_index=1,
        episode_seed=33,
        focal_policy_id="policy_000021",
        opponent_policy_id="B3 HeuristicPublicAggro",
    )

    result = run_final_eval_matchup(
        output_dir=output_dir,
        focal_index=0,
        opponent_index=1,
        focal_policy_id="policy_000021",
        opponent_policy_id="B3 HeuristicPublicAggro",
        paired_seeds=[11, 22, 33, 44],
        stage1_paired_seeds=2,
        max_paired_seeds=4,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        runner=runner,
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        scheme="S0",
        sample_count=8,
    )

    assert final_eval_module._run_matchup is run_final_eval_matchup
    assert final_eval_module._scheduled_game is scheduled_game
    assert final_eval_module._matchup_dir_name is matchup_dir_name
    assert first_swap.seat0_policy_id == "B3 HeuristicPublicAggro"
    assert first_swap.seat1_policy_id == "policy_000021"
    assert first_swap.focal_seat == 1
    assert first_swap.seat0_deck == "preset:aggro_deck_5hy_nino_v1"
    assert first_swap.seat1_deck == "preset:main_deck_5hy_yotsuba_v1"
    assert len(runner.calls) == 4
    assert [(call.pair_index, call.swap_index, call.focal_seat) for call in runner.calls] == [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    assert result["matchup_dir"].name == "00_policy_000021__vs__01_b3_heuristicpublicaggro"
    assert result["used_paired_seeds"] == (11, 22)
    assert len(result["records"]) == 4
    assert len(result["replay_samples"]) == 1
    assert result["summary"]["evaluation_context"] == {
        "artifact_scope": "final_eval",
        "focal_policy_index": 0,
        "opponent_policy_index": 1,
        "stage1_paired_seeds": 2,
        "max_paired_seeds": 4,
        "used_paired_seeds": [11, 22],
    }
    assert result["diagnostics"]["god_search"] == {"enabled": True, "checked": 4}
    assert len(result["episodes_path"].read_text(encoding="utf-8").splitlines()) == 4
    assert (result["matchup_dir"] / "matchup_summary.json").is_file()
    assert (result["matchup_dir"] / "matchup_summary.csv").is_file()
    assert (result["matchup_dir"] / "diagnostics.json").is_file()
    posterior_payload = json.loads((result["matchup_dir"] / "posterior_samples.json").read_text(encoding="utf-8"))
    assert posterior_payload["requested_sample_count"] == 8
    assert posterior_payload["sample_count"] == 8
    assert posterior_payload["has_payoff_samples"] is True


def test_final_eval_matchup_split_modules_preserve_legacy_exports(tmp_path: Path) -> None:
    from weiss_rl.eval import final_eval_matchup_outputs, final_eval_matchup_schedule, final_eval_matchups

    matchup_dir = tmp_path / "matchups" / "00_policy_alpha__vs__01_policy_beta"

    assert final_eval_matchups.scheduled_game is final_eval_matchup_schedule.scheduled_game
    assert final_eval_matchups.matchup_dir_name is final_eval_matchup_schedule.matchup_dir_name
    assert final_eval_matchups.slug is final_eval_matchup_schedule.slug
    assert final_eval_matchups.bootstrap_seed is final_eval_matchup_schedule.bootstrap_seed
    assert final_eval_matchups.build_matchup_payload is final_eval_matchup_outputs.build_matchup_payload
    assert final_eval_matchups.matchup_posterior_samples is final_eval_matchup_outputs.matchup_posterior_samples
    assert final_eval_matchups.build_matchup_diagnostics is final_eval_matchup_outputs.build_matchup_diagnostics
    assert final_eval_matchups.write_matchup_artifacts is final_eval_matchup_outputs.write_matchup_artifacts
    assert final_eval_matchups.matchup_artifact_paths is final_eval_matchup_outputs.matchup_artifact_paths

    paths = final_eval_matchup_outputs.matchup_artifact_paths(matchup_dir)
    assert paths.summary_json == matchup_dir / "matchup_summary.json"
    assert paths.summary_csv == matchup_dir / "matchup_summary.csv"
    assert paths.diagnostics_json == matchup_dir / "diagnostics.json"
    assert paths.posterior_samples_json == matchup_dir / "posterior_samples.json"
    assert final_eval_matchup_outputs.build_posterior_samples_payload(
        focal_policy_id="policy_alpha",
        opponent_policy_id="policy_beta",
        sample_count=8,
        posterior_samples=(0.25, 0.75),
        summary_payload={"has_payoff_samples": True},
    ) == {
        "focal_policy_id": "policy_alpha",
        "opponent_policy_id": "policy_beta",
        "requested_sample_count": 8,
        "sample_count": 2,
        "has_payoff_samples": True,
        "samples": [0.25, 0.75],
    }


def test_final_eval_artifact_writer_preserves_direct_output_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    output_dir.mkdir(parents=True)
    payload = {
        "metadata": {"selection": {"mode": "explicit", "policy_count": 2}},
        "policy_ids": ["policy_a", "policy_b"],
        "posterior_samples": {"policy_ids": ["policy_a", "policy_b"], "values": [[[], []], [[], []]]},
        "matrices": {
            "mean": {"policy_ids": ["policy_a", "policy_b"], "values": [[0.5, 0.75], [0.25, 0.5]]},
        },
        "matchups": [],
    }
    matchup_results = [
        {
            "focal_policy_id": "policy_a",
            "opponent_policy_id": "policy_b",
            "matchup_dir": output_dir / "matchups" / "00_policy_a__vs__01_policy_b",
            "summary": {
                "paired_seeds": 3,
                "observed_paired_seeds": 3,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "budget",
            },
            "records": [],
            "replay_samples": [],
        }
    ]

    write_final_eval_artifacts(output_dir=output_dir, payload=payload, matchup_results=matchup_results)
    rows = final_eval_matchup_manifest_rows(output_dir=output_dir, matchup_results=matchup_results)

    assert final_eval_module._write_final_eval_artifacts is write_final_eval_artifacts
    assert json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")) == payload["metadata"]
    assert json.loads((output_dir / "policy_set.json").read_text(encoding="utf-8")) == {
        "policy_ids": ["policy_a", "policy_b"]
    }
    assert (
        json.loads((output_dir / "posterior_samples.json").read_text(encoding="utf-8")) == payload["posterior_samples"]
    )
    assert (output_dir / "matrices" / "mean.json").is_file()
    with (output_dir / "matrices" / "mean.csv").open("r", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["focal_policy_id", "policy_a", "policy_b"],
            ["policy_a", "0.5", "0.75"],
            ["policy_b", "0.25", "0.5"],
        ]
    with (output_dir / "matchups.csv").open("r", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert rows[0]["matchup_dir"] == "matchups/00_policy_a__vs__01_policy_b"
    assert manifest[0]["matchup_dir"] == "matchups/00_policy_a__vs__01_policy_b"
    assert manifest[0]["has_payoff_samples"] == "True"


def test_final_eval_artifact_writer_preserves_canonical_layout_exports(tmp_path: Path) -> None:
    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.ensure_directories()
    payload = {
        "metadata": {"selection": {"mode": "explicit", "policy_count": 2}},
        "policy_ids": ["policy_a", "policy_b"],
        "posterior_samples": {
            "policy_ids": ["policy_a", "policy_b"],
            "values": [[[0.5], [0.75]], [[0.25], [0.5]]],
        },
        "matrices": {
            "mean": {"policy_ids": ["policy_a", "policy_b"], "values": [[0.5, 0.75], [0.25, 0.5]]},
            "paired_seed_count": {"policy_ids": ["policy_a", "policy_b"], "values": [[3, 4], [4, 3]]},
        },
        "matchups": [],
    }
    matchup_results = [
        {
            "focal_policy_id": "policy_a",
            "opponent_policy_id": "policy_b",
            "matchup_dir": layout.final_eval_matchups_dir / "00_policy_a__vs__01_policy_b",
            "summary": {
                "paired_seeds": 4,
                "observed_paired_seeds": 4,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "budget",
                "summary": {"games": 8, "truncations": 2},
            },
            "records": [],
            "replay_samples": [],
        }
    ]

    write_final_eval_artifacts(
        output_dir=layout.final_eval_dir,
        payload=payload,
        matchup_results=matchup_results,
    )

    assert layout.final_eval_posterior_samples_npz().is_file()
    assert layout.final_eval_payoff_matrix_csv("p_mean").read_text(encoding="utf-8").splitlines() == [
        "focal_policy_id,policy_a,policy_b",
        "policy_a,0.5,0.75",
        "policy_b,0.25,0.5",
    ]
    assert layout.truncation_heatmap_csv().read_text(encoding="utf-8").splitlines() == [
        "focal_policy_id,policy_a,policy_b",
        "policy_a,0.0,0.25",
        "policy_b,0.25,0.0",
    ]
    replay_verification = json.loads(layout.replay_verification_json().read_text(encoding="utf-8"))
    replay_index = json.loads(layout.replay_index_json().read_text(encoding="utf-8"))
    hashes = json.loads(layout.final_eval_aggregate_hashes_json().read_text(encoding="utf-8"))
    assert replay_verification["status"] == "not_sampled"
    assert replay_verification["index_path"] == "replays/index.json"
    assert replay_index == {"kind": "replay_index_v1", "samples": []}
    assert "eval/final_eval/summary.json" in hashes["artifacts"]
    assert "eval/diagnostics/replay_verification.json" in hashes["artifacts"]
    assert "replays/index.json" in hashes["artifacts"]


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
