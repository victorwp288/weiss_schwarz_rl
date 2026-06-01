from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from weiss_rl.experiments.guarded_league_bootstrap import (
    GuardedLeagueBootstrapConfig,
    LeagueSegmentRuntime,
    build_targeted_confirm_command,
    evaluate_guard,
    latest_policy_snapshot,
    recent_policy_snapshots,
    run_guarded_league_bootstrap,
    runtime_overrides_with_defaults,
)
from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_registry(run_dir: Path, *, snapshots: list[tuple[str, int]]) -> None:
    _write_json(
        run_dir / "training" / "snapshots" / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {
                    "policy_id": policy_id,
                    "update": update,
                    "path": f"training/snapshots/{policy_id}/weights.pt",
                    "weights_sha256": f"sha-{policy_id}",
                }
                for policy_id, update in snapshots
            ],
        },
    )
    for _policy_id, update in snapshots:
        checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_selection(path: Path, *, run_dir: Path, policy_id: str, update: int, score: float = 0.62) -> None:
    _write_json(
        path,
        {
            "selected": {
                "run_dir": run_dir.as_posix(),
                "run_name": run_dir.name,
                "snapshot_policy_id": policy_id,
                "update_count": update,
                "eligible": True,
                "selection_score": score,
                "selection_paired_seeds": 4,
                "selection_anchor_scores": {
                    "B1 NoLeague baseline": 0.55,
                    "B2 HeuristicPublic": 0.59,
                    "B3 HeuristicPublicAggro": 0.56,
                    "B4 HeuristicPublicControl": 0.57,
                },
            },
            "run_summaries": [{"run_name": run_dir.name, "latest_minus_best": -0.03}],
        },
    )


def test_runtime_overrides_keep_pinned_seed_defaults_with_custom_overrides() -> None:
    overrides = runtime_overrides_with_defaults(
        [
            "training.structured_aux.policy_anchor_coef=0.2",
            "league.pool.seed_snapshot_import_filter=pinned_or_source_champions",
        ]
    )

    assert "league.pool.seed_snapshot_champion_import=pinned" in overrides
    assert "league.pool.seed_snapshot_import_filter=pinned" not in overrides
    assert "league.pool.seed_snapshot_import_filter=pinned_or_source_champions" in overrides
    assert overrides[-1] == "league.pool.seed_snapshot_import_filter=pinned_or_source_champions"


def test_runtime_overrides_can_use_stack_seed_snapshot_policy_without_pinned_defaults() -> None:
    overrides = runtime_overrides_with_defaults(
        ["training.structured_aux.policy_anchor_coef=0.2"],
        apply_seed_snapshot_defaults=False,
    )

    assert overrides == ("training.structured_aux.policy_anchor_coef=0.2",)
    assert "league.pool.seed_snapshot_champion_import=pinned" not in overrides
    assert "league.pool.seed_snapshot_import_filter=pinned" not in overrides


def test_guarded_league_bootstrap_cli_builds_config_and_preserves_helper_exports(tmp_path: Path) -> None:
    from weiss_rl.experiments import guarded_league_bootstrap_cli, guarded_league_bootstrap_entrypoint

    reference_summary = tmp_path / "reference.json"
    _write_json(reference_summary, {"rows": [{"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.61}]})
    args = guarded_league_bootstrap_cli.build_guarded_league_bootstrap_parser().parse_args(
        [
            "--init-from-checkpoint",
            str(tmp_path / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"),
            "--seed-snapshot-run-dir",
            str(tmp_path / "seed"),
            "--b1-baseline-run-dir",
            str(tmp_path / "b1"),
            "--run-prefix",
            "guarded_test",
            "--segments",
            "2",
            "--segment-updates",
            "7",
            "--first-init-schedule-offset-updates",
            "0",
            "--num-envs",
            "16",
            "--unroll-length",
            "8",
            "--runtime-mode",
            "train_ordered",
            "--profile",
            "debug",
            "--device",
            "cpu",
            "--checkpoint-interval-updates",
            "3",
            "--collection-backend",
            "central",
            "--override",
            "training.profile_timers=true",
            "--required-anchor",
            "B2 HeuristicPublic",
            "--confirm-opponent",
            "B2 HeuristicPublic",
            "--confirm-paired-seeds",
            "32",
            "--publish-min-confirm-paired-seeds",
            "64",
            "--confirm-recent-candidate-count",
            "2",
            "--reference-summary-json",
            str(reference_summary),
            "--multiobjective-fixed-opponent",
            "B4 HeuristicPublicControl",
            "--learned-guard-opponent",
            "seed_selected",
            "--continue-unpublished-confirmed",
            "--dry-run",
        ]
    )

    config = guarded_league_bootstrap_cli.guarded_league_config_from_args(args=args, repo_root=tmp_path)

    assert config.repo_root == tmp_path
    assert config.run_prefix == "guarded_test"
    assert config.segments == 2
    assert config.first_init_schedule_offset_updates == 0
    assert config.b1_baseline_run_dir == tmp_path / "b1"
    assert config.runtime == LeagueSegmentRuntime(
        num_envs=16,
        unroll_length=8,
        segment_updates=7,
        runtime_mode="train_ordered",
        simulator_profile="debug",
        device="cpu",
        checkpoint_interval_updates=3,
        collection_backend="central",
        profile_timers=True,
        overrides=(
            "league.pool.seed_snapshot_champion_import=pinned",
            "league.pool.seed_snapshot_import_filter=pinned",
            "training.profile_timers=true",
        ),
    )
    assert config.required_anchors == ("B2 HeuristicPublic",)
    assert config.confirm_opponents == ("B2 HeuristicPublic",)
    assert config.confirm_paired_seeds == 32
    assert config.publish_min_confirm_paired_seeds == 64
    assert config.confirm_recent_candidate_count == 2
    assert config.reference_anchor_scores["B2 HeuristicPublic"] == pytest.approx(0.61)
    assert config.multiobjective_reference_summary_jsons == (reference_summary,)
    assert config.multiobjective_fixed_opponents == ("B4 HeuristicPublicControl",)
    assert config.learned_guard_opponents == ("seed_selected",)
    assert config.continue_unpublished_confirmed is True
    assert config.dry_run is True
    assert (
        guarded_league_bootstrap_entrypoint.parse_args
        is guarded_league_bootstrap_cli.parse_guarded_league_bootstrap_args
    )
    assert guarded_league_bootstrap_entrypoint._uses_seed_snapshot_opponents is (
        guarded_league_bootstrap_cli.uses_seed_snapshot_opponents
    )
    assert guarded_league_bootstrap_entrypoint._has_seed_snapshot_pool_override is (
        guarded_league_bootstrap_cli.has_seed_snapshot_pool_override
    )
    assert guarded_league_bootstrap_entrypoint._validate_seed_snapshot_policy is (
        guarded_league_bootstrap_cli.validate_guarded_league_bootstrap_args
    )


def test_guarded_league_bootstrap_cli_validates_seed_policy_and_counts(tmp_path: Path) -> None:
    from weiss_rl.experiments.guarded_league_bootstrap_cli import (
        build_guarded_league_bootstrap_parser,
        guarded_league_config_from_args,
    )

    parser = build_guarded_league_bootstrap_parser()
    seed_args = parser.parse_args(
        [
            "--init-from-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--seed-snapshot-run-dir",
            str(tmp_path / "seed"),
            "--required-anchor",
            "seed_example",
        ]
    )
    with pytest.raises(SystemExit, match="seed_\\* learned opponents require"):
        guarded_league_config_from_args(args=seed_args, repo_root=tmp_path)

    invalid_offset_args = parser.parse_args(
        [
            "--init-from-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--seed-snapshot-run-dir",
            str(tmp_path / "seed"),
            "--first-init-schedule-offset-updates",
            "-1",
        ]
    )
    with pytest.raises(SystemExit, match="--first-init-schedule-offset-updates must be >= 0"):
        guarded_league_config_from_args(args=invalid_offset_args, repo_root=tmp_path)

    invalid_count_args = parser.parse_args(
        [
            "--init-from-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--seed-snapshot-run-dir",
            str(tmp_path / "seed"),
            "--confirm-recent-candidate-count",
            "0",
        ]
    )
    with pytest.raises(SystemExit, match="--confirm-recent-candidate-count must be >= 1"):
        guarded_league_config_from_args(args=invalid_count_args, repo_root=tmp_path)


def test_guarded_league_bootstrap_reexports_selection_helpers() -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_selection

    assert guarded_league_bootstrap.SnapshotCandidate is guarded_league_bootstrap_selection.SnapshotCandidate
    assert guarded_league_bootstrap.policy_snapshots is guarded_league_bootstrap_selection.policy_snapshots
    assert guarded_league_bootstrap.latest_policy_snapshot is guarded_league_bootstrap_selection.latest_policy_snapshot
    assert (
        guarded_league_bootstrap.recent_policy_snapshots is guarded_league_bootstrap_selection.recent_policy_snapshots
    )
    assert guarded_league_bootstrap.evaluate_guard is guarded_league_bootstrap_selection.evaluate_guard
    assert (
        guarded_league_bootstrap.evaluate_multiobjective_guard
        is guarded_league_bootstrap_selection.evaluate_multiobjective_guard
    )
    assert (
        guarded_league_bootstrap.load_reference_scores_or_empty
        is guarded_league_bootstrap_selection.load_reference_scores_or_empty
    )
    assert guarded_league_bootstrap._resolve_repo_path is guarded_league_bootstrap_selection.resolve_repo_path
    assert (
        guarded_league_bootstrap._selected_confirm_summary_path
        is guarded_league_bootstrap_selection.selected_confirm_summary_path
    )


def test_guarded_league_bootstrap_segment_plan_builds_dry_run_record(tmp_path: Path) -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_segments

    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    seed_run_dir = tmp_path / "runs" / "seed"
    diagnostics_dir = tmp_path / "diagnostics"
    config = GuardedLeagueBootstrapConfig(
        repo_root=tmp_path,
        init_checkpoint_path=checkpoint,
        seed_snapshot_run_dir=seed_run_dir,
        run_prefix="guarded_segments",
        first_init_schedule_offset_updates=0,
        runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
        confirm_paired_seeds=8,
        confirm_recent_candidate_count=2,
        learned_guard_opponents=("seed_selected",),
    )

    assert (
        guarded_league_bootstrap.GuardedLeagueSegmentPlan is guarded_league_bootstrap_segments.GuardedLeagueSegmentPlan
    )
    assert guarded_league_bootstrap.build_segment_plan is guarded_league_bootstrap_segments.build_segment_plan
    assert (
        guarded_league_bootstrap.build_initial_segment_record
        is guarded_league_bootstrap_segments.build_initial_segment_record
    )
    assert (
        guarded_league_bootstrap.populate_dry_run_segment_record
        is guarded_league_bootstrap_segments.populate_dry_run_segment_record
    )
    assert (
        guarded_league_bootstrap._effective_confirm_opponents
        is guarded_league_bootstrap_segments.effective_confirm_opponents
    )
    assert (
        guarded_league_bootstrap._effective_learned_guard_opponents
        is guarded_league_bootstrap_segments.effective_learned_guard_opponents
    )
    assert (
        guarded_league_bootstrap._is_seed_wrapped_suffix_match
        is guarded_league_bootstrap_segments.is_seed_wrapped_suffix_match
    )

    plan = guarded_league_bootstrap_segments.build_segment_plan(
        config=config,
        repo_root=tmp_path,
        diagnostics_dir=diagnostics_dir,
        segment_index=1,
        init_checkpoint_path=checkpoint,
    )
    record = guarded_league_bootstrap_segments.build_initial_segment_record(
        plan=plan,
        repo_root=tmp_path,
        source_checkpoint=checkpoint,
    )
    guarded_league_bootstrap_segments.populate_dry_run_segment_record(
        record=record,
        config=config,
        plan=plan,
    )

    assert plan.run_label == "guarded_segments_seg01"
    assert _option(plan.train_command, "--init-schedule-offset-updates") == "0"
    assert record["status"] == "planned"
    assert record["run_dir"] == "runs/guarded_segments_seg01"
    assert record["preselect_json"] == "diagnostics/guarded_segments_seg01_candidate_preconfirm.json"
    assert _option(record["preselect_command"]["argv"], "--output-json").endswith(
        "diagnostics/guarded_segments_seg01_candidate_preconfirm.json"
    )
    assert _option(record["targeted_confirm_command_template"]["argv"], "--focal-policy-id") == "<candidate-policy-id>"
    assert _option(record["targeted_confirm_command_template"]["argv"], "--output-subdir") == (
        "guard_confirm8_<candidate-policy-id>"
    )
    assert record["confirm_recent_candidate_count"] == 2
    assert guarded_league_bootstrap_segments.effective_learned_guard_opponents(config) == ("seed_selected",)


def test_guarded_league_bootstrap_outcome_helpers_preserve_stop_statuses() -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_outcomes

    assert guarded_league_bootstrap.SegmentStopOutcome is guarded_league_bootstrap_outcomes.SegmentStopOutcome
    assert (
        guarded_league_bootstrap.rejected_segment_outcome is guarded_league_bootstrap_outcomes.rejected_segment_outcome
    )
    assert (
        guarded_league_bootstrap.publish_confirmation_skip_payload
        is guarded_league_bootstrap_outcomes.publish_confirmation_skip_payload
    )
    assert (
        guarded_league_bootstrap.unpublished_confirmation_stop_outcome
        is guarded_league_bootstrap_outcomes.unpublished_confirmation_stop_outcome
    )
    assert (
        guarded_league_bootstrap.apply_segment_stop_outcome
        is guarded_league_bootstrap_outcomes.apply_segment_stop_outcome
    )

    assert guarded_league_bootstrap_outcomes.rejected_segment_outcome(
        selected={"eligible": False},
        guard={"passed": True},
        multiobjective_guard={"passed": True},
    ) == guarded_league_bootstrap_outcomes.SegmentStopOutcome(
        segment_status="rejected",
        summary_status="stopped_ineligible",
        stop_reason="selected checkpoint did not meet required anchor threshold",
    )
    assert guarded_league_bootstrap_outcomes.rejected_segment_outcome(
        selected={"eligible": True},
        guard={"passed": False},
        multiobjective_guard={"passed": False},
    ) == guarded_league_bootstrap_outcomes.SegmentStopOutcome(
        segment_status="rejected",
        summary_status="stopped_guard_failed",
        stop_reason="selected checkpoint failed B2/B3/B4 guard",
    )
    assert guarded_league_bootstrap_outcomes.rejected_segment_outcome(
        selected={"eligible": True},
        guard={"passed": True},
        multiobjective_guard={"passed": False},
    ) == guarded_league_bootstrap_outcomes.SegmentStopOutcome(
        segment_status="rejected",
        summary_status="stopped_multiobjective_guard_failed",
        stop_reason="selected checkpoint failed fixed/learned multi-objective guard",
    )
    assert (
        guarded_league_bootstrap_outcomes.rejected_segment_outcome(
            selected={"eligible": True},
            guard={"passed": True},
            multiobjective_guard=None,
        )
        is None
    )

    publish_skipped = guarded_league_bootstrap_outcomes.publish_confirmation_skip_payload(
        confirm_paired_seeds=32,
        publish_min_confirm_paired_seeds=256,
        continue_unpublished_confirmed=True,
    )
    assert publish_skipped == {
        "reason": "confirmation_seed_count_below_publish_minimum",
        "confirm_paired_seeds": 32,
        "publish_min_confirm_paired_seeds": 256,
        "continued_without_publish": True,
    }
    assert (
        guarded_league_bootstrap_outcomes.unpublished_confirmation_stop_outcome(
            continue_unpublished_confirmed=True,
            has_more_segments=True,
        )
        is None
    )
    assert guarded_league_bootstrap_outcomes.unpublished_confirmation_stop_outcome(
        continue_unpublished_confirmed=True,
        has_more_segments=False,
    ) == guarded_league_bootstrap_outcomes.SegmentStopOutcome(
        segment_status="accepted_unpublished",
        summary_status="completed_unpublished_confirmation_insufficient",
        stop_reason=(
            "all requested segments passed guard but were not published because confirmation seed count "
            "is below publish_min_confirm_paired_seeds"
        ),
    )
    assert guarded_league_bootstrap_outcomes.unpublished_confirmation_stop_outcome(
        continue_unpublished_confirmed=False,
        has_more_segments=True,
    ) == guarded_league_bootstrap_outcomes.SegmentStopOutcome(
        segment_status="accepted_unpublished",
        summary_status="stopped_publish_confirmation_insufficient",
        stop_reason=(
            "selected checkpoint passed guard but was not published because confirmation seed count "
            "is below publish_min_confirm_paired_seeds"
        ),
    )

    segment_record: dict[str, object] = {}
    summary: dict[str, object] = {}
    guarded_league_bootstrap_outcomes.apply_segment_stop_outcome(
        segment_record=segment_record,
        summary=summary,
        outcome=guarded_league_bootstrap_outcomes.SegmentStopOutcome(
            segment_status="rejected",
            summary_status="stopped_guard_failed",
            stop_reason="selected checkpoint failed B2/B3/B4 guard",
        ),
    )
    assert segment_record == {"status": "rejected"}
    assert summary == {
        "status": "stopped_guard_failed",
        "stop_reason": "selected checkpoint failed B2/B3/B4 guard",
    }


def test_guarded_league_bootstrap_final_selection_helpers_resolve_scores_and_artifacts(tmp_path: Path) -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_final_selection

    assert (
        guarded_league_bootstrap.selected_anchor_scores
        is guarded_league_bootstrap_final_selection.selected_anchor_scores
    )
    assert (
        guarded_league_bootstrap.populate_selected_segment_record
        is guarded_league_bootstrap_final_selection.populate_selected_segment_record
    )
    assert (
        guarded_league_bootstrap.evaluate_selected_multiobjective_guard
        is guarded_league_bootstrap_final_selection.evaluate_selected_multiobjective_guard
    )
    assert (
        guarded_league_bootstrap.write_multiobjective_guard_artifact
        is guarded_league_bootstrap_final_selection.write_multiobjective_guard_artifact
    )

    final_selected = {
        "eligible": True,
        "selection_confirmation_summary_path": "runs/seg/eval/confirm/targeted_confirm8_summary.json",
    }
    targeted_records = [
        {
            "summary_path": "runs/seg/eval/other/targeted_confirm8_summary.json",
            "anchor_scores": {"B2 HeuristicPublic": 0.42},
        },
        {
            "summary_path": "runs/seg/eval/confirm/targeted_confirm8_summary.json",
            "anchor_scores": {"B2 HeuristicPublic": 0.66},
        },
    ]

    scores = guarded_league_bootstrap_final_selection.selected_anchor_scores(
        final_selected=final_selected,
        selected_confirmation_summary_path="runs/seg/eval/confirm/targeted_confirm8_summary.json",
        targeted_confirm_records=targeted_records,
    )
    assert scores == {"B2 HeuristicPublic": 0.66}
    assert guarded_league_bootstrap_final_selection.selected_anchor_scores(
        final_selected={"selection_anchor_scores": {"B2 HeuristicPublic": 0.71}},
        selected_confirmation_summary_path="",
        targeted_confirm_records=targeted_records,
    ) == {"B2 HeuristicPublic": 0.71}
    assert guarded_league_bootstrap_final_selection.selected_anchor_scores(
        final_selected={},
        selected_confirmation_summary_path="missing",
        targeted_confirm_records=targeted_records,
    ) == {"B2 HeuristicPublic": 0.66}

    segment_record: dict[str, object] = {}
    guard = {"passed": True, "failures": []}
    guarded_league_bootstrap_final_selection.populate_selected_segment_record(
        segment_record=segment_record,
        repo_root=tmp_path,
        final_selected=final_selected,
        selected_confirmation_summary_path="runs/seg/eval/confirm/targeted_confirm8_summary.json",
        targeted_confirm_records=targeted_records,
        anchor_scores=scores,
        guard=guard,
    )
    assert segment_record["targeted_confirm_summary"] == "runs/seg/eval/confirm/targeted_confirm8_summary.json"
    assert segment_record["targeted_anchor_scores"] == scores
    assert segment_record["selected"] == final_selected
    assert segment_record["anchor_scores"] == scores
    assert segment_record["guard"] == guard

    artifact_record: dict[str, object] = {}
    (tmp_path / "diagnostics").mkdir()
    artifact = guarded_league_bootstrap_final_selection.write_multiobjective_guard_artifact(
        segment_record=artifact_record,
        multiobjective_guard={"passed": True, "groups": {"learned_opponents": {"mean": 0.75}}},
        diagnostics_dir=tmp_path / "diagnostics",
        run_label="guarded_seg01",
        repo_root=tmp_path,
    )
    assert artifact == tmp_path / "diagnostics" / "guarded_seg01_multiobjective_gate.json"
    assert artifact_record["multiobjective_guard_json"] == "diagnostics/guarded_seg01_multiobjective_gate.json"
    assert artifact_record["multiobjective_guard"] == {"passed": True, "groups": {"learned_opponents": {"mean": 0.75}}}
    assert json.loads(artifact.read_text(encoding="utf-8")) == artifact_record["multiobjective_guard"]


def test_guarded_league_bootstrap_confirmation_helpers_preserve_candidate_and_record_bookkeeping(
    tmp_path: Path,
) -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_confirmations

    assert (
        guarded_league_bootstrap.confirm_focal_policy_ids
        is guarded_league_bootstrap_confirmations.confirm_focal_policy_ids
    )
    assert (
        guarded_league_bootstrap.populate_confirm_candidate_segment_record
        is guarded_league_bootstrap_confirmations.populate_confirm_candidate_segment_record
    )
    assert (
        guarded_league_bootstrap.record_targeted_confirm_result
        is guarded_league_bootstrap_confirmations.record_targeted_confirm_result
    )
    assert (
        guarded_league_bootstrap.populate_targeted_confirm_segment_record
        is guarded_league_bootstrap_confirmations.populate_targeted_confirm_segment_record
    )

    recent = [
        guarded_league_bootstrap.SnapshotCandidate(
            policy_id="policy_000001",
            update=5,
            checkpoint_path=tmp_path / "runs" / "seg" / "training" / "checkpoints" / "checkpoint_5.pt",
        ),
        guarded_league_bootstrap.SnapshotCandidate(
            policy_id="policy_000002",
            update=10,
            checkpoint_path=tmp_path / "runs" / "seg" / "training" / "checkpoints" / "checkpoint_10.pt",
        ),
        guarded_league_bootstrap.SnapshotCandidate(
            policy_id="policy_000003",
            update=15,
            checkpoint_path=tmp_path / "runs" / "seg" / "training" / "checkpoints" / "checkpoint_15.pt",
        ),
    ]
    focal_policy_ids = guarded_league_bootstrap_confirmations.confirm_focal_policy_ids(
        preselected={"snapshot_policy_id": "policy_000002"},
        recent_candidates=recent,
        latest_policy_id="policy_000003",
        limit=3,
    )
    assert focal_policy_ids == ["policy_000002", "policy_000001", "policy_000003"]
    assert guarded_league_bootstrap_confirmations.confirm_focal_policy_ids(
        preselected=None,
        recent_candidates=[],
        latest_policy_id="policy_latest",
        limit=3,
    ) == ["policy_latest"]

    segment_record: dict[str, object] = {}
    guarded_league_bootstrap_confirmations.populate_confirm_candidate_segment_record(
        segment_record=segment_record,
        latest=recent[-1],
        repo_root=tmp_path,
        confirm_recent_candidate_count=3,
        focal_policy_ids=focal_policy_ids,
        preselected={"snapshot_policy_id": "policy_000002"},
    )
    assert segment_record["latest_policy_id"] == "policy_000003"
    assert segment_record["latest_update"] == 15
    assert segment_record["latest_checkpoint"] == "runs/seg/training/checkpoints/checkpoint_15.pt"
    assert segment_record["confirm_recent_candidate_count"] == 3
    assert segment_record["confirm_focal_policy_ids"] == focal_policy_ids
    assert segment_record["preselected"] == {"snapshot_policy_id": "policy_000002"}

    run_dir = tmp_path / "runs" / "seg"
    confirm_record = {
        "focal_policy_id": "policy_000002",
        "output_subdir": "guard_confirm4_policy_000002",
        "command": {"argv": ["uv", "run", "confirm"], "display": "uv run confirm"},
    }
    _write_json(
        run_dir / "eval" / "guard_confirm4_policy_000002" / "targeted_confirm4_summary.json",
        {"rows": [{"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.62}]},
    )
    summary_path = guarded_league_bootstrap_confirmations.record_targeted_confirm_result(
        confirm_record=confirm_record,
        run_dir=run_dir,
        paired_seeds=4,
        repo_root=tmp_path,
    )
    assert summary_path == run_dir / "eval" / "guard_confirm4_policy_000002" / "targeted_confirm4_summary.json"
    assert confirm_record["summary_path"] == "runs/seg/eval/guard_confirm4_policy_000002/targeted_confirm4_summary.json"
    assert confirm_record["anchor_scores"] == {"B2 HeuristicPublic": 0.62}

    aggregate_record: dict[str, object] = {}
    guarded_league_bootstrap_confirmations.populate_targeted_confirm_segment_record(
        segment_record=aggregate_record,
        targeted_confirm_records=[confirm_record],
    )
    assert aggregate_record["targeted_confirm_commands"] == [confirm_record["command"]]
    assert aggregate_record["targeted_confirm_records"] == [confirm_record]
    assert aggregate_record["confirm_focal_policy_id"] == "policy_000002"
    assert aggregate_record["targeted_confirm_command"] == confirm_record["command"]


def test_guarded_league_bootstrap_summary_helpers_preserve_summary_contract(tmp_path: Path) -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_summary

    assert (
        guarded_league_bootstrap.build_guarded_league_bootstrap_summary
        is guarded_league_bootstrap_summary.build_guarded_league_bootstrap_summary
    )
    assert (
        guarded_league_bootstrap.guarded_bootstrap_summary_path
        is guarded_league_bootstrap_summary.guarded_bootstrap_summary_path
    )
    assert (
        guarded_league_bootstrap.write_guarded_league_bootstrap_summary
        is guarded_league_bootstrap_summary.write_guarded_league_bootstrap_summary
    )

    reference_summary = tmp_path / "diagnostics" / "reference_summary.json"
    config = GuardedLeagueBootstrapConfig(
        repo_root=tmp_path,
        init_checkpoint_path=tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt",
        seed_snapshot_run_dir=tmp_path / "runs" / "seed",
        b1_baseline_run_dir=tmp_path / "runs" / "b1",
        run_prefix="guarded_summary",
        stack_config=tmp_path / "configs" / "stack.yaml",
        segments=3,
        confirm_paired_seeds=32,
        publish_min_confirm_paired_seeds=128,
        confirm_recent_candidate_count=2,
        required_anchors=("B3 HeuristicPublicAggro", "B2 HeuristicPublic"),
        confirm_opponents=("seed_B2 HeuristicPublic", "seed_selected"),
        reference_anchor_scores={"B3 HeuristicPublicAggro": 0.55, "B2 HeuristicPublic": 0.6},
        multiobjective_reference_summary_jsons=(reference_summary,),
        learned_guard_opponents=(),
        continue_unpublished_confirmed=True,
        dry_run=True,
    )

    summary = guarded_league_bootstrap_summary.build_guarded_league_bootstrap_summary(
        config=config,
        repo_root=tmp_path,
        effective_learned_guard_opponents=("seed_selected",),
        created_unix=123.5,
    )

    assert summary["kind"] == "guarded_league_bootstrap_v1"
    assert summary["created_unix"] == 123.5
    assert summary["repo_root"] == tmp_path.as_posix()
    assert summary["stack_config"] == "configs/stack.yaml"
    assert summary["seed_snapshot_run_dir"] == "runs/seed"
    assert summary["b1_baseline_run_dir"] == "runs/b1"
    assert summary["initial_checkpoint"] == "runs/seed/training/checkpoints/checkpoint_25.pt"
    assert summary["run_prefix"] == "guarded_summary"
    assert summary["segments_requested"] == 3
    assert summary["confirm_paired_seeds"] == 32
    assert summary["publish_min_confirm_paired_seeds"] == 128
    assert summary["confirm_recent_candidate_count"] == 2
    assert summary["required_anchors"] == ["B3 HeuristicPublicAggro", "B2 HeuristicPublic"]
    assert summary["confirm_opponents"] == ["seed_B2 HeuristicPublic", "seed_selected"]
    assert summary["effective_confirm_opponents"] == ["seed_B2 HeuristicPublic", "seed_selected"]
    assert summary["reference_anchor_scores"] == {"B2 HeuristicPublic": 0.6, "B3 HeuristicPublicAggro": 0.55}
    assert summary["multiobjective_reference_summary_jsons"] == ["diagnostics/reference_summary.json"]
    assert summary["configured_learned_guard_opponents"] == []
    assert summary["learned_guard_opponents"] == ["seed_selected"]
    assert summary["learned_guard_opponents_inferred"] is True
    assert summary["multiobjective_thresholds"] == {
        "max_fixed_reference_drop": 0.0,
        "max_learned_reference_drop": None,
        "min_fixed_score": 0.5,
        "min_learned_mean": 0.5,
        "min_learned_reference_delta": 0.0,
        "min_learned_score": 0.5,
    }
    assert summary["continue_unpublished_confirmed"] is True
    assert summary["segments"] == []
    assert summary["status"] == "planned"

    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir()
    summary_path = guarded_league_bootstrap_summary.guarded_bootstrap_summary_path(
        diagnostics_dir=diagnostics_dir,
        run_prefix="guarded_summary",
    )
    guarded_league_bootstrap_summary.write_guarded_league_bootstrap_summary(
        summary=summary,
        summary_path=summary_path,
    )
    assert summary["summary_path"] == summary_path.as_posix()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_guarded_league_bootstrap_segment_runner_builds_dry_run_step(tmp_path: Path) -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_segment_runner

    assert (
        guarded_league_bootstrap.GuardedLeagueSegmentStepResult
        is guarded_league_bootstrap_segment_runner.GuardedLeagueSegmentStepResult
    )
    assert (
        guarded_league_bootstrap.run_guarded_league_segment_step
        is guarded_league_bootstrap_segment_runner.run_guarded_league_segment_step
    )

    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    config = GuardedLeagueBootstrapConfig(
        repo_root=tmp_path,
        init_checkpoint_path=checkpoint,
        seed_snapshot_run_dir=tmp_path / "runs" / "seed",
        run_prefix="guarded_runner",
        first_init_schedule_offset_updates=0,
        runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
        confirm_paired_seeds=8,
        dry_run=True,
    )
    summary: dict[str, object] = {"segments": [], "status": "planned"}

    result = guarded_league_bootstrap_segment_runner.run_guarded_league_segment_step(
        config=config,
        repo_root=tmp_path,
        diagnostics_dir=tmp_path / "diagnostics",
        segment_index=1,
        current_checkpoint=checkpoint,
        summary=summary,
        effective_learned_guard_opponents=(),
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )

    assert result == guarded_league_bootstrap_segment_runner.GuardedLeagueSegmentStepResult(
        current_checkpoint=checkpoint,
        stop=True,
    )
    assert len(summary["segments"]) == 1
    segment = summary["segments"][0]
    assert segment["status"] == "planned"
    assert segment["run_label"] == "guarded_runner_seg01"
    assert segment["source_checkpoint"] == "runs/seed/training/checkpoints/checkpoint_25.pt"
    assert _option(segment["train_command"]["argv"], "--init-schedule-offset-updates") == "0"
    assert _option(segment["targeted_confirm_command_template"]["argv"], "--output-subdir") == (
        "guard_confirm8_<candidate-policy-id>"
    )
    assert "final_selector_command" in segment
    assert "publish_selector_command" in segment


def test_guarded_league_bootstrap_publish_helpers_preserve_checkpoint_records(tmp_path: Path) -> None:
    from weiss_rl.experiments import guarded_league_bootstrap, guarded_league_bootstrap_publish

    assert (
        guarded_league_bootstrap.selected_snapshot_policy_id
        is guarded_league_bootstrap_publish.selected_snapshot_policy_id
    )
    assert (
        guarded_league_bootstrap.resolve_selected_snapshot_checkpoint
        is guarded_league_bootstrap_publish.resolve_selected_snapshot_checkpoint
    )
    assert (
        guarded_league_bootstrap.record_selected_checkpoint
        is guarded_league_bootstrap_publish.record_selected_checkpoint
    )
    assert (
        guarded_league_bootstrap.populate_published_segment_record
        is guarded_league_bootstrap_publish.populate_published_segment_record
    )

    run_dir = tmp_path / "runs" / "seg"
    _write_registry(run_dir, snapshots=[("policy_000001", 5), ("main_league_selected", 10)])

    assert (
        guarded_league_bootstrap_publish.selected_snapshot_policy_id(
            selected={"snapshot_policy_id": " policy_000001 "},
            selection_json=tmp_path / "selection.json",
        )
        == "policy_000001"
    )
    with pytest.raises(RuntimeError, match="candidate selector did not record a snapshot_policy_id"):
        guarded_league_bootstrap_publish.selected_snapshot_policy_id(
            selected={},
            selection_json=tmp_path / "selection.json",
        )

    selected_checkpoint = guarded_league_bootstrap_publish.resolve_selected_snapshot_checkpoint(
        selected={"snapshot_policy_id": "policy_000001"},
        selection_json=tmp_path / "selection.json",
        run_dir=run_dir,
    )
    assert selected_checkpoint == run_dir / "training" / "checkpoints" / "checkpoint_5.pt"

    segment_record: dict[str, object] = {}
    guarded_league_bootstrap_publish.record_selected_checkpoint(
        segment_record=segment_record,
        selected_checkpoint=selected_checkpoint,
        repo_root=tmp_path,
    )
    assert segment_record == {"selected_checkpoint": "runs/seg/training/checkpoints/checkpoint_5.pt"}

    published_checkpoint = guarded_league_bootstrap_publish.resolve_selected_snapshot_checkpoint(
        selected={"snapshot_policy_id": "main_league_selected"},
        selection_json=tmp_path / "published.json",
        run_dir=run_dir,
    )
    guarded_league_bootstrap_publish.populate_published_segment_record(
        segment_record=segment_record,
        published_selected={"snapshot_policy_id": "policy_000001", "eligible": True},
        selected_alias_policy_id="main_league_selected",
        selected_checkpoint=published_checkpoint,
        repo_root=tmp_path,
    )
    assert segment_record == {
        "published_selected": {"snapshot_policy_id": "policy_000001", "eligible": True},
        "selected_alias_policy_id": "main_league_selected",
        "selected_checkpoint": "runs/seg/training/checkpoints/checkpoint_10.pt",
        "status": "accepted",
    }


def test_latest_policy_snapshot_ignores_imported_seed_aliases(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "segment"
    _write_registry(
        run_dir,
        snapshots=[
            ("seed_imported_selected", 0),
            ("policy_000001", 5),
            ("policy_000002", 10),
        ],
    )

    latest = latest_policy_snapshot(run_dir)

    assert latest.policy_id == "policy_000002"
    assert latest.update == 10
    assert latest.checkpoint_path == run_dir / "training" / "checkpoints" / "checkpoint_10.pt"


def test_recent_policy_snapshots_returns_chronological_train_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "segment"
    _write_registry(
        run_dir,
        snapshots=[
            ("policy_000001", 1),
            ("seed_imported_policy_000009", 99),
            ("policy_000002", 2),
            ("policy_000003", 3),
        ],
    )

    recent = recent_policy_snapshots(run_dir, count=2)

    assert [candidate.policy_id for candidate in recent] == ["policy_000002", "policy_000003"]
    assert [candidate.update for candidate in recent] == [2, 3]


def test_evaluate_guard_rejects_anchor_below_reference_drop() -> None:
    guard = evaluate_guard(
        scores={
            "B2 HeuristicPublic": 0.59,
            "B3 HeuristicPublicAggro": 0.49,
            "B4 HeuristicPublicControl": 0.55,
        },
        required_anchors=("B2 HeuristicPublic", "B3 HeuristicPublicAggro", "B4 HeuristicPublicControl"),
        min_required_anchor_score=0.5,
        reference_anchor_scores={"B3 HeuristicPublicAggro": 0.57},
        max_reference_drop=0.04,
    )

    assert guard["passed"] is False
    assert guard["failures"] == [
        {
            "anchor": "B3 HeuristicPublicAggro",
            "reason": "below_min_required_anchor_score",
            "score": 0.49,
            "threshold": 0.5,
        },
        {
            "anchor": "B3 HeuristicPublicAggro",
            "reason": "below_reference_drop_limit",
            "score": 0.49,
            "reference": 0.57,
            "delta": -0.07999999999999996,
            "threshold": -0.04,
        },
    ]


def test_guarded_league_bootstrap_dry_run_writes_train_and_confirm_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_demo",
            segments=3,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            first_init_schedule_offset_updates=0,
            dry_run=True,
        )
    )

    assert summary["status"] == "planned"
    assert len(summary["segments"]) == 1
    segment = summary["segments"][0]
    assert segment["source_checkpoint"].endswith("runs/seed/training/checkpoints/checkpoint_25.pt")
    assert _option(segment["train_command"]["argv"], "--run-label") == "guarded_demo_seg01"
    assert _option(segment["train_command"]["argv"], "--b1-baseline-run-dir").endswith("runs/seed")
    assert _option(segment["train_command"]["argv"], "--init-schedule-offset-updates") == "0"
    assert "preselect_command" in segment
    assert "targeted_confirm_command_template" in segment
    assert "final_selector_command" in segment
    assert "publish_selector_command" in segment
    written = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert written["status"] == "planned"


def test_guarded_league_bootstrap_can_use_separate_b1_baseline_run_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    b1_run_dir = tmp_path / "runs" / "locked_b1"

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            b1_baseline_run_dir=b1_run_dir,
            run_prefix="guarded_separate_b1",
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            dry_run=True,
        )
    )

    segment = summary["segments"][0]
    assert summary["b1_baseline_run_dir"].endswith("runs/locked_b1")
    assert _option(segment["train_command"]["argv"], "--seed-snapshot-run-dir").endswith("runs/seed")
    assert _option(segment["train_command"]["argv"], "--b1-baseline-run-dir").endswith("runs/locked_b1")


def test_targeted_confirm_command_uses_separate_b1_baseline_run_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    segment_run_dir = tmp_path / "runs" / "segment"
    b1_run_dir = tmp_path / "runs" / "locked_b1"

    command = build_targeted_confirm_command(
        config=GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=tmp_path / "runs" / "seed",
            b1_baseline_run_dir=b1_run_dir,
            confirm_paired_seeds=64,
        ),
        run_dir=segment_run_dir,
        focal_policy_id="policy_000003",
        output_subdir="confirm_policy_000003",
    )

    assert _option(command, "--run-dir").endswith("runs/segment")
    assert _option(command, "--b1-baseline-run-dir").endswith("runs/locked_b1")


def test_targeted_confirm_command_includes_learned_guard_opponents(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    segment_run_dir = tmp_path / "runs" / "segment"

    command = build_targeted_confirm_command(
        config=GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=tmp_path / "runs" / "seed",
            confirm_opponents=("B1 NoLeague baseline", "B2 HeuristicPublic", "seed_champion"),
            learned_guard_opponents=("seed_champion", "seed_hard_negative"),
        ),
        run_dir=segment_run_dir,
        focal_policy_id="policy_000003",
        output_subdir="confirm_policy_000003",
    )

    opponents = [command[index + 1] for index, value in enumerate(command) if value == "--opponent"]
    assert opponents == [
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "seed_champion",
        "seed_hard_negative",
    ]


def test_guarded_league_bootstrap_stops_when_b3_guard_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((command, _.get("env")))
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_json(
                output_json,
                {
                    "selected": {
                        "run_dir": run_dir.as_posix(),
                        "run_name": run_dir.name,
                        "snapshot_policy_id": "policy_000001",
                        "update_count": 5,
                        "eligible": True,
                        "selection_score": 0.50,
                        "selection_anchor_scores": {
                            "B1 NoLeague baseline": 0.55,
                            "B2 HeuristicPublic": 0.58,
                            "B3 HeuristicPublicAggro": 0.50,
                            "B4 HeuristicPublicControl": 0.56,
                        },
                    }
                },
            )
        elif "weiss_rl.eval.targeted_confirm_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.55},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.58},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.45},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.56},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_fail",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=4,
            reference_anchor_scores={"B3 HeuristicPublicAggro": 0.55},
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "stopped_guard_failed"
    assert len(summary["segments"]) == 1
    guard = summary["segments"][0]["guard"]
    assert guard["passed"] is False
    assert guard["failures"][0]["anchor"] == "B3 HeuristicPublicAggro"


def test_guarded_league_bootstrap_stops_when_learned_multiobjective_guard_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    reference_summary = tmp_path / "diagnostics" / "selected_reference_confirm4_summary.json"
    _write_json(
        reference_summary,
        {
            "rows": [
                *[
                    {
                        "opponent_policy_id": opponent,
                        "wins": 5,
                        "games": 8,
                        "mean": 5 / 8,
                    }
                    for opponent in FIXED_THESIS_OPPONENTS
                ],
                {"opponent_policy_id": "seed_champion", "wins": 5, "games": 8, "mean": 5 / 8},
                {"opponent_policy_id": "seed_hard_negative", "wins": 5, "games": 8, "mean": 5 / 8},
            ]
        },
    )
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((command, _.get("env")))
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_json(
                output_json,
                {
                    "selected": {
                        "run_dir": run_dir.as_posix(),
                        "run_name": run_dir.name,
                        "snapshot_policy_id": "policy_000001",
                        "update_count": 5,
                        "eligible": True,
                        "selection_score": 0.64,
                        "selection_anchor_scores": {
                            "B1 NoLeague baseline": 0.64,
                            "B2 HeuristicPublic": 0.64,
                            "B3 HeuristicPublicAggro": 0.64,
                            "B4 HeuristicPublicControl": 0.64,
                        },
                    }
                },
            )
        elif "weiss_rl.eval.targeted_confirm_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "paired_seeds": paired_seeds,
                    "rows": [
                        *[
                            {
                                "opponent_policy_id": opponent,
                                "wins": 6,
                                "games": 8,
                                "mean": 6 / 8,
                            }
                            for opponent in FIXED_THESIS_OPPONENTS
                        ],
                        {"opponent_policy_id": "seed_champion", "wins": 4, "games": 8, "mean": 4 / 8},
                        {"opponent_policy_id": "seed_hard_negative", "wins": 4, "games": 8, "mean": 4 / 8},
                    ],
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_learned_gate_fail",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=4,
            confirm_opponents=(*FIXED_THESIS_OPPONENTS, "seed_champion", "seed_hard_negative"),
            publish_min_confirm_paired_seeds=256,
            continue_unpublished_confirmed=True,
            multiobjective_reference_summary_jsons=(reference_summary,),
            min_learned_guard_reference_delta=0.0,
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "stopped_multiobjective_guard_failed"
    assert len(summary["segments"]) == 1
    segment = summary["segments"][0]
    assert segment["status"] == "rejected"
    assert "publish_skipped" not in segment
    assert summary["learned_guard_opponents"] == ["seed_champion", "seed_hard_negative"]
    assert summary["learned_guard_opponents_inferred"] is True
    assert segment["multiobjective_guard"]["passed"] is False
    assert segment["multiobjective_guard"]["groups"]["fixed_baselines"]["mean"] == pytest.approx(0.75)
    assert segment["multiobjective_guard"]["groups"]["learned_opponents"]["reference_delta"] == pytest.approx(-0.125)
    train_commands = [command for command, _env in calls if "weiss_rl.training.train_entrypoint" in command]
    train_envs = [env for command, env in calls if "weiss_rl.training.train_entrypoint" in command]
    assert len(train_commands) == 1
    assert train_envs[0] is not None
    assert train_envs[0]["PYTHONHASHSEED"] == "0"


def test_guarded_league_bootstrap_writes_multiobjective_guard_when_legacy_guard_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    reference_summary = tmp_path / "diagnostics" / "selected_reference_confirm4_summary.json"
    _write_json(
        reference_summary,
        {
            "rows": [
                *[
                    {
                        "opponent_policy_id": opponent,
                        "wins": 5,
                        "games": 8,
                        "mean": 5 / 8,
                    }
                    for opponent in FIXED_THESIS_OPPONENTS
                ],
                {"opponent_policy_id": "seed_champion", "wins": 5, "games": 8, "mean": 5 / 8},
            ]
        },
    )

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_json(
                output_json,
                {
                    "selected": {
                        "run_dir": run_dir.as_posix(),
                        "run_name": run_dir.name,
                        "snapshot_policy_id": "policy_000001",
                        "update_count": 5,
                        "eligible": True,
                        "selection_score": 0.62,
                        "selection_anchor_scores": {
                            "B1 NoLeague baseline": 0.62,
                            "B2 HeuristicPublic": 0.62,
                            "B3 HeuristicPublicAggro": 0.62,
                            "B4 HeuristicPublicControl": 0.62,
                        },
                    }
                },
            )
        elif "weiss_rl.eval.targeted_confirm_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        *[
                            {
                                "opponent_policy_id": opponent,
                                "wins": 6,
                                "games": 8,
                                "mean": 6 / 8,
                            }
                            for opponent in FIXED_THESIS_OPPONENTS
                        ],
                        {"opponent_policy_id": "seed_champion", "wins": 6, "games": 8, "mean": 6 / 8},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_legacy_fail_keeps_multiobjective",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=4,
            reference_anchor_scores={"B4 HeuristicPublicControl": 0.70},
            multiobjective_reference_summary_jsons=(reference_summary,),
            learned_guard_opponents=("seed_champion",),
            min_learned_guard_reference_delta=0.0,
        ),
        runner=fake_runner,
    )

    segment = summary["segments"][0]
    assert summary["status"] == "stopped_guard_failed"
    assert segment["status"] == "rejected"
    assert segment["guard"]["passed"] is False
    assert segment["multiobjective_guard"]["passed"] is True
    assert segment["multiobjective_guard"]["groups"]["learned_opponents"]["reference_delta"] == pytest.approx(0.125)
    assert Path(tmp_path / segment["multiobjective_guard_json"]).is_file()


def test_guarded_league_bootstrap_advances_checkpoint_after_guard_passes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5), ("policy_000002", 10)])
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_selection(output_json, run_dir=run_dir, policy_id="policy_000001", update=5)
            if "--publish-selected-alias" in command:
                _write_registry(
                    run_dir,
                    snapshots=[
                        ("policy_000001", 5),
                        ("policy_000002", 10),
                        ("main_league_selected", 5),
                    ],
                )
        elif "weiss_rl.eval.targeted_confirm_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.55},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.59},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.56},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.57},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_pass",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            first_init_schedule_offset_updates=0,
            confirm_paired_seeds=4,
            publish_min_confirm_paired_seeds=4,
            reference_anchor_scores={"B3 HeuristicPublicAggro": 0.55},
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "completed"
    assert [segment["status"] for segment in summary["segments"]] == ["accepted", "accepted"]
    assert summary["segments"][0]["latest_policy_id"] == "policy_000002"
    assert summary["segments"][0]["selected"]["snapshot_policy_id"] == "policy_000001"
    assert summary["segments"][1]["source_checkpoint"].endswith(
        "runs/guarded_pass_seg01/training/checkpoints/checkpoint_5.pt"
    )
    train_commands = [command for command in calls if "weiss_rl.training.train_entrypoint" in command]
    confirm_commands = [command for command in calls if "weiss_rl.eval.targeted_confirm_entrypoint" in command]
    selector_commands = [
        command for command in calls if "weiss_rl.experiments.select_b1_candidate_entrypoint" in command
    ]
    assert len(train_commands) == 2
    assert len(confirm_commands) == 2
    assert len(selector_commands) == 6
    assert _option(train_commands[0], "--init-schedule-offset-updates") == "0"
    assert "--init-schedule-offset-updates" not in train_commands[1]
    assert _option(train_commands[1], "--init-from-checkpoint").endswith(
        "runs/guarded_pass_seg01/training/checkpoints/checkpoint_5.pt"
    )
    assert _option(confirm_commands[0], "--focal-policy-id") == "policy_000001"


def test_guarded_league_bootstrap_confirms_recent_candidates_before_selecting(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(
                run_dir,
                snapshots=[
                    ("policy_000001", 1),
                    ("policy_000002", 2),
                    ("policy_000003", 3),
                ],
            )
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            if output_json.name.endswith("_candidate_preconfirm.json"):
                _write_json(output_json, {"ranked_candidates": []})
            else:
                _write_selection(output_json, run_dir=run_dir, policy_id="policy_000002", update=2, score=0.64)
        elif "weiss_rl.eval.targeted_confirm_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.62},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.63},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.61},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_recent_candidates",
            segments=1,
            runtime=LeagueSegmentRuntime(segment_updates=3, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=64,
            confirm_recent_candidate_count=3,
            continue_unpublished_confirmed=True,
        ),
        runner=fake_runner,
    )

    segment = summary["segments"][0]
    confirm_commands = [command for command in calls if "weiss_rl.eval.targeted_confirm_entrypoint" in command]
    assert [_option(command, "--focal-policy-id") for command in confirm_commands] == [
        "policy_000001",
        "policy_000002",
        "policy_000003",
    ]
    assert segment["confirm_focal_policy_ids"] == ["policy_000001", "policy_000002", "policy_000003"]
    assert segment["selected"]["snapshot_policy_id"] == "policy_000002"
    assert segment["selected_checkpoint"].endswith(
        "runs/guarded_recent_candidates_seg01/training/checkpoints/checkpoint_2.pt"
    )
    assert summary["status"] == "completed_unpublished_confirmation_insufficient"


def test_guarded_league_bootstrap_does_not_publish_when_confirmation_below_publish_floor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_selection(output_json, run_dir=run_dir, policy_id="policy_000001", update=5)
        elif "weiss_rl.eval.targeted_confirm_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.62},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.63},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.61},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_unpublished",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=64,
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "stopped_publish_confirmation_insufficient"
    assert summary["segments"][0]["status"] == "accepted_unpublished"
    assert summary["segments"][0]["publish_skipped"] == {
        "reason": "confirmation_seed_count_below_publish_minimum",
        "confirm_paired_seeds": 64,
        "publish_min_confirm_paired_seeds": 256,
        "continued_without_publish": False,
    }
    selector_commands = [
        command for command in calls if "weiss_rl.experiments.select_b1_candidate_entrypoint" in command
    ]
    assert len(selector_commands) == 2
    assert all("--publish-selected-alias" not in command for command in selector_commands)


def test_guarded_league_bootstrap_can_continue_unpublished_confirmed_segments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "seed" / "training" / "checkpoints" / "checkpoint_25.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    seed_run_dir = tmp_path / "runs" / "seed"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append(command)
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, snapshots=[("policy_000001", 5)])
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_json = tmp_path / _option(command, "--output-json")
            _write_selection(output_json, run_dir=run_dir, policy_id="policy_000001", update=5)
        elif "weiss_rl.eval.targeted_confirm_entrypoint" in command:
            run_dir = tmp_path / _option(command, "--run-dir")
            output_subdir = _option(command, "--output-subdir")
            paired_seeds = int(_option(command, "--paired-seeds"))
            _write_json(
                run_dir / "eval" / output_subdir / f"targeted_confirm{paired_seeds}_summary.json",
                {
                    "rows": [
                        {"opponent_policy_id": "B1 NoLeague baseline", "mean": 0.62},
                        {"opponent_policy_id": "B2 HeuristicPublic", "mean": 0.64},
                        {"opponent_policy_id": "B3 HeuristicPublicAggro", "mean": 0.63},
                        {"opponent_policy_id": "B4 HeuristicPublicControl", "mean": 0.61},
                    ]
                },
            )
        return subprocess.CompletedProcess(command, 0)

    summary = run_guarded_league_bootstrap(
        GuardedLeagueBootstrapConfig(
            repo_root=tmp_path,
            init_checkpoint_path=checkpoint,
            seed_snapshot_run_dir=seed_run_dir,
            run_prefix="guarded_unpublished_continue",
            segments=2,
            runtime=LeagueSegmentRuntime(segment_updates=5, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=64,
            continue_unpublished_confirmed=True,
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "completed_unpublished_confirmation_insufficient"
    assert [segment["status"] for segment in summary["segments"]] == ["accepted_unpublished", "accepted_unpublished"]
    assert all(segment["publish_skipped"]["continued_without_publish"] is True for segment in summary["segments"])
    assert summary["segments"][1]["source_checkpoint"].endswith(
        "runs/guarded_unpublished_continue_seg01/training/checkpoints/checkpoint_5.pt"
    )
    train_commands = [command for command in calls if "weiss_rl.training.train_entrypoint" in command]
    confirm_commands = [command for command in calls if "weiss_rl.eval.targeted_confirm_entrypoint" in command]
    selector_commands = [
        command for command in calls if "weiss_rl.experiments.select_b1_candidate_entrypoint" in command
    ]
    assert len(train_commands) == 2
    assert len(confirm_commands) == 2
    assert len(selector_commands) == 4
    assert all("--publish-selected-alias" not in command for command in selector_commands)


def test_guarded_league_bootstrap_script_rejects_seed_opponents_with_pinned_import_defaults(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "python/scripts/guarded_league_bootstrap.py",
            "--init-from-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--seed-snapshot-run-dir",
            str(tmp_path / "seed_run"),
            "--required-anchor",
            "seed_example_policy",
            "--confirm-opponent",
            "seed_example_policy",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--use-stack-seed-snapshot-policy" in result.stderr
