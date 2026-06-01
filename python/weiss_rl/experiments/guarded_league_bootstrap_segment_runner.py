from __future__ import annotations

import subprocess
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.bootstrap_commands import command_record, fixed_hash_seed_env, run_command
from weiss_rl.experiments.guarded_league_bootstrap_confirmations import (
    confirm_focal_policy_ids,
    populate_confirm_candidate_segment_record,
    populate_targeted_confirm_segment_record,
    record_targeted_confirm_result,
)
from weiss_rl.experiments.guarded_league_bootstrap_final_selection import (
    evaluate_selected_multiobjective_guard,
    populate_selected_segment_record,
    selected_anchor_scores,
    write_multiobjective_guard_artifact,
)
from weiss_rl.experiments.guarded_league_bootstrap_outcomes import (
    apply_segment_stop_outcome,
    publish_confirmation_skip_payload,
    rejected_segment_outcome,
    unpublished_confirmation_stop_outcome,
)
from weiss_rl.experiments.guarded_league_bootstrap_publish import (
    populate_published_segment_record,
    record_selected_checkpoint,
    resolve_selected_snapshot_checkpoint,
)
from weiss_rl.experiments.guarded_league_bootstrap_segments import (
    build_initial_segment_record,
    build_segment_plan,
    build_selector_command,
    build_targeted_confirm_record,
    populate_dry_run_segment_record,
)
from weiss_rl.experiments.guarded_league_bootstrap_selection import (
    evaluate_guard,
    latest_policy_snapshot,
    recent_policy_snapshots,
    selected_candidate,
    selected_candidate_or_none,
)

CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True, slots=True)
class GuardedLeagueSegmentStepResult:
    current_checkpoint: Path
    stop: bool


def run_guarded_league_segment_step(
    *,
    config: Any,
    repo_root: Path,
    diagnostics_dir: Path,
    segment_index: int,
    current_checkpoint: Path,
    summary: MutableMapping[str, Any],
    effective_learned_guard_opponents: Sequence[str],
    runner: CommandRunner,
) -> GuardedLeagueSegmentStepResult:
    segment_plan = build_segment_plan(
        config=config,
        repo_root=repo_root,
        diagnostics_dir=diagnostics_dir,
        segment_index=segment_index,
        init_checkpoint_path=current_checkpoint,
    )
    segment_record = build_initial_segment_record(
        plan=segment_plan,
        repo_root=repo_root,
        source_checkpoint=current_checkpoint,
    )
    summary["segments"].append(segment_record)

    if config.dry_run:
        populate_dry_run_segment_record(
            record=segment_record,
            config=config,
            plan=segment_plan,
        )
        return GuardedLeagueSegmentStepResult(current_checkpoint=current_checkpoint, stop=True)

    run_command(segment_plan.train_command, cwd=repo_root, runner=runner, env=fixed_hash_seed_env())
    candidate_limit = int(config.confirm_recent_candidate_count)
    recent_candidates = recent_policy_snapshots(segment_plan.run_dir, count=candidate_limit)
    latest = latest_policy_snapshot(segment_plan.run_dir)
    preselect_command = build_selector_command(
        config=config,
        run_dir=segment_plan.run_dir,
        output_json=segment_plan.preselect_json,
        publish_alias=False,
    )
    segment_record["preselect_command"] = command_record(preselect_command)
    run_command(preselect_command, cwd=repo_root, runner=runner)
    preselected = selected_candidate_or_none(segment_plan.preselect_json)
    focal_policy_ids = confirm_focal_policy_ids(
        preselected=preselected,
        recent_candidates=recent_candidates,
        latest_policy_id=latest.policy_id,
        limit=candidate_limit,
    )
    populate_confirm_candidate_segment_record(
        segment_record=segment_record,
        latest=latest,
        repo_root=repo_root,
        confirm_recent_candidate_count=config.confirm_recent_candidate_count,
        focal_policy_ids=focal_policy_ids,
        preselected=preselected,
    )
    confirm_env = fixed_hash_seed_env()
    targeted_confirm_records: list[dict[str, Any]] = []
    for focal_policy_id in focal_policy_ids:
        confirm_record = build_targeted_confirm_record(
            config=config,
            run_dir=segment_plan.run_dir,
            focal_policy_id=focal_policy_id,
        )
        targeted_confirm_records.append(confirm_record)
        confirm_command = list(confirm_record["command"]["argv"])
        run_command(confirm_command, cwd=repo_root, runner=runner, env=confirm_env)
        record_targeted_confirm_result(
            confirm_record=confirm_record,
            run_dir=segment_plan.run_dir,
            paired_seeds=config.confirm_paired_seeds,
            repo_root=repo_root,
        )
    populate_targeted_confirm_segment_record(
        segment_record=segment_record,
        targeted_confirm_records=targeted_confirm_records,
    )
    final_selector_command = build_selector_command(
        config=config,
        run_dir=segment_plan.run_dir,
        output_json=segment_plan.final_selection_json,
        publish_alias=False,
    )
    segment_record["final_selector_command"] = command_record(final_selector_command)
    run_command(final_selector_command, cwd=repo_root, runner=runner)
    final_selected = selected_candidate(segment_plan.final_selection_json)
    selected_confirm_summary_path = str(final_selected.get("selection_confirmation_summary_path") or "").strip()
    scores = selected_anchor_scores(
        final_selected=final_selected,
        selected_confirmation_summary_path=selected_confirm_summary_path,
        targeted_confirm_records=targeted_confirm_records,
    )
    guard = evaluate_guard(
        scores=scores,
        required_anchors=config.required_anchors,
        min_required_anchor_score=config.min_required_anchor_score,
        reference_anchor_scores=config.reference_anchor_scores,
        max_reference_drop=config.max_reference_drop,
    )
    populate_selected_segment_record(
        segment_record=segment_record,
        repo_root=repo_root,
        final_selected=final_selected,
        selected_confirmation_summary_path=selected_confirm_summary_path,
        targeted_confirm_records=targeted_confirm_records,
        anchor_scores=scores,
        guard=guard,
    )
    multiobjective_guard = evaluate_selected_multiobjective_guard(
        config=config,
        repo_root=repo_root,
        selected_confirmation_summary_path=selected_confirm_summary_path,
        targeted_confirm_records=targeted_confirm_records,
        effective_learned_guard_opponents=effective_learned_guard_opponents,
    )
    if multiobjective_guard is not None:
        write_multiobjective_guard_artifact(
            segment_record=segment_record,
            multiobjective_guard=multiobjective_guard,
            diagnostics_dir=diagnostics_dir,
            run_label=segment_plan.run_label,
            repo_root=repo_root,
        )
    rejection = rejected_segment_outcome(
        selected=final_selected,
        guard=guard,
        multiobjective_guard=multiobjective_guard,
    )
    if rejection is not None:
        apply_segment_stop_outcome(
            segment_record=segment_record,
            summary=summary,
            outcome=rejection,
        )
        return GuardedLeagueSegmentStepResult(current_checkpoint=current_checkpoint, stop=True)
    if int(config.confirm_paired_seeds) < int(config.publish_min_confirm_paired_seeds):
        segment_record["status"] = "accepted_unpublished"
        segment_record["publish_skipped"] = publish_confirmation_skip_payload(
            confirm_paired_seeds=config.confirm_paired_seeds,
            publish_min_confirm_paired_seeds=config.publish_min_confirm_paired_seeds,
            continue_unpublished_confirmed=config.continue_unpublished_confirmed,
        )
        if bool(config.continue_unpublished_confirmed):
            selected_checkpoint = resolve_selected_snapshot_checkpoint(
                selected=final_selected,
                selection_json=segment_plan.final_selection_json,
                run_dir=segment_plan.run_dir,
            )
            record_selected_checkpoint(
                segment_record=segment_record,
                selected_checkpoint=selected_checkpoint,
                repo_root=repo_root,
            )
            unpublished_stop = unpublished_confirmation_stop_outcome(
                continue_unpublished_confirmed=config.continue_unpublished_confirmed,
                has_more_segments=int(segment_index) < int(config.segments),
            )
            if unpublished_stop is None:
                return GuardedLeagueSegmentStepResult(current_checkpoint=selected_checkpoint, stop=False)
            apply_segment_stop_outcome(
                segment_record=segment_record,
                summary=summary,
                outcome=unpublished_stop,
            )
            return GuardedLeagueSegmentStepResult(current_checkpoint=selected_checkpoint, stop=True)
        unpublished_stop = unpublished_confirmation_stop_outcome(
            continue_unpublished_confirmed=config.continue_unpublished_confirmed,
            has_more_segments=int(segment_index) < int(config.segments),
        )
        if unpublished_stop is not None:
            apply_segment_stop_outcome(
                segment_record=segment_record,
                summary=summary,
                outcome=unpublished_stop,
            )
        return GuardedLeagueSegmentStepResult(current_checkpoint=current_checkpoint, stop=True)

    publish_selector_command = build_selector_command(
        config=config,
        run_dir=segment_plan.run_dir,
        output_json=segment_plan.publish_selection_json,
        publish_alias=True,
    )
    segment_record["publish_selector_command"] = command_record(publish_selector_command)
    run_command(publish_selector_command, cwd=repo_root, runner=runner)
    published_selected = selected_candidate(segment_plan.publish_selection_json)
    selected_checkpoint = resolve_selected_snapshot_checkpoint(
        selected={"snapshot_policy_id": str(config.selected_alias_policy_id)},
        selection_json=segment_plan.publish_selection_json,
        run_dir=segment_plan.run_dir,
    )
    populate_published_segment_record(
        segment_record=segment_record,
        published_selected=published_selected,
        selected_alias_policy_id=str(config.selected_alias_policy_id),
        selected_checkpoint=selected_checkpoint,
        repo_root=repo_root,
    )
    summary["status"] = "completed"
    return GuardedLeagueSegmentStepResult(current_checkpoint=selected_checkpoint, stop=False)


__all__ = [
    "CommandRunner",
    "GuardedLeagueSegmentStepResult",
    "run_guarded_league_segment_step",
]
