from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.bootstrap_commands import (
    command_record,
    fixed_hash_seed_env,
    resolve_snapshot_checkpoint_path,
    run_command,
)
from weiss_rl.experiments.segmented_b1_guided_bootstrap_outcomes import (
    populate_completed_segment_record,
    read_segment_selection_result,
    selected_candidate,
    stop_decision,
)
from weiss_rl.experiments.segmented_b1_guided_bootstrap_plan import (
    SegmentedBootstrapConfigProtocol,
    build_segment_record,
    build_selector_command,
    build_targeted_confirm_command,
    build_train_segment_command,
    populate_dry_run_segment_plan,
)

CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True, slots=True)
class SegmentedBootstrapStepResult:
    segment_record: dict[str, Any]
    next_run_dir: Path
    next_seed_run_dir: Path
    next_policy_id: str
    next_previous_score: float | None
    terminal_status: str | None = None
    terminal_reason: str | None = None

    @property
    def should_stop(self) -> bool:
        return self.terminal_status is not None


def run_segmented_bootstrap_step(
    *,
    config: SegmentedBootstrapConfigProtocol,
    repo_root: Path,
    diagnostics_dir: Path,
    segment_index: int,
    current_run_dir: Path,
    current_seed_run_dir: Path,
    current_policy_id: str,
    previous_score: float | None,
    runner: CommandRunner,
) -> SegmentedBootstrapStepResult:
    segment_run_label = f"{config.run_prefix}_seg{segment_index:02d}"
    segment_run_dir = repo_root / "runs" / segment_run_label
    init_checkpoint_path = resolve_snapshot_checkpoint_path(
        run_dir=current_run_dir,
        policy_id=current_policy_id,
    )
    train_command = build_train_segment_command(
        config=config,
        segment_run_label=segment_run_label,
        init_checkpoint_path=init_checkpoint_path,
        seed_run_dir=current_seed_run_dir,
    )
    preselect_json = diagnostics_dir / f"{segment_run_label}_candidate_preconfirm.json"
    final_json = diagnostics_dir / f"{segment_run_label}_candidate_selection.json"
    segment_record = build_segment_record(
        config=config,
        repo_root=repo_root,
        segment_index=segment_index,
        segment_run_label=segment_run_label,
        segment_run_dir=segment_run_dir,
        source_run_dir=current_run_dir,
        source_policy_id=current_policy_id,
        source_checkpoint_path=init_checkpoint_path,
        seed_run_dir=current_seed_run_dir,
        train_command=train_command,
        preselect_json=preselect_json,
        final_json=final_json,
    )

    if config.dry_run:
        populate_dry_run_segment_plan(
            segment_record,
            config=config,
            segment_run_dir=segment_run_dir,
            preselect_json=preselect_json,
            final_json=final_json,
        )
        return SegmentedBootstrapStepResult(
            segment_record=segment_record,
            next_run_dir=current_run_dir,
            next_seed_run_dir=current_seed_run_dir,
            next_policy_id=current_policy_id,
            next_previous_score=previous_score,
            terminal_status="planned",
        )

    run_command(train_command, cwd=repo_root, runner=runner, env=fixed_hash_seed_env())
    preselect_command = build_selector_command(
        config=config,
        run_dir=segment_run_dir,
        output_json=preselect_json,
        publish_alias=False,
    )
    segment_record["preselect_command"] = command_record(preselect_command)
    run_command(preselect_command, cwd=repo_root, runner=runner)

    preselected = selected_candidate(preselect_json)
    focal_policy_id = str(preselected.get("snapshot_policy_id", "")).strip()
    if not focal_policy_id:
        raise RuntimeError(f"selected candidate has no snapshot_policy_id: {preselect_json}")
    confirm_subdir = f"segmented_confirm{int(config.confirm_paired_seeds)}_{focal_policy_id}"
    confirm_command = build_targeted_confirm_command(
        config=config,
        run_dir=segment_run_dir,
        focal_policy_id=focal_policy_id,
        output_subdir=confirm_subdir,
    )
    segment_record["targeted_confirm_command"] = command_record(confirm_command)
    run_command(confirm_command, cwd=repo_root, runner=runner, env=fixed_hash_seed_env())

    final_selector_command = build_selector_command(
        config=config,
        run_dir=segment_run_dir,
        output_json=final_json,
        publish_alias=True,
    )
    segment_record["final_selector_command"] = command_record(final_selector_command)
    run_command(final_selector_command, cwd=repo_root, runner=runner)

    selection_result = read_segment_selection_result(final_json, previous_score=previous_score)
    populate_completed_segment_record(segment_record, selection_result)
    decision = stop_decision(
        selection_result,
        max_selected_drop=config.max_selected_drop,
        stop_on_latest_falloff=config.stop_on_latest_falloff,
        max_latest_drop=config.max_latest_drop,
    )
    if decision.should_stop:
        return SegmentedBootstrapStepResult(
            segment_record=segment_record,
            next_run_dir=current_run_dir,
            next_seed_run_dir=current_seed_run_dir,
            next_policy_id=current_policy_id,
            next_previous_score=previous_score,
            terminal_status=decision.status,
            terminal_reason=decision.stop_reason,
        )

    return SegmentedBootstrapStepResult(
        segment_record=segment_record,
        next_run_dir=segment_run_dir,
        next_seed_run_dir=segment_run_dir,
        next_policy_id=config.alias_policy_id,
        next_previous_score=selection_result.selection_score,
    )
