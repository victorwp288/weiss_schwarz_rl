from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_selection import DEFAULT_REQUIRED_ANCHORS
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_NAME
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
    SegmentStopOutcome,
    apply_segment_stop_outcome,
    publish_confirmation_skip_payload,
    rejected_segment_outcome,
    unpublished_confirmation_stop_outcome,
)
from weiss_rl.experiments.guarded_league_bootstrap_publish import (
    populate_published_segment_record,
    record_selected_checkpoint,
    resolve_selected_snapshot_checkpoint,
    selected_snapshot_policy_id,
)
from weiss_rl.experiments.guarded_league_bootstrap_segment_runner import (
    GuardedLeagueSegmentStepResult,
    run_guarded_league_segment_step,
)
from weiss_rl.experiments.guarded_league_bootstrap_segments import (
    GuardedLeagueSegmentPlan,
    build_initial_segment_record,
    build_segment_plan,
    build_selector_command,
    build_targeted_confirm_command,
    build_targeted_confirm_record,
    build_train_segment_command,
    populate_dry_run_segment_record,
    targeted_confirm_command_payloads,
)
from weiss_rl.experiments.guarded_league_bootstrap_segments import (
    effective_confirm_opponents as _effective_confirm_opponents,
)
from weiss_rl.experiments.guarded_league_bootstrap_segments import (
    effective_learned_guard_opponents as _effective_learned_guard_opponents,
)
from weiss_rl.experiments.guarded_league_bootstrap_segments import (
    is_seed_wrapped_suffix_match as _is_seed_wrapped_suffix_match,
)
from weiss_rl.experiments.guarded_league_bootstrap_selection import (
    SnapshotCandidate,
    evaluate_guard,
    evaluate_multiobjective_guard,
    latest_policy_snapshot,
    load_reference_scores_or_empty,
    load_targeted_confirm_scores,
    policy_snapshots,
    recent_policy_snapshots,
    selected_candidate,
    selected_candidate_or_none,
    selection_anchor_scores,
    targeted_confirm_summary_path,
)
from weiss_rl.experiments.guarded_league_bootstrap_selection import (
    resolve_repo_path as _resolve_repo_path,
)
from weiss_rl.experiments.guarded_league_bootstrap_selection import (
    selected_confirm_summary_path as _selected_confirm_summary_path,
)
from weiss_rl.experiments.guarded_league_bootstrap_summary import (
    build_guarded_league_bootstrap_summary,
    guarded_bootstrap_summary_path,
    write_guarded_league_bootstrap_summary,
)
from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS

DEFAULT_STACK_CONFIG = Path(
    "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
)
DEFAULT_RUN_PREFIX = "guarded_league_bootstrap"
DEFAULT_SELECTED_ALIAS_POLICY_ID = "main_league_selected"
DEFAULT_CONFIRM_OPPONENTS = (NOLEAGUE_BASELINE_NAME, *DEFAULT_REQUIRED_ANCHORS)
DEFAULT_RUNTIME_OVERRIDES = (
    "league.pool.seed_snapshot_champion_import=pinned",
    "league.pool.seed_snapshot_import_filter=pinned",
)

__all__ = [
    "DEFAULT_CONFIRM_OPPONENTS",
    "DEFAULT_RUN_PREFIX",
    "DEFAULT_RUNTIME_OVERRIDES",
    "DEFAULT_SELECTED_ALIAS_POLICY_ID",
    "DEFAULT_STACK_CONFIG",
    "FIXED_THESIS_OPPONENTS",
    "GuardedLeagueBootstrapConfig",
    "GuardedLeagueSegmentPlan",
    "GuardedLeagueSegmentStepResult",
    "LeagueSegmentRuntime",
    "SegmentStopOutcome",
    "SnapshotCandidate",
    "_effective_confirm_opponents",
    "_effective_learned_guard_opponents",
    "_is_seed_wrapped_suffix_match",
    "_resolve_repo_path",
    "_selected_confirm_summary_path",
    "build_initial_segment_record",
    "build_guarded_league_bootstrap_summary",
    "build_selector_command",
    "build_segment_plan",
    "build_targeted_confirm_command",
    "build_targeted_confirm_record",
    "build_train_segment_command",
    "confirm_focal_policy_ids",
    "evaluate_selected_multiobjective_guard",
    "evaluate_guard",
    "evaluate_multiobjective_guard",
    "apply_segment_stop_outcome",
    "latest_policy_snapshot",
    "load_reference_scores_or_empty",
    "load_targeted_confirm_scores",
    "policy_snapshots",
    "populate_confirm_candidate_segment_record",
    "populate_dry_run_segment_record",
    "populate_published_segment_record",
    "publish_confirmation_skip_payload",
    "populate_targeted_confirm_segment_record",
    "recent_policy_snapshots",
    "rejected_segment_outcome",
    "run_guarded_league_bootstrap",
    "runtime_overrides_with_defaults",
    "run_guarded_league_segment_step",
    "record_targeted_confirm_result",
    "record_selected_checkpoint",
    "resolve_selected_snapshot_checkpoint",
    "selected_candidate",
    "selected_anchor_scores",
    "selected_candidate_or_none",
    "selected_snapshot_policy_id",
    "populate_selected_segment_record",
    "selection_anchor_scores",
    "targeted_confirm_command_payloads",
    "targeted_confirm_summary_path",
    "unpublished_confirmation_stop_outcome",
    "write_multiobjective_guard_artifact",
    "guarded_bootstrap_summary_path",
    "write_guarded_league_bootstrap_summary",
]


def runtime_overrides_with_defaults(
    overrides: Sequence[str] | None,
    *,
    apply_seed_snapshot_defaults: bool = True,
) -> tuple[str, ...]:
    requested = tuple(str(override) for override in (overrides or ()))
    requested_keys = {override.split("=", 1)[0].strip() for override in requested}
    defaults = (
        tuple(
            override
            for override in DEFAULT_RUNTIME_OVERRIDES
            if override.split("=", 1)[0].strip() not in requested_keys
        )
        if apply_seed_snapshot_defaults
        else ()
    )
    return (*defaults, *requested)


@dataclass(frozen=True, slots=True)
class LeagueSegmentRuntime:
    num_envs: int = 288
    unroll_length: int = 64
    segment_updates: int = 10
    runtime_mode: str = "train_async_fast"
    simulator_profile: str = "fast"
    device: str = "cuda"
    checkpoint_interval_updates: int = 5
    collection_backend: str = "process"
    profile_timers: bool = True
    overrides: tuple[str, ...] = DEFAULT_RUNTIME_OVERRIDES


@dataclass(frozen=True, slots=True)
class GuardedLeagueBootstrapConfig:
    repo_root: Path
    init_checkpoint_path: Path
    seed_snapshot_run_dir: Path
    b1_baseline_run_dir: Path | None = None
    run_prefix: str = DEFAULT_RUN_PREFIX
    stack_config: Path = DEFAULT_STACK_CONFIG
    segments: int = 4
    runtime: LeagueSegmentRuntime = LeagueSegmentRuntime()
    first_init_schedule_offset_updates: int | None = None
    confirm_paired_seeds: int = 64
    publish_min_confirm_paired_seeds: int = 256
    confirm_recent_candidate_count: int = 1
    bootstrap_samples: int = 2000
    required_anchors: tuple[str, ...] = DEFAULT_REQUIRED_ANCHORS
    confirm_opponents: tuple[str, ...] = DEFAULT_CONFIRM_OPPONENTS
    min_required_anchor_score: float = 0.5
    reference_anchor_scores: Mapping[str, float] = field(default_factory=dict)
    multiobjective_reference_summary_jsons: tuple[Path, ...] = ()
    multiobjective_fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    learned_guard_opponents: tuple[str, ...] = ()
    min_multiobjective_fixed_score: float = 0.5
    max_multiobjective_fixed_reference_drop: float = 0.0
    min_learned_guard_score: float = 0.5
    min_learned_guard_mean: float = 0.5
    min_learned_guard_reference_delta: float | None = 0.0
    max_learned_guard_reference_drop: float | None = None
    reference_label: str = "reference"
    max_reference_drop: float = 0.04
    selected_alias_policy_id: str = DEFAULT_SELECTED_ALIAS_POLICY_ID
    continue_unpublished_confirmed: bool = False
    dry_run: bool = False


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def run_guarded_league_bootstrap(
    config: GuardedLeagueBootstrapConfig,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    repo_root = config.repo_root.resolve()
    diagnostics_dir = repo_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = guarded_bootstrap_summary_path(diagnostics_dir=diagnostics_dir, run_prefix=config.run_prefix)
    current_checkpoint = config.init_checkpoint_path.resolve()
    effective_learned_guard_opponents = _effective_learned_guard_opponents(config)
    summary = build_guarded_league_bootstrap_summary(
        config=config,
        repo_root=repo_root,
        effective_learned_guard_opponents=effective_learned_guard_opponents,
    )

    for segment_index in range(1, int(config.segments) + 1):
        segment_result = run_guarded_league_segment_step(
            config=config,
            repo_root=repo_root,
            diagnostics_dir=diagnostics_dir,
            segment_index=segment_index,
            current_checkpoint=current_checkpoint,
            summary=summary,
            effective_learned_guard_opponents=effective_learned_guard_opponents,
            runner=runner,
        )
        current_checkpoint = segment_result.current_checkpoint
        if segment_result.stop:
            break

    write_guarded_league_bootstrap_summary(summary=summary, summary_path=summary_path)
    return summary
