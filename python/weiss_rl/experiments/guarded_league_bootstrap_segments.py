from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.experiments.bootstrap_commands import (
    build_candidate_selector_command,
    build_targeted_confirm_entrypoint_command,
    build_train_entrypoint_command,
    command_record,
    repo_relative,
)


class LeagueSegmentRuntimeLike(Protocol):
    @property
    def num_envs(self) -> int: ...

    @property
    def unroll_length(self) -> int: ...

    @property
    def segment_updates(self) -> int: ...

    @property
    def runtime_mode(self) -> str: ...

    @property
    def simulator_profile(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def checkpoint_interval_updates(self) -> int: ...

    @property
    def collection_backend(self) -> str: ...

    @property
    def profile_timers(self) -> bool: ...

    @property
    def overrides(self) -> tuple[str, ...]: ...


class GuardedLeagueBootstrapCommandConfig(Protocol):
    @property
    def repo_root(self) -> Path: ...

    @property
    def stack_config(self) -> Path: ...

    @property
    def seed_snapshot_run_dir(self) -> Path: ...

    @property
    def b1_baseline_run_dir(self) -> Path | None: ...

    @property
    def run_prefix(self) -> str: ...

    @property
    def runtime(self) -> LeagueSegmentRuntimeLike: ...

    @property
    def first_init_schedule_offset_updates(self) -> int | None: ...

    @property
    def confirm_paired_seeds(self) -> int: ...

    @property
    def confirm_recent_candidate_count(self) -> int: ...

    @property
    def bootstrap_samples(self) -> int: ...

    @property
    def required_anchors(self) -> Sequence[str]: ...

    @property
    def confirm_opponents(self) -> Sequence[str]: ...

    @property
    def min_required_anchor_score(self) -> float: ...

    @property
    def multiobjective_fixed_opponents(self) -> Sequence[str]: ...

    @property
    def learned_guard_opponents(self) -> Sequence[str]: ...

    @property
    def selected_alias_policy_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class GuardedLeagueSegmentPlan:
    segment: int
    run_label: str
    run_dir: Path
    preselect_json: Path
    final_selection_json: Path
    publish_selection_json: Path
    train_command: list[str]


def build_train_segment_command(
    *,
    config: GuardedLeagueBootstrapCommandConfig,
    segment_run_label: str,
    init_checkpoint_path: Path,
    init_schedule_offset_updates: int | None = None,
) -> list[str]:
    runtime = config.runtime
    b1_baseline_run_dir = config.b1_baseline_run_dir or config.seed_snapshot_run_dir
    return build_train_entrypoint_command(
        repo_root=config.repo_root,
        stack_config=config.stack_config,
        run_label=segment_run_label,
        num_envs=runtime.num_envs,
        unroll_length=runtime.unroll_length,
        max_updates=runtime.segment_updates,
        runtime_mode=runtime.runtime_mode,
        simulator_profile=runtime.simulator_profile,
        device=runtime.device,
        checkpoint_interval_updates=runtime.checkpoint_interval_updates,
        seed_snapshot_run_dir=config.seed_snapshot_run_dir,
        init_checkpoint_path=init_checkpoint_path,
        collection_backend=runtime.collection_backend,
        b1_baseline_run_dir=b1_baseline_run_dir,
        init_schedule_offset_updates=init_schedule_offset_updates,
        profile_timers=runtime.profile_timers,
        overrides=runtime.overrides,
    )


def build_selector_command(
    *,
    config: GuardedLeagueBootstrapCommandConfig,
    run_dir: Path,
    output_json: Path,
    publish_alias: bool,
) -> list[str]:
    return build_candidate_selector_command(
        repo_root=config.repo_root,
        stack_config=config.stack_config,
        run_dir=run_dir,
        output_json=output_json,
        min_required_anchor_score=config.min_required_anchor_score,
        confirm_paired_seeds=config.confirm_paired_seeds,
        required_anchors=config.required_anchors,
        confirm_opponents=config.confirm_opponents,
        publish_alias=publish_alias,
        selected_alias_policy_id=config.selected_alias_policy_id,
    )


def build_targeted_confirm_command(
    *,
    config: GuardedLeagueBootstrapCommandConfig,
    run_dir: Path,
    focal_policy_id: str,
    output_subdir: str,
) -> list[str]:
    b1_baseline_run_dir = config.b1_baseline_run_dir or run_dir
    return build_targeted_confirm_entrypoint_command(
        repo_root=config.repo_root,
        stack_config=config.stack_config,
        run_dir=run_dir,
        b1_baseline_run_dir=b1_baseline_run_dir,
        focal_policy_id=focal_policy_id,
        paired_seeds=config.confirm_paired_seeds,
        bootstrap_samples=config.bootstrap_samples,
        output_subdir=output_subdir,
        opponents=effective_confirm_opponents(config),
    )


def effective_confirm_opponents(config: GuardedLeagueBootstrapCommandConfig) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*config.confirm_opponents, *effective_learned_guard_opponents(config)]))


def effective_learned_guard_opponents(config: GuardedLeagueBootstrapCommandConfig) -> tuple[str, ...]:
    configured = tuple(dict.fromkeys(str(opponent) for opponent in config.learned_guard_opponents))
    if configured:
        return configured
    fixed_opponents = tuple(str(opponent) for opponent in config.multiobjective_fixed_opponents)
    return tuple(
        dict.fromkeys(
            opponent
            for opponent in config.confirm_opponents
            if not any(is_seed_wrapped_suffix_match(str(opponent), fixed) for fixed in fixed_opponents)
        )
    )


def is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def segment_run_label(*, config: GuardedLeagueBootstrapCommandConfig, segment_index: int) -> str:
    return f"{config.run_prefix}_seg{int(segment_index):02d}"


def targeted_confirm_output_subdir(*, config: GuardedLeagueBootstrapCommandConfig, focal_policy_id: str) -> str:
    return f"guard_confirm{int(config.confirm_paired_seeds)}_{focal_policy_id}"


def build_segment_plan(
    *,
    config: GuardedLeagueBootstrapCommandConfig,
    repo_root: Path,
    diagnostics_dir: Path,
    segment_index: int,
    init_checkpoint_path: Path,
) -> GuardedLeagueSegmentPlan:
    run_label = segment_run_label(config=config, segment_index=segment_index)
    run_dir = repo_root / "runs" / run_label
    return GuardedLeagueSegmentPlan(
        segment=int(segment_index),
        run_label=run_label,
        run_dir=run_dir,
        preselect_json=diagnostics_dir / f"{run_label}_candidate_preconfirm.json",
        final_selection_json=diagnostics_dir / f"{run_label}_candidate_selection.json",
        publish_selection_json=diagnostics_dir / f"{run_label}_candidate_published.json",
        train_command=build_train_segment_command(
            config=config,
            segment_run_label=run_label,
            init_checkpoint_path=init_checkpoint_path,
            init_schedule_offset_updates=(
                config.first_init_schedule_offset_updates if int(segment_index) == 1 else None
            ),
        ),
    )


def build_initial_segment_record(
    *,
    plan: GuardedLeagueSegmentPlan,
    repo_root: Path,
    source_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "segment": int(plan.segment),
        "run_label": plan.run_label,
        "run_dir": repo_relative(plan.run_dir, repo_root=repo_root).as_posix(),
        "source_checkpoint": repo_relative(source_checkpoint, repo_root=repo_root).as_posix(),
        "train_command": command_record(plan.train_command),
        "preselect_json": repo_relative(plan.preselect_json, repo_root=repo_root).as_posix(),
        "final_selection_json": repo_relative(plan.final_selection_json, repo_root=repo_root).as_posix(),
        "publish_selection_json": repo_relative(plan.publish_selection_json, repo_root=repo_root).as_posix(),
    }


def populate_dry_run_segment_record(
    *,
    record: dict[str, Any],
    config: GuardedLeagueBootstrapCommandConfig,
    plan: GuardedLeagueSegmentPlan,
) -> None:
    record["status"] = "planned"
    record["preselect_command"] = command_record(
        build_selector_command(
            config=config,
            run_dir=plan.run_dir,
            output_json=plan.preselect_json,
            publish_alias=False,
        )
    )
    record["targeted_confirm_command_template"] = command_record(
        build_targeted_confirm_command(
            config=config,
            run_dir=plan.run_dir,
            focal_policy_id="<candidate-policy-id>",
            output_subdir=targeted_confirm_output_subdir(
                config=config,
                focal_policy_id="<candidate-policy-id>",
            ),
        )
    )
    record["confirm_recent_candidate_count"] = int(config.confirm_recent_candidate_count)
    record["final_selector_command"] = command_record(
        build_selector_command(
            config=config,
            run_dir=plan.run_dir,
            output_json=plan.final_selection_json,
            publish_alias=False,
        )
    )
    record["publish_selector_command"] = command_record(
        build_selector_command(
            config=config,
            run_dir=plan.run_dir,
            output_json=plan.publish_selection_json,
            publish_alias=True,
        )
    )


def build_targeted_confirm_record(
    *,
    config: GuardedLeagueBootstrapCommandConfig,
    run_dir: Path,
    focal_policy_id: str,
) -> dict[str, Any]:
    output_subdir = targeted_confirm_output_subdir(config=config, focal_policy_id=focal_policy_id)
    command = build_targeted_confirm_command(
        config=config,
        run_dir=run_dir,
        focal_policy_id=focal_policy_id,
        output_subdir=output_subdir,
    )
    return {
        "focal_policy_id": focal_policy_id,
        "output_subdir": output_subdir,
        "command": command_record(command),
    }


def targeted_confirm_command_payloads(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record["command"]) for record in records if isinstance(record.get("command"), Mapping)]


__all__ = [
    "GuardedLeagueBootstrapCommandConfig",
    "GuardedLeagueSegmentPlan",
    "LeagueSegmentRuntimeLike",
    "build_initial_segment_record",
    "build_segment_plan",
    "build_selector_command",
    "build_targeted_confirm_command",
    "build_targeted_confirm_record",
    "build_train_segment_command",
    "effective_confirm_opponents",
    "effective_learned_guard_opponents",
    "is_seed_wrapped_suffix_match",
    "populate_dry_run_segment_record",
    "segment_run_label",
    "targeted_confirm_command_payloads",
    "targeted_confirm_output_subdir",
]
