from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.experiments.bootstrap_commands import (
    build_candidate_selector_command,
    build_targeted_confirm_entrypoint_command,
    build_train_entrypoint_command,
    command_record,
    repo_relative,
)


class SegmentRuntimeProtocol(Protocol):
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


class SegmentedBootstrapConfigProtocol(Protocol):
    @property
    def repo_root(self) -> Path: ...

    @property
    def initial_run_dir(self) -> Path: ...

    @property
    def initial_policy_id(self) -> str: ...

    @property
    def run_prefix(self) -> str: ...

    @property
    def stack_config(self) -> Path: ...

    @property
    def seed_run_dir(self) -> Path | None: ...

    @property
    def alias_policy_id(self) -> str: ...

    @property
    def segments(self) -> int: ...

    @property
    def runtime(self) -> SegmentRuntimeProtocol: ...

    @property
    def confirm_paired_seeds(self) -> int: ...

    @property
    def bootstrap_samples(self) -> int: ...

    @property
    def required_anchors(self) -> tuple[str, ...]: ...

    @property
    def confirm_opponents(self) -> tuple[str, ...]: ...

    @property
    def min_required_anchor_score(self) -> float: ...

    @property
    def max_selected_drop(self) -> float: ...

    @property
    def stop_on_latest_falloff(self) -> bool: ...

    @property
    def max_latest_drop(self) -> float: ...

    @property
    def dry_run(self) -> bool: ...


def build_segmented_bootstrap_summary(
    *,
    config: SegmentedBootstrapConfigProtocol,
    repo_root: Path,
    created_unix: float,
    initial_selection_score: float | None,
) -> dict[str, Any]:
    return {
        "kind": "segmented_b1_guided_bootstrap_v1",
        "created_unix": float(created_unix),
        "repo_root": repo_root.as_posix(),
        "stack_config": repo_relative(config.stack_config, repo_root=repo_root).as_posix(),
        "initial_run_dir": repo_relative(config.initial_run_dir, repo_root=repo_root).as_posix(),
        "initial_policy_id": config.initial_policy_id,
        "alias_policy_id": config.alias_policy_id,
        "segments_requested": int(config.segments),
        "confirm_paired_seeds": int(config.confirm_paired_seeds),
        "min_required_anchor_score": float(config.min_required_anchor_score),
        "max_selected_drop": float(config.max_selected_drop),
        "max_latest_drop": float(config.max_latest_drop),
        "stop_on_latest_falloff": bool(config.stop_on_latest_falloff),
        "initial_selection_score": initial_selection_score,
        "segments": [],
        "status": "planned" if config.dry_run else "running",
    }


def build_train_segment_command(
    *,
    config: SegmentedBootstrapConfigProtocol,
    segment_run_label: str,
    init_checkpoint_path: Path,
    seed_run_dir: Path,
) -> list[str]:
    runtime = config.runtime
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
        seed_snapshot_run_dir=seed_run_dir,
        init_checkpoint_path=init_checkpoint_path,
        collection_backend=runtime.collection_backend,
        profile_timers=runtime.profile_timers,
    )


def build_selector_command(
    *,
    config: SegmentedBootstrapConfigProtocol,
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
        selected_alias_policy_id=config.alias_policy_id,
    )


def build_targeted_confirm_command(
    *,
    config: SegmentedBootstrapConfigProtocol,
    run_dir: Path,
    focal_policy_id: str,
    output_subdir: str,
) -> list[str]:
    return build_targeted_confirm_entrypoint_command(
        repo_root=config.repo_root,
        stack_config=config.stack_config,
        run_dir=run_dir,
        b1_baseline_run_dir=run_dir,
        focal_policy_id=focal_policy_id,
        paired_seeds=config.confirm_paired_seeds,
        bootstrap_samples=config.bootstrap_samples,
        output_subdir=output_subdir,
        opponents=config.confirm_opponents,
    )


def build_segment_record(
    *,
    config: SegmentedBootstrapConfigProtocol,
    repo_root: Path,
    segment_index: int,
    segment_run_label: str,
    segment_run_dir: Path,
    source_run_dir: Path,
    source_policy_id: str,
    source_checkpoint_path: Path,
    seed_run_dir: Path,
    train_command: Sequence[str],
    preselect_json: Path,
    final_json: Path,
) -> dict[str, Any]:
    return {
        "segment": int(segment_index),
        "run_label": segment_run_label,
        "run_dir": repo_relative(segment_run_dir, repo_root=repo_root).as_posix(),
        "source_run_dir": repo_relative(source_run_dir, repo_root=repo_root).as_posix(),
        "source_policy_id": source_policy_id,
        "source_checkpoint": repo_relative(source_checkpoint_path, repo_root=repo_root).as_posix(),
        "seed_run_dir": repo_relative(seed_run_dir, repo_root=repo_root).as_posix(),
        "train_command": command_record(train_command),
        "preselect_json": repo_relative(preselect_json, repo_root=repo_root).as_posix(),
        "final_selection_json": repo_relative(final_json, repo_root=repo_root).as_posix(),
    }


def populate_dry_run_segment_plan(
    segment_record: dict[str, Any],
    *,
    config: SegmentedBootstrapConfigProtocol,
    segment_run_dir: Path,
    preselect_json: Path,
    final_json: Path,
) -> None:
    segment_record["status"] = "planned"
    segment_record["preselect_command"] = command_record(
        build_selector_command(config=config, run_dir=segment_run_dir, output_json=preselect_json, publish_alias=False)
    )
    segment_record["targeted_confirm_command_template"] = command_record(
        build_targeted_confirm_command(
            config=config,
            run_dir=segment_run_dir,
            focal_policy_id="<selected-policy-id>",
            output_subdir=f"segmented_confirm{int(config.confirm_paired_seeds)}_<selected-policy-id>",
        )
    )
    segment_record["final_selector_command"] = command_record(
        build_selector_command(config=config, run_dir=segment_run_dir, output_json=final_json, publish_alias=True)
    )
