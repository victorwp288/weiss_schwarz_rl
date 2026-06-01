from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_selection import (
    DEFAULT_CONFIRM_OPPONENTS,
    DEFAULT_REQUIRED_ANCHORS,
    build_b1_candidate_selection,
)
from weiss_rl.experiments.bootstrap_commands import (
    resolve_snapshot_checkpoint_path as resolve_snapshot_checkpoint_path,
)
from weiss_rl.experiments.segmented_b1_guided_bootstrap_outcomes import (
    selection_score,
)
from weiss_rl.experiments.segmented_b1_guided_bootstrap_plan import (
    build_segmented_bootstrap_summary,
)
from weiss_rl.experiments.segmented_b1_guided_bootstrap_runner import (
    run_segmented_bootstrap_step,
)

DEFAULT_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml")
DEFAULT_ALIAS_POLICY_ID = "guided_bootstrap_floor_segmented_selected"
DEFAULT_RUN_LABEL = "b1_guided_floor_segmented"


@dataclass(frozen=True, slots=True)
class SegmentRuntime:
    num_envs: int = 288
    unroll_length: int = 64
    segment_updates: int = 25
    runtime_mode: str = "train_async_fast"
    simulator_profile: str = "fast"
    device: str = "cuda"
    checkpoint_interval_updates: int = 5
    collection_backend: str = "process"
    profile_timers: bool = True


@dataclass(frozen=True, slots=True)
class SegmentedBootstrapConfig:
    repo_root: Path
    initial_run_dir: Path
    initial_policy_id: str
    run_prefix: str
    stack_config: Path = DEFAULT_STACK_CONFIG
    seed_run_dir: Path | None = None
    alias_policy_id: str = DEFAULT_ALIAS_POLICY_ID
    segments: int = 4
    runtime: SegmentRuntime = SegmentRuntime()
    confirm_paired_seeds: int = 64
    bootstrap_samples: int = 2000
    required_anchors: tuple[str, ...] = DEFAULT_REQUIRED_ANCHORS
    confirm_opponents: tuple[str, ...] = DEFAULT_CONFIRM_OPPONENTS
    min_required_anchor_score: float = 0.5
    max_selected_drop: float = 0.02
    stop_on_latest_falloff: bool = False
    max_latest_drop: float = 0.05
    dry_run: bool = False


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def _initial_selection_score(config: SegmentedBootstrapConfig) -> float | None:
    summary = build_b1_candidate_selection(
        [config.initial_run_dir],
        stack_config=config.stack_config,
        required_anchors=config.required_anchors,
        confirm_opponents=config.confirm_opponents,
        min_required_anchor_score=config.min_required_anchor_score,
        confirm_paired_seeds=config.confirm_paired_seeds,
    )
    selected = summary.get("selected")
    if not isinstance(selected, Mapping):
        return None
    return selection_score(selected)


def run_segmented_b1_guided_bootstrap(
    config: SegmentedBootstrapConfig,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    repo_root = config.repo_root.resolve()
    current_run_dir = config.initial_run_dir.resolve()
    current_seed_run_dir = (config.seed_run_dir or config.initial_run_dir).resolve()
    current_policy_id = str(config.initial_policy_id).strip()
    diagnostics_dir = repo_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / f"{config.run_prefix}_segmented_bootstrap_summary.json"
    previous_score = _initial_selection_score(config)
    summary = build_segmented_bootstrap_summary(
        config=config,
        repo_root=repo_root,
        created_unix=time.time(),
        initial_selection_score=previous_score,
    )

    for segment_index in range(1, int(config.segments) + 1):
        step_result = run_segmented_bootstrap_step(
            config=config,
            repo_root=repo_root,
            diagnostics_dir=diagnostics_dir,
            segment_index=segment_index,
            current_run_dir=current_run_dir,
            current_seed_run_dir=current_seed_run_dir,
            current_policy_id=current_policy_id,
            previous_score=previous_score,
            runner=runner,
        )
        summary["segments"].append(step_result.segment_record)

        if step_result.should_stop:
            summary["status"] = step_result.terminal_status
            if step_result.terminal_reason is not None:
                summary["stop_reason"] = step_result.terminal_reason
            break

        previous_score = step_result.next_previous_score
        current_run_dir = step_result.next_run_dir
        current_seed_run_dir = step_result.next_seed_run_dir
        current_policy_id = step_result.next_policy_id
        summary["status"] = "completed"

    summary["summary_path"] = summary_path.as_posix()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
