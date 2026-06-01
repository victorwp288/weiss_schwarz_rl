from __future__ import annotations

import json
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.experiments.bootstrap_commands import repo_relative


class GuardedLeagueBootstrapSummaryConfig(Protocol):
    @property
    def stack_config(self) -> Path: ...

    @property
    def seed_snapshot_run_dir(self) -> Path: ...

    @property
    def b1_baseline_run_dir(self) -> Path | None: ...

    @property
    def init_checkpoint_path(self) -> Path: ...

    @property
    def run_prefix(self) -> str: ...

    @property
    def segments(self) -> int: ...

    @property
    def confirm_paired_seeds(self) -> int: ...

    @property
    def publish_min_confirm_paired_seeds(self) -> int: ...

    @property
    def confirm_recent_candidate_count(self) -> int: ...

    @property
    def required_anchors(self) -> Sequence[str]: ...

    @property
    def confirm_opponents(self) -> Sequence[str]: ...

    @property
    def min_required_anchor_score(self) -> float: ...

    @property
    def first_init_schedule_offset_updates(self) -> int | None: ...

    @property
    def reference_label(self) -> str: ...

    @property
    def reference_anchor_scores(self) -> Mapping[str, float]: ...

    @property
    def multiobjective_reference_summary_jsons(self) -> Sequence[Path]: ...

    @property
    def multiobjective_fixed_opponents(self) -> Sequence[str]: ...

    @property
    def learned_guard_opponents(self) -> Sequence[str]: ...

    @property
    def min_multiobjective_fixed_score(self) -> float: ...

    @property
    def max_multiobjective_fixed_reference_drop(self) -> float: ...

    @property
    def min_learned_guard_score(self) -> float: ...

    @property
    def min_learned_guard_mean(self) -> float: ...

    @property
    def min_learned_guard_reference_delta(self) -> float | None: ...

    @property
    def max_learned_guard_reference_drop(self) -> float | None: ...

    @property
    def max_reference_drop(self) -> float: ...

    @property
    def selected_alias_policy_id(self) -> str: ...

    @property
    def continue_unpublished_confirmed(self) -> bool: ...

    @property
    def dry_run(self) -> bool: ...


def guarded_bootstrap_summary_path(*, diagnostics_dir: Path, run_prefix: str) -> Path:
    return diagnostics_dir / f"{run_prefix}_guarded_league_bootstrap_summary.json"


def build_guarded_league_bootstrap_summary(
    *,
    config: GuardedLeagueBootstrapSummaryConfig,
    repo_root: Path,
    effective_learned_guard_opponents: Sequence[str],
    created_unix: float | None = None,
) -> dict[str, Any]:
    return {
        "kind": "guarded_league_bootstrap_v1",
        "created_unix": time.time() if created_unix is None else float(created_unix),
        "repo_root": repo_root.as_posix(),
        "stack_config": repo_relative(config.stack_config, repo_root=repo_root).as_posix(),
        "seed_snapshot_run_dir": repo_relative(config.seed_snapshot_run_dir, repo_root=repo_root).as_posix(),
        "b1_baseline_run_dir": repo_relative(
            config.b1_baseline_run_dir or config.seed_snapshot_run_dir,
            repo_root=repo_root,
        ).as_posix(),
        "initial_checkpoint": repo_relative(config.init_checkpoint_path, repo_root=repo_root).as_posix(),
        "run_prefix": config.run_prefix,
        "segments_requested": int(config.segments),
        "confirm_paired_seeds": int(config.confirm_paired_seeds),
        "publish_min_confirm_paired_seeds": int(config.publish_min_confirm_paired_seeds),
        "confirm_recent_candidate_count": int(config.confirm_recent_candidate_count),
        "required_anchors": list(config.required_anchors),
        "confirm_opponents": list(config.confirm_opponents),
        "effective_confirm_opponents": list(
            dict.fromkeys([*config.confirm_opponents, *effective_learned_guard_opponents])
        ),
        "min_required_anchor_score": float(config.min_required_anchor_score),
        "first_init_schedule_offset_updates": config.first_init_schedule_offset_updates,
        "reference_label": str(config.reference_label),
        "reference_anchor_scores": {key: float(value) for key, value in sorted(config.reference_anchor_scores.items())},
        "multiobjective_reference_summary_jsons": [
            repo_relative(path, repo_root=repo_root).as_posix()
            for path in config.multiobjective_reference_summary_jsons
        ],
        "multiobjective_fixed_opponents": list(config.multiobjective_fixed_opponents),
        "configured_learned_guard_opponents": list(config.learned_guard_opponents),
        "learned_guard_opponents": list(effective_learned_guard_opponents),
        "learned_guard_opponents_inferred": not bool(config.learned_guard_opponents)
        and bool(effective_learned_guard_opponents),
        "multiobjective_thresholds": {
            "min_fixed_score": float(config.min_multiobjective_fixed_score),
            "max_fixed_reference_drop": float(config.max_multiobjective_fixed_reference_drop),
            "min_learned_score": float(config.min_learned_guard_score),
            "min_learned_mean": float(config.min_learned_guard_mean),
            "min_learned_reference_delta": config.min_learned_guard_reference_delta,
            "max_learned_reference_drop": config.max_learned_guard_reference_drop,
        },
        "max_reference_drop": float(config.max_reference_drop),
        "selected_alias_policy_id": str(config.selected_alias_policy_id),
        "continue_unpublished_confirmed": bool(config.continue_unpublished_confirmed),
        "segments": [],
        "status": "planned" if config.dry_run else "running",
    }


def write_guarded_league_bootstrap_summary(
    *,
    summary: MutableMapping[str, Any],
    summary_path: Path,
) -> None:
    summary["summary_path"] = summary_path.as_posix()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "GuardedLeagueBootstrapSummaryConfig",
    "build_guarded_league_bootstrap_summary",
    "guarded_bootstrap_summary_path",
    "write_guarded_league_bootstrap_summary",
]
