from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from weiss_rl.experiments.bootstrap_commands import repo_relative
from weiss_rl.experiments.guarded_league_bootstrap_selection import (
    evaluate_multiobjective_guard,
    resolve_repo_path,
    selected_confirm_summary_path,
    selection_anchor_scores,
)


class GuardedLeagueBootstrapMultiobjectiveConfig(Protocol):
    @property
    def multiobjective_reference_summary_jsons(self) -> Sequence[Path]: ...

    @property
    def multiobjective_fixed_opponents(self) -> Sequence[str]: ...

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


def selected_anchor_scores(
    *,
    final_selected: Mapping[str, Any],
    selected_confirmation_summary_path: str,
    targeted_confirm_records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    scores = selection_anchor_scores(final_selected)
    if scores:
        return scores
    for record in targeted_confirm_records:
        if str(record.get("summary_path", "")) == selected_confirmation_summary_path:
            return dict(cast(Mapping[str, float], record.get("anchor_scores", {})))
    if targeted_confirm_records:
        return dict(cast(Mapping[str, float], targeted_confirm_records[-1].get("anchor_scores", {})))
    return {}


def populate_selected_segment_record(
    *,
    segment_record: MutableMapping[str, Any],
    repo_root: Path,
    final_selected: Mapping[str, Any],
    selected_confirmation_summary_path: str,
    targeted_confirm_records: Sequence[Mapping[str, Any]],
    anchor_scores: Mapping[str, float],
    guard: Mapping[str, Any],
) -> None:
    if selected_confirmation_summary_path:
        segment_record["targeted_confirm_summary"] = repo_relative(
            repo_root / selected_confirmation_summary_path,
            repo_root=repo_root,
        ).as_posix()
    elif targeted_confirm_records:
        segment_record["targeted_confirm_summary"] = targeted_confirm_records[-1].get("summary_path")
    segment_record["targeted_anchor_scores"] = dict(anchor_scores)
    segment_record["selected"] = dict(final_selected)
    segment_record["anchor_scores"] = dict(anchor_scores)
    segment_record["guard"] = dict(guard)


def evaluate_selected_multiobjective_guard(
    *,
    config: GuardedLeagueBootstrapMultiobjectiveConfig,
    repo_root: Path,
    selected_confirmation_summary_path: str,
    targeted_confirm_records: Sequence[Mapping[str, Any]],
    effective_learned_guard_opponents: Sequence[str],
) -> dict[str, Any] | None:
    candidate_summary_path = selected_confirm_summary_path(
        raw_path=selected_confirmation_summary_path,
        fallback_record=targeted_confirm_records[-1] if targeted_confirm_records else None,
        repo_root=repo_root,
    )
    if candidate_summary_path is None:
        return None
    return evaluate_multiobjective_guard(
        candidate_summary_json=candidate_summary_path,
        reference_summary_jsons=tuple(
            resolve_repo_path(path, repo_root=repo_root) for path in config.multiobjective_reference_summary_jsons
        ),
        fixed_opponents=config.multiobjective_fixed_opponents,
        learned_opponents=effective_learned_guard_opponents,
        min_fixed_score=config.min_multiobjective_fixed_score,
        max_fixed_reference_drop=config.max_multiobjective_fixed_reference_drop,
        min_learned_score=config.min_learned_guard_score,
        min_learned_mean=config.min_learned_guard_mean,
        min_learned_reference_delta=config.min_learned_guard_reference_delta,
        max_learned_reference_drop=config.max_learned_guard_reference_drop,
    )


def write_multiobjective_guard_artifact(
    *,
    segment_record: MutableMapping[str, Any],
    multiobjective_guard: Mapping[str, Any],
    diagnostics_dir: Path,
    run_label: str,
    repo_root: Path,
) -> Path:
    output_path = diagnostics_dir / f"{run_label}_multiobjective_gate.json"
    output_path.write_text(
        json.dumps(multiobjective_guard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    segment_record["multiobjective_guard"] = dict(multiobjective_guard)
    segment_record["multiobjective_guard_json"] = repo_relative(output_path, repo_root=repo_root).as_posix()
    return output_path


__all__ = [
    "GuardedLeagueBootstrapMultiobjectiveConfig",
    "evaluate_selected_multiobjective_guard",
    "populate_selected_segment_record",
    "selected_anchor_scores",
    "write_multiobjective_guard_artifact",
]
