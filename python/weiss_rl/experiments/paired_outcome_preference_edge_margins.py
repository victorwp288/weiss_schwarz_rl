from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.experiments.opponent_context_coverage import context_coverage_failures_from_report
from weiss_rl.experiments.paired_outcome_preference_decisions import (
    _aligned_edges_for_pair,
    _load_optional_action_catalog,
    _selected_bundles,
)
from weiss_rl.experiments.paired_outcome_preference_margins import (
    _load_model_for_dataset,
    _model_action_logp,
)
from weiss_rl.replay.trajectory_bc import load_replay_trajectory_bc_dataset
from weiss_rl.training.run_metadata import load_json_object


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceEdgeMarginConfig:
    dataset_path: Path
    stack_config_path: Path
    run_dir: Path
    checkpoint_path: Path
    reference_checkpoint_path: Path
    spec_bundle_json: Path | None = None
    include_same_action: bool = False
    min_mean_delta: float = 0.0
    min_min_delta: float = 0.0
    min_edge_improved_fraction: float = 1.0
    max_edge_worsened_fraction: float = 0.0
    min_same_state_mean_delta: float = 0.0
    min_required_group_mean_delta: float = 0.0
    required_groups: tuple[str, ...] = ()
    require_context: bool = True


def build_paired_outcome_preference_edge_margin_report(
    config: PairedOutcomePreferenceEdgeMarginConfig,
) -> dict[str, Any]:
    """Return per-aligned-decision preference movement before game eval escalation."""

    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    spec_bundle = load_json_object(Path(config.run_dir) / "spec_bundle.json", label="spec bundle")
    current_model = _load_model_for_dataset(
        checkpoint_path=config.checkpoint_path,
        stack_config_path=config.stack_config_path,
        run_dir=config.run_dir,
        observation_dim=int(dataset.obs.shape[-1]),
        spec_bundle=spec_bundle,
    )
    reference_model = _load_model_for_dataset(
        checkpoint_path=config.reference_checkpoint_path,
        stack_config_path=config.stack_config_path,
        run_dir=config.run_dir,
        observation_dim=int(dataset.obs.shape[-1]),
        spec_bundle=spec_bundle,
    )
    current_logp, current_context_coverage = _model_action_logp(current_model, dataset)
    reference_logp, reference_context_coverage = _model_action_logp(reference_model, dataset)
    catalog = _load_optional_action_catalog(config.spec_bundle_json)
    rows = edge_margin_rows_from_logps(
        dataset=dataset,
        selected_bundles=_selected_bundles(dataset),
        current_action_logp=current_logp,
        reference_action_logp=reference_logp,
        catalog=catalog,
        include_same_action=bool(config.include_same_action),
    )
    return evaluate_edge_margin_rows(
        rows,
        config=config,
        episode_count=int(dataset.episode_count),
        train_rows=int(np.count_nonzero(dataset.policy_train_mask)),
        current_context_coverage=current_context_coverage,
        reference_context_coverage=reference_context_coverage,
    )


def edge_margin_rows_from_logps(
    *,
    dataset: Any,
    selected_bundles: Sequence[Mapping[str, Any]],
    current_action_logp: np.ndarray,
    reference_action_logp: np.ndarray,
    catalog: Any | None = None,
    include_same_action: bool = False,
) -> list[dict[str, Any]]:
    if current_action_logp.shape != reference_action_logp.shape:
        raise ValueError("current and reference logp arrays must have the same shape")
    if current_action_logp.shape != dataset.actions.shape:
        raise ValueError("logp arrays must match dataset action shape")

    pair_roles = _episodes_by_pair_role(selected_bundles)
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(pair_roles):
        by_role = pair_roles[pair_id]
        preferred = list(by_role.get(1, []))
        rejected = list(by_role.get(0, []))
        if not preferred or not rejected:
            continue
        preferred_episode = int(preferred[0])
        rejected_episode = int(rejected[0])
        aligned_edges = _aligned_edges_for_pair(
            dataset,
            bundles=selected_bundles,
            catalog=catalog,
            preference_pair_id=pair_id,
            preferred_episode=preferred_episode,
            rejected_episode=rejected_episode,
        )
        preferred_bundle = selected_bundles[preferred_episode]
        for edge in aligned_edges:
            if bool(edge["same_action"]) and not include_same_action:
                continue
            step = int(edge["step_index"])
            cur_pref = float(current_action_logp[step, preferred_episode])
            cur_rej = float(current_action_logp[step, rejected_episode])
            ref_pref = float(reference_action_logp[step, preferred_episode])
            ref_rej = float(reference_action_logp[step, rejected_episode])
            if not all(np.isfinite(value) for value in (cur_pref, cur_rej, ref_pref, ref_rej)):
                continue
            row = dict(edge)
            row.update(
                {
                    "group_label": str(
                        preferred_bundle.get("merge_source_dataset_label")
                        or preferred_bundle.get("source_dataset_label")
                        or ""
                    ),
                    "current_preferred_logp": cur_pref,
                    "current_rejected_logp": cur_rej,
                    "reference_preferred_logp": ref_pref,
                    "reference_rejected_logp": ref_rej,
                    "current_raw_margin": cur_pref - cur_rej,
                    "reference_raw_margin": ref_pref - ref_rej,
                    "edge_delta": (cur_pref - ref_pref) - (cur_rej - ref_rej),
                }
            )
            rows.append(row)
    return rows


def evaluate_edge_margin_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: PairedOutcomePreferenceEdgeMarginConfig,
    episode_count: int,
    train_rows: int,
    current_context_coverage: Mapping[str, Any],
    reference_context_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    deltas = [float(row["edge_delta"]) for row in rows]
    same_state_deltas = [float(row["edge_delta"]) for row in rows if bool(row.get("same_current_state"))]
    edge_count = len(rows)
    edge_improved = sum(1 for delta in deltas if delta > 0.0)
    edge_worsened = sum(1 for delta in deltas if delta < 0.0)
    edge_improved_fraction = 0.0 if edge_count <= 0 else edge_improved / edge_count
    edge_worsened_fraction = 0.0 if edge_count <= 0 else edge_worsened / edge_count
    mean_delta = 0.0 if not deltas else float(np.mean(deltas))
    min_delta = 0.0 if not deltas else min(deltas)
    same_state_mean_delta = 0.0 if not same_state_deltas else float(np.mean(same_state_deltas))
    summaries = _summaries(rows, required_groups=config.required_groups)
    failures: list[str] = []

    if config.require_context:
        coverage_report = {
            "episode_count": int(episode_count),
            "current_context_episode_count": int(current_context_coverage.get("context_episode_count") or 0),
            "reference_context_episode_count": int(reference_context_coverage.get("context_episode_count") or 0),
            "current_context_coverage": dict(current_context_coverage),
            "reference_context_coverage": dict(reference_context_coverage),
        }
        failures.extend(
            context_coverage_failures_from_report(
                coverage_report,
                coverage_key="current_context_coverage",
                context_count_key="current_context_episode_count",
                prefix="current",
            )
        )
        failures.extend(
            context_coverage_failures_from_report(
                coverage_report,
                coverage_key="reference_context_coverage",
                context_count_key="reference_context_episode_count",
                prefix="reference",
            )
        )
    if edge_count <= 0:
        failures.append("empty_edge_surface")
    if mean_delta < float(config.min_mean_delta):
        failures.append(f"mean_delta_below:{mean_delta:.6g}<{float(config.min_mean_delta):.6g}")
    if min_delta < float(config.min_min_delta):
        failures.append(f"min_delta_below:{min_delta:.6g}<{float(config.min_min_delta):.6g}")
    if edge_improved_fraction < float(config.min_edge_improved_fraction):
        failures.append(
            f"edge_improved_fraction_below:{edge_improved_fraction:.6g}<"
            f"{float(config.min_edge_improved_fraction):.6g}"
        )
    if edge_worsened_fraction > float(config.max_edge_worsened_fraction):
        failures.append(
            f"edge_worsened_fraction_above:{edge_worsened_fraction:.6g}>"
            f"{float(config.max_edge_worsened_fraction):.6g}"
        )
    if same_state_deltas and same_state_mean_delta < float(config.min_same_state_mean_delta):
        failures.append(
            f"same_state_mean_delta_below:{same_state_mean_delta:.6g}<"
            f"{float(config.min_same_state_mean_delta):.6g}"
        )

    required = {str(group) for group in config.required_groups}
    present = {str(group["label"]) for group in summaries["groups"]}
    missing = sorted(required - present)
    if missing:
        failures.append("missing_required_groups:" + ",".join(missing))
    for group in summaries["groups"]:
        if bool(group["required"]) and float(group["delta_mean"]) < float(config.min_required_group_mean_delta):
            failures.append(
                f"required_group_mean_delta_below:{group['label']}:{float(group['delta_mean']):.6g}<"
                f"{float(config.min_required_group_mean_delta):.6g}"
            )

    return {
        "kind": "paired_outcome_preference_edge_margin_report_v1",
        "passed": not failures,
        "failures": failures,
        "dataset_path": config.dataset_path.as_posix(),
        "stack_config_path": config.stack_config_path.as_posix(),
        "run_dir": config.run_dir.as_posix(),
        "checkpoint_path": config.checkpoint_path.as_posix(),
        "reference_checkpoint_path": config.reference_checkpoint_path.as_posix(),
        "spec_bundle_json": None if config.spec_bundle_json is None else config.spec_bundle_json.as_posix(),
        "episode_count": int(episode_count),
        "train_rows": int(train_rows),
        "edge_count": edge_count,
        "same_current_state_edge_count": sum(1 for row in rows if bool(row.get("same_current_state"))),
        "same_history_edge_count": sum(1 for row in rows if bool(row.get("same_history"))),
        "current_context_episode_count": int(current_context_coverage.get("context_episode_count") or 0),
        "reference_context_episode_count": int(reference_context_coverage.get("context_episode_count") or 0),
        "current_context_coverage": dict(current_context_coverage),
        "reference_context_coverage": dict(reference_context_coverage),
        "thresholds": {
            "include_same_action": bool(config.include_same_action),
            "min_mean_delta": float(config.min_mean_delta),
            "min_min_delta": float(config.min_min_delta),
            "min_edge_improved_fraction": float(config.min_edge_improved_fraction),
            "max_edge_worsened_fraction": float(config.max_edge_worsened_fraction),
            "min_same_state_mean_delta": float(config.min_same_state_mean_delta),
            "min_required_group_mean_delta": float(config.min_required_group_mean_delta),
            "required_groups": list(config.required_groups),
            "require_context": bool(config.require_context),
        },
        "summary": {
            "mean_delta": mean_delta,
            "min_delta": min_delta,
            "edge_improved": edge_improved,
            "edge_worsened": edge_worsened,
            "edge_improved_fraction": edge_improved_fraction,
            "edge_worsened_fraction": edge_worsened_fraction,
            "same_state_mean_delta": same_state_mean_delta,
        },
        "groups": summaries["groups"],
        "pairs": summaries["pairs"],
        "rows": [dict(row) for row in rows],
    }


def write_paired_outcome_preference_edge_margin_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _episodes_by_pair_role(bundles: Sequence[Mapping[str, Any]]) -> dict[int, dict[int, list[int]]]:
    result: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for episode_index, bundle in enumerate(bundles):
        pair_id = _optional_int(bundle.get("preference_pair_id"))
        role = _optional_int(bundle.get("preference_role"))
        if pair_id is None or role not in {0, 1}:
            continue
        result[int(pair_id)][int(role)].append(int(episode_index))
    return result


def _summaries(rows: Sequence[Mapping[str, Any]], *, required_groups: Sequence[str]) -> dict[str, Any]:
    required = {str(group) for group in required_groups}
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_pair: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("group_label") or "")].append(row)
        by_pair[int(row["preference_pair_id"])].append(row)
    return {
        "groups": [
            _summary_row(label, group_rows, required=label in required)
            for label, group_rows in sorted(by_group.items())
        ],
        "pairs": [
            _pair_summary_row(pair_id, pair_rows)
            for pair_id, pair_rows in sorted(by_pair.items())
        ],
    }


def _summary_row(label: str, rows: Sequence[Mapping[str, Any]], *, required: bool) -> dict[str, Any]:
    deltas = [float(row["edge_delta"]) for row in rows]
    return {
        "label": label,
        "required": bool(required),
        "edge_count": len(rows),
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_min": min(deltas) if deltas else 0.0,
        "improved_edges": sum(1 for delta in deltas if delta > 0.0),
        "worsened_edges": sum(1 for delta in deltas if delta < 0.0),
        "same_current_state_edges": sum(1 for row in rows if bool(row.get("same_current_state"))),
    }


def _pair_summary_row(pair_id: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["edge_delta"]) for row in rows]
    first = rows[0] if rows else {}
    return {
        "preference_pair_id": int(pair_id),
        "group_label": str(first.get("group_label") or ""),
        "source_opponent_policy_id": str(first.get("source_opponent_policy_id") or ""),
        "source_pair_index": first.get("source_pair_index"),
        "edge_count": len(rows),
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_min": min(deltas) if deltas else 0.0,
        "improved_edges": sum(1 for delta in deltas if delta > 0.0),
        "worsened_edges": sum(1 for delta in deltas if delta < 0.0),
        "same_current_state_edges": sum(1 for row in rows if bool(row.get("same_current_state"))),
    }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PairedOutcomePreferenceEdgeMarginConfig",
    "build_paired_outcome_preference_edge_margin_report",
    "edge_margin_rows_from_logps",
    "evaluate_edge_margin_rows",
    "write_paired_outcome_preference_edge_margin_report",
]
