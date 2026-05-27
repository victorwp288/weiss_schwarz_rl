"""Row-level anti-regression guard for paired outcome preference replay."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.experiments.opponent_context_coverage import context_coverage_failures_from_report
from weiss_rl.experiments.paired_outcome_preference_margins import (
    _load_model_for_dataset,
    _source_opponent_policy_ids_by_episode,
)
from weiss_rl.experiments.paired_swing_context_margins import (
    _contextual_packed_scores,
    _logp_for_action,
    _rank_for_action,
)
from weiss_rl.replay.trajectory_bc import load_replay_trajectory_bc_dataset, replay_trajectory_bc_batch
from weiss_rl.training.run_metadata import load_json_object


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceRowGuardConfig:
    dataset_path: Path
    stack_config_path: Path
    run_dir: Path
    checkpoint_path: Path
    reference_checkpoint_path: Path
    protected_groups: tuple[str, ...] = ()
    required_groups: tuple[str, ...] = ()
    min_required_group_mean_logp_delta: float = 0.0
    min_protected_mean_logp_delta: float = 0.0
    max_protected_row_worsened_fraction: float = 0.0
    max_protected_rank_worsened_fraction: float = 0.0
    max_protected_top_family_changed_rate: float = 0.0
    top_action_near_tie_margin: float = 1e-5
    max_protected_lost_target_non_near_tie_rate: float = 0.0
    require_context: bool = True
    max_examples: int = 25


def build_paired_outcome_preference_row_guard(
    config: PairedOutcomePreferenceRowGuardConfig,
) -> dict[str, Any]:
    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    spec_bundle = load_json_object(Path(config.run_dir) / "spec_bundle.json", label="spec bundle")
    action_catalog = ActionCatalog.from_spec_bundle(spec_bundle)
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
    current_scores, current_context_coverage = _packed_scores_for_dataset(current_model, dataset)
    reference_scores, reference_context_coverage = _packed_scores_for_dataset(reference_model, dataset)
    rows = row_guard_rows_from_packed_scores(
        dataset=dataset,
        current_packed_scores=current_scores,
        reference_packed_scores=reference_scores,
        action_catalog=action_catalog,
        max_examples=int(config.max_examples),
    )
    return evaluate_row_guard_rows(
        rows,
        config=config,
        episode_count=int(dataset.episode_count),
        train_rows=int(np.count_nonzero(dataset.policy_train_mask)),
        current_context_coverage=current_context_coverage,
        reference_context_coverage=reference_context_coverage,
    )


def row_guard_rows_from_packed_scores(
    *,
    dataset: Any,
    current_packed_scores: Sequence[float] | np.ndarray,
    reference_packed_scores: Sequence[float] | np.ndarray,
    action_catalog: ActionCatalog,
    max_examples: int = 25,
) -> list[dict[str, Any]]:
    current_scores = torch.as_tensor(current_packed_scores, dtype=torch.float32)
    reference_scores = torch.as_tensor(reference_packed_scores, dtype=torch.float32)
    if tuple(current_scores.shape) != tuple(reference_scores.shape):
        raise ValueError("current and reference packed scores must have the same shape")

    bundles = dataset.metadata.get("selected_bundles")
    selected_bundles = bundles if isinstance(bundles, list) else []
    rows: list[dict[str, Any]] = []
    for step_index, episode_index in zip(*np.nonzero(dataset.policy_train_mask.astype(bool)), strict=False):
        step = int(step_index)
        episode = int(episode_index)
        row_index = step * int(dataset.episode_count) + episode
        start = int(dataset.legal_offsets[row_index])
        stop = int(dataset.legal_offsets[row_index + 1])
        row_ids = np.asarray(dataset.legal_ids[start:stop], dtype=np.int64)
        if row_ids.size <= 0:
            continue
        current_logp = torch.log_softmax(current_scores[start:stop], dim=0)
        reference_logp = torch.log_softmax(reference_scores[start:stop], dim=0)
        target_action = int(dataset.actions[step, episode])
        current_target_logp = _logp_for_action(row_ids, current_logp, target_action)
        reference_target_logp = _logp_for_action(row_ids, reference_logp, target_action)
        if not np.isfinite(current_target_logp) or not np.isfinite(reference_target_logp):
            continue
        current_rank = _rank_for_action(row_ids, current_logp, target_action)
        reference_rank = _rank_for_action(row_ids, reference_logp, target_action)
        current_top_index = int(torch.argmax(current_logp).detach().cpu().item())
        reference_top_index = int(torch.argmax(reference_logp).detach().cpu().item())
        current_top_action = int(row_ids[current_top_index])
        reference_top_action = int(row_ids[reference_top_index])
        current_top_logp = float(current_logp[current_top_index].detach().cpu().item())
        reference_top_logp = float(reference_logp[reference_top_index].detach().cpu().item())
        bundle = (
            selected_bundles[episode]
            if episode < len(selected_bundles) and isinstance(selected_bundles[episode], Mapping)
            else {}
        )
        current_target_prob = float(np.exp(np.clip(current_target_logp, -80.0, 0.0)))
        reference_target_prob = float(np.exp(np.clip(reference_target_logp, -80.0, 0.0)))
        current_target_family = _family_for_action(action_catalog, target_action)
        rows.append(
            {
                "step_index": step,
                "episode_index": episode,
                "target_action": target_action,
                "target_family": current_target_family,
                "group_label": _group_label(bundle),
                "preference_role_label": str(bundle.get("preference_role_label") or ""),
                "source_opponent_policy_id": str(bundle.get("source_opponent_policy_id") or ""),
                "source_pair_index": _jsonable(bundle.get("source_pair_index")),
                "preference_pair_id": _jsonable(bundle.get("preference_pair_id")),
                "current_target_logp": current_target_logp,
                "reference_target_logp": reference_target_logp,
                "target_logp_delta": current_target_logp - reference_target_logp,
                "current_target_probability": current_target_prob,
                "reference_target_probability": reference_target_prob,
                "target_probability_delta": current_target_prob - reference_target_prob,
                "current_target_rank": current_rank,
                "reference_target_rank": reference_rank,
                "target_rank_delta": None
                if current_rank is None or reference_rank is None
                else current_rank - reference_rank,
                "rank_worsened": bool(
                    current_rank is not None and reference_rank is not None and current_rank > reference_rank
                ),
                "row_worsened": bool(current_target_logp < reference_target_logp),
                "current_top_action": current_top_action,
                "reference_top_action": reference_top_action,
                "current_top_family": _family_for_action(action_catalog, current_top_action),
                "reference_top_family": _family_for_action(action_catalog, reference_top_action),
                "current_top_logp": current_top_logp,
                "reference_top_logp": reference_top_logp,
                "top_action_changed": current_top_action != reference_top_action,
                "top_family_changed": _family_for_action(action_catalog, current_top_action)
                != _family_for_action(action_catalog, reference_top_action),
                "lost_target_top_action": reference_top_action == target_action and current_top_action != target_action,
                "current_top_over_target_logp_margin": current_top_logp - current_target_logp,
            }
        )
    del max_examples
    return rows


def evaluate_row_guard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: PairedOutcomePreferenceRowGuardConfig,
    episode_count: int,
    train_rows: int,
    current_context_coverage: Mapping[str, Any],
    reference_context_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    protected = {str(group) for group in config.protected_groups}
    required = {str(group) for group in config.required_groups}
    groups = _group_summaries(
        rows,
        protected_groups=protected,
        required_groups=required,
        top_action_near_tie_margin=float(config.top_action_near_tie_margin),
        max_examples=int(config.max_examples),
    )
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

    present_groups = {str(group["label"]) for group in groups}
    missing_required = sorted(required - present_groups)
    if missing_required:
        failures.append("missing_required_groups:" + ",".join(missing_required))
    missing_protected = sorted(protected - present_groups)
    if missing_protected:
        failures.append("missing_protected_groups:" + ",".join(missing_protected))

    for group in groups:
        label = str(group["label"])
        if bool(group["required"]) and float(group["mean_target_logp_delta"]) < float(
            config.min_required_group_mean_logp_delta
        ):
            failures.append(
                f"required_group_mean_logp_delta_below:{label}:"
                f"{float(group['mean_target_logp_delta']):.6g}<"
                f"{float(config.min_required_group_mean_logp_delta):.6g}"
            )
        if not bool(group["protected"]):
            continue
        if float(group["mean_target_logp_delta"]) < float(config.min_protected_mean_logp_delta):
            failures.append(
                f"protected_mean_logp_delta_below:{label}:{float(group['mean_target_logp_delta']):.6g}<"
                f"{float(config.min_protected_mean_logp_delta):.6g}"
            )
        if float(group["row_worsened_fraction"]) > float(config.max_protected_row_worsened_fraction):
            failures.append(
                f"protected_row_worsened_fraction_above:{label}:{float(group['row_worsened_fraction']):.6g}>"
                f"{float(config.max_protected_row_worsened_fraction):.6g}"
            )
        if float(group["rank_worsened_fraction"]) > float(config.max_protected_rank_worsened_fraction):
            failures.append(
                f"protected_rank_worsened_fraction_above:{label}:{float(group['rank_worsened_fraction']):.6g}>"
                f"{float(config.max_protected_rank_worsened_fraction):.6g}"
            )
        if float(group["top_family_changed_rate"]) > float(config.max_protected_top_family_changed_rate):
            failures.append(
                f"protected_top_family_changed_rate_above:{label}:{float(group['top_family_changed_rate']):.6g}>"
                f"{float(config.max_protected_top_family_changed_rate):.6g}"
            )
        if float(group["lost_target_non_near_tie_rate"]) > float(config.max_protected_lost_target_non_near_tie_rate):
            failures.append(
                f"protected_lost_target_non_near_tie_rate_above:{label}:"
                f"{float(group['lost_target_non_near_tie_rate']):.6g}>"
                f"{float(config.max_protected_lost_target_non_near_tie_rate):.6g}"
            )

    return {
        "kind": "paired_outcome_preference_row_guard_v1",
        "passed": not failures,
        "failures": failures,
        "dataset_path": config.dataset_path.as_posix(),
        "stack_config_path": config.stack_config_path.as_posix(),
        "run_dir": config.run_dir.as_posix(),
        "checkpoint_path": config.checkpoint_path.as_posix(),
        "reference_checkpoint_path": config.reference_checkpoint_path.as_posix(),
        "episode_count": int(episode_count),
        "train_rows": int(train_rows),
        "row_count": len(rows),
        "current_context_episode_count": int(current_context_coverage.get("context_episode_count") or 0),
        "reference_context_episode_count": int(reference_context_coverage.get("context_episode_count") or 0),
        "current_context_coverage": dict(current_context_coverage),
        "reference_context_coverage": dict(reference_context_coverage),
        "thresholds": {
            "protected_groups": list(config.protected_groups),
            "required_groups": list(config.required_groups),
            "min_required_group_mean_logp_delta": float(config.min_required_group_mean_logp_delta),
            "min_protected_mean_logp_delta": float(config.min_protected_mean_logp_delta),
            "max_protected_row_worsened_fraction": float(config.max_protected_row_worsened_fraction),
            "max_protected_rank_worsened_fraction": float(config.max_protected_rank_worsened_fraction),
            "max_protected_top_family_changed_rate": float(config.max_protected_top_family_changed_rate),
            "top_action_near_tie_margin": float(config.top_action_near_tie_margin),
            "max_protected_lost_target_non_near_tie_rate": float(config.max_protected_lost_target_non_near_tie_rate),
            "require_context": bool(config.require_context),
        },
        "groups": groups,
        "summary": _summary_row(
            rows,
            label="all",
            protected=False,
            required=False,
            top_action_near_tie_margin=float(config.top_action_near_tie_margin),
            max_examples=int(config.max_examples),
        ),
        "rows": [dict(row) for row in rows[: max(0, int(config.max_examples))]],
    }


def write_paired_outcome_preference_row_guard(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _packed_scores_for_dataset(model: Any, dataset: Any) -> tuple[np.ndarray, dict[str, Any]]:
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    opponent_context_indices = np.asarray(
        model.opponent_context_indices_for_policy_ids(opponent_ids),
        dtype=np.int64,
    )
    from weiss_rl.experiments.opponent_context_coverage import summarize_opponent_context_coverage

    context_coverage = summarize_opponent_context_coverage(opponent_ids, opponent_context_indices.tolist())
    hidden = model.initial_seat_hidden(
        int(dataset.episode_count),
        device=torch.device("cpu"),
        opponent_context_indices=opponent_context_indices,
    )
    batch = replay_trajectory_bc_batch(
        dataset,
        episode_indices=list(range(int(dataset.episode_count))),
        initial_hidden_state=hidden.detach().cpu().numpy(),
        opponent_context_indices=opponent_context_indices,
    )
    return _contextual_packed_scores(model, batch), context_coverage


def _group_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    protected_groups: set[str],
    required_groups: set[str],
    top_action_near_tie_margin: float,
    max_examples: int,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("group_label") or "")].append(row)
    return [
        _summary_row(
            group_rows,
            label=label,
            protected=label in protected_groups,
            required=label in required_groups,
            top_action_near_tie_margin=top_action_near_tie_margin,
            max_examples=max_examples,
        )
        for label, group_rows in sorted(by_group.items())
    ]


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    protected: bool,
    required: bool,
    top_action_near_tie_margin: float,
    max_examples: int,
) -> dict[str, Any]:
    row_count = len(rows)
    logp_deltas = [float(row["target_logp_delta"]) for row in rows]
    probability_deltas = [float(row["target_probability_delta"]) for row in rows]
    rank_worsened = [row for row in rows if bool(row.get("rank_worsened"))]
    row_worsened = [row for row in rows if bool(row.get("row_worsened"))]
    top_family_changed = [row for row in rows if bool(row.get("top_family_changed"))]
    lost_target_top = [row for row in rows if bool(row.get("lost_target_top_action"))]
    lost_non_near_tie = [
        row
        for row in lost_target_top
        if float(row.get("current_top_over_target_logp_margin") or 0.0) > float(top_action_near_tie_margin)
    ]
    return {
        "label": str(label),
        "protected": bool(protected),
        "required": bool(required),
        "row_count": row_count,
        "mean_target_logp_delta": 0.0 if not logp_deltas else float(np.mean(logp_deltas)),
        "min_target_logp_delta": 0.0 if not logp_deltas else min(logp_deltas),
        "mean_target_probability_delta": 0.0 if not probability_deltas else float(np.mean(probability_deltas)),
        "min_target_probability_delta": 0.0 if not probability_deltas else min(probability_deltas),
        "row_worsened_count": len(row_worsened),
        "row_worsened_fraction": 0.0 if row_count <= 0 else len(row_worsened) / row_count,
        "rank_worsened_count": len(rank_worsened),
        "rank_worsened_fraction": 0.0 if row_count <= 0 else len(rank_worsened) / row_count,
        "top_family_changed_count": len(top_family_changed),
        "top_family_changed_rate": 0.0 if row_count <= 0 else len(top_family_changed) / row_count,
        "lost_target_top_action_count": len(lost_target_top),
        "lost_target_top_action_rate": 0.0 if row_count <= 0 else len(lost_target_top) / row_count,
        "lost_target_non_near_tie_count": len(lost_non_near_tie),
        "lost_target_non_near_tie_rate": 0.0 if row_count <= 0 else len(lost_non_near_tie) / row_count,
        "examples": {
            "row_worsened": [_row_example(row) for row in row_worsened[:max_examples]],
            "rank_worsened": [_row_example(row) for row in rank_worsened[:max_examples]],
            "lost_target_non_near_tie": [_row_example(row) for row in lost_non_near_tie[:max_examples]],
        },
    }


def _row_example(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "step_index",
        "episode_index",
        "target_action",
        "target_family",
        "group_label",
        "preference_role_label",
        "source_opponent_policy_id",
        "source_pair_index",
        "target_logp_delta",
        "target_probability_delta",
        "current_target_rank",
        "reference_target_rank",
        "current_top_action",
        "reference_top_action",
        "current_top_family",
        "reference_top_family",
        "current_top_over_target_logp_margin",
    )
    return {key: _jsonable(row.get(key)) for key in keys}


def _group_label(bundle: Mapping[str, Any]) -> str:
    return str(bundle.get("merge_source_dataset_label") or bundle.get("source_dataset_label") or "")


def _family_for_action(action_catalog: ActionCatalog, action_id: int) -> str:
    try:
        return str(action_catalog.decode(int(action_id)).family)
    except Exception:
        return ""


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


__all__ = [
    "PairedOutcomePreferenceRowGuardConfig",
    "build_paired_outcome_preference_row_guard",
    "evaluate_row_guard_rows",
    "row_guard_rows_from_packed_scores",
    "write_paired_outcome_preference_row_guard",
]
