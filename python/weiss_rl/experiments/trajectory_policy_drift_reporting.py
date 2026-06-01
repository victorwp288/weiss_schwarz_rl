from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.diagnostics.trajectory_policy_drift import (
    summarize_policy_drift,
    summarize_policy_drift_by_group,
    summarize_policy_scores,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


@dataclass(frozen=True, slots=True)
class PolicySpec:
    label: str
    run_dir: Path
    checkpoint_relpath: str


@dataclass(frozen=True, slots=True)
class PolicyScores:
    label: str
    top_actions: np.ndarray
    top_log_probs: np.ndarray
    target_log_probs: np.ndarray
    target_probabilities: np.ndarray
    top_families: np.ndarray
    values: np.ndarray
    opponent_context_episode_count: int


def parse_policy_spec(raw: str) -> PolicySpec:
    parts = [part.strip() for part in str(raw).split("|")]
    if len(parts) != 3 or not all(parts):
        raise SystemExit("--policy must use LABEL|RUN_DIR|CHECKPOINT_RELPATH")
    return PolicySpec(label=parts[0], run_dir=Path(parts[1]), checkpoint_relpath=parts[2].replace("\\", "/"))


def parse_policy_specs(raw_specs: list[str]) -> list[PolicySpec]:
    policy_specs = [parse_policy_spec(raw) for raw in raw_specs]
    if len({spec.label for spec in policy_specs}) != len(policy_specs):
        raise SystemExit("--policy labels must be unique")
    return policy_specs


def source_opponent_policy_ids_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    selected = dataset.metadata.get("selected_bundles")
    selected_bundles = selected if isinstance(selected, list) else []
    opponent_ids: list[str] = []
    for episode_index in range(int(dataset.episode_count)):
        bundle = selected_bundles[episode_index] if episode_index < len(selected_bundles) else {}
        if isinstance(bundle, dict):
            opponent_ids.append(str(bundle.get("source_opponent_policy_id") or "").strip())
        else:
            opponent_ids.append("")
    return opponent_ids


def trajectory_row_coordinates(dataset: ReplayTrajectoryDataset) -> list[dict[str, Any]]:
    selected = dataset.metadata.get("selected_bundles")
    selected_bundles = selected if isinstance(selected, list) else []
    coordinates: list[dict[str, Any]] = []
    batch_size = int(dataset.episode_count)
    for step_index in range(int(dataset.time_steps)):
        for episode_index in range(batch_size):
            bundle_meta = selected_bundles[episode_index] if episode_index < len(selected_bundles) else {}
            if not isinstance(bundle_meta, dict):
                bundle_meta = {}
            coordinates.append(
                {
                    "row_index": int(step_index * batch_size + episode_index),
                    "step_index": int(step_index),
                    "episode_index": int(episode_index),
                    "pair_index": bundle_meta.get("pair_index"),
                    "swap_index": bundle_meta.get("swap_index"),
                    "focal_seat": bundle_meta.get("focal_seat"),
                    "episode_seed": bundle_meta.get("episode_seed"),
                    "preference_pair_id": bundle_meta.get("preference_pair_id"),
                    "preference_role": bundle_meta.get("preference_role"),
                    "preference_role_label": bundle_meta.get("preference_role_label"),
                    "source_opponent_policy_id": bundle_meta.get("source_opponent_policy_id"),
                }
            )
    return coordinates


def trajectory_row_group_labels(dataset: ReplayTrajectoryDataset) -> dict[str, np.ndarray]:
    selected = dataset.metadata.get("selected_bundles")
    selected_bundles = selected if isinstance(selected, list) else []
    role_labels: list[str] = []
    opponent_ids: list[str] = []
    batch_size = int(dataset.episode_count)
    for _step_index in range(int(dataset.time_steps)):
        for episode_index in range(batch_size):
            bundle_meta = selected_bundles[episode_index] if episode_index < len(selected_bundles) else {}
            if not isinstance(bundle_meta, dict):
                bundle_meta = {}
            role_labels.append(str(bundle_meta.get("preference_role_label") or "").strip())
            opponent_ids.append(str(bundle_meta.get("source_opponent_policy_id") or "").strip())
    return {
        "preference_role_label": np.asarray(role_labels, dtype=object),
        "source_opponent_policy_id": np.asarray(opponent_ids, dtype=object),
    }


def build_trajectory_policy_drift_report(
    *,
    stack_config: Path,
    dataset_path: Path,
    dataset: ReplayTrajectoryDataset,
    device: Any,
    python_hash_seed: int,
    torch_threads: int,
    output_json: Path,
    reference_label: str,
    policy_specs: list[PolicySpec],
    scores_by_label: dict[str, PolicyScores],
    family_names: tuple[str, ...],
    family_by_action: np.ndarray,
    max_examples: int,
) -> dict[str, Any]:
    if reference_label not in scores_by_label:
        raise SystemExit(f"reference label {reference_label!r} was not among --policy labels")

    target_actions = dataset.actions.reshape(-1)
    target_families = family_by_action[np.clip(target_actions, 0, family_by_action.shape[0] - 1)]
    row_mask = dataset.policy_train_mask.reshape(-1)
    row_coordinates = trajectory_row_coordinates(dataset)
    row_group_labels = trajectory_row_group_labels(dataset)
    policy_summaries = []
    for label, scores in scores_by_label.items():
        summary = summarize_policy_scores(
            label=label,
            top_actions=scores.top_actions,
            target_actions=target_actions,
            target_probabilities=scores.target_probabilities,
            target_log_probs=scores.target_log_probs,
            top_families=scores.top_families,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            values=scores.values,
        )
        summary["opponent_context_episode_count"] = int(scores.opponent_context_episode_count)
        policy_summaries.append(summary)

    reference = scores_by_label[reference_label]
    drift_summaries = []
    for label, scores in scores_by_label.items():
        if label == reference_label:
            continue
        summary = summarize_policy_drift(
            reference_label=reference_label,
            candidate_label=label,
            reference_top_actions=reference.top_actions,
            candidate_top_actions=scores.top_actions,
            reference_target_probabilities=reference.target_probabilities,
            candidate_target_probabilities=scores.target_probabilities,
            reference_top_families=reference.top_families,
            candidate_top_families=scores.top_families,
            target_actions=target_actions,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            reference_target_log_probs=reference.target_log_probs,
            candidate_target_log_probs=scores.target_log_probs,
            reference_top_log_probs=reference.top_log_probs,
            candidate_top_log_probs=scores.top_log_probs,
            reference_values=reference.values,
            candidate_values=scores.values,
            row_coordinates=row_coordinates,
            max_examples=int(max_examples),
        )
        summary["preference_role_drift_summaries"] = summarize_policy_drift_by_group(
            group_name="preference_role_label",
            group_labels=row_group_labels["preference_role_label"],
            reference_label=reference_label,
            candidate_label=label,
            reference_top_actions=reference.top_actions,
            candidate_top_actions=scores.top_actions,
            reference_target_probabilities=reference.target_probabilities,
            candidate_target_probabilities=scores.target_probabilities,
            reference_top_families=reference.top_families,
            candidate_top_families=scores.top_families,
            target_actions=target_actions,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            reference_target_log_probs=reference.target_log_probs,
            candidate_target_log_probs=scores.target_log_probs,
            reference_top_log_probs=reference.top_log_probs,
            candidate_top_log_probs=scores.top_log_probs,
            reference_values=reference.values,
            candidate_values=scores.values,
            row_coordinates=row_coordinates,
            max_examples=int(max_examples),
        )
        summary["source_opponent_drift_summaries"] = summarize_policy_drift_by_group(
            group_name="source_opponent_policy_id",
            group_labels=row_group_labels["source_opponent_policy_id"],
            reference_label=reference_label,
            candidate_label=label,
            reference_top_actions=reference.top_actions,
            candidate_top_actions=scores.top_actions,
            reference_target_probabilities=reference.target_probabilities,
            candidate_target_probabilities=scores.target_probabilities,
            reference_top_families=reference.top_families,
            candidate_top_families=scores.top_families,
            target_actions=target_actions,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            reference_target_log_probs=reference.target_log_probs,
            candidate_target_log_probs=scores.target_log_probs,
            reference_top_log_probs=reference.top_log_probs,
            candidate_top_log_probs=scores.top_log_probs,
            reference_values=reference.values,
            candidate_values=scores.values,
            row_coordinates=row_coordinates,
            max_examples=int(max_examples),
        )
        drift_summaries.append(summary)

    return {
        "format": "trajectory_policy_drift_v1",
        "stack_config": stack_config.as_posix(),
        "dataset": dataset_path.as_posix(),
        "dataset_metadata": {
            "bundle_count": int(dataset.metadata.get("bundle_count", 0)),
            "train_rows": int(dataset.metadata.get("train_rows", 0)),
            "row_count": int(dataset.metadata.get("row_count", 0)),
            "unsupported_target_rows": int(dataset.metadata.get("unsupported_target_rows", 0)),
            "spec_hash256": dataset.metadata.get("spec_hash256"),
        },
        "device": str(device),
        "python_hash_seed": int(python_hash_seed),
        "torch_threads": int(torch_threads),
        "output_json": output_json.as_posix(),
        "reference_label": reference_label,
        "policies": [
            {
                "label": spec.label,
                "run_dir": spec.run_dir.as_posix(),
                "checkpoint_relpath": spec.checkpoint_relpath,
            }
            for spec in policy_specs
        ],
        "policy_summaries": policy_summaries,
        "drift_summaries": drift_summaries,
    }


def write_trajectory_policy_drift_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_trajectory_policy_drift_summary(report: dict[str, Any]) -> None:
    print(f"trajectory policy drift dataset: {report['dataset']}")
    for summary in report["policy_summaries"]:
        print(
            f"{summary['label']}: rows={summary['row_count']} "
            f"top_action_match={summary['top_action_matches_target_rate']:.4f} "
            f"top_family_match={summary['top_family_matches_target_rate']:.4f} "
            f"p_target={summary['mean_probability_on_target_action']:.4f}"
        )
    for summary in report["drift_summaries"]:
        print(
            f"{summary['reference_label']} -> {summary['candidate_label']}: "
            f"top_action_changed={summary['top_action_changed_rate']:.4f} "
            f"top_family_changed={summary['top_family_changed_rate']:.4f} "
            f"lost_target_top={summary['lost_target_top_action_rate']:.4f} "
            f"mean_p_delta={summary['mean_target_action_probability_delta']:.4f}"
        )
    print(f"output: {report['output_json']}")


__all__ = [
    "PolicyScores",
    "PolicySpec",
    "build_trajectory_policy_drift_report",
    "parse_policy_spec",
    "parse_policy_specs",
    "print_trajectory_policy_drift_summary",
    "source_opponent_policy_ids_by_episode",
    "trajectory_row_coordinates",
    "trajectory_row_group_labels",
    "write_trajectory_policy_drift_report",
]
