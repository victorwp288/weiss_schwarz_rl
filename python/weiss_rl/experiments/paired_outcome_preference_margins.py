"""Mechanistic margin reports for paired outcome preference replay datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import load_stack_config
from weiss_rl.experiments.opponent_context_coverage import summarize_opponent_context_coverage
from weiss_rl.learners.action_logp import packed_selected_action_logp
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, build_policy_value_model
from weiss_rl.models.loading import restore_model_guidance_from_payload
from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat
from weiss_rl.replay.trajectory_bc import load_replay_trajectory_bc_dataset, replay_trajectory_bc_batch
from weiss_rl.training.run_metadata import load_json_object


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceMarginConfig:
    dataset_path: Path
    stack_config_path: Path
    run_dir: Path
    checkpoint_path: Path
    reference_checkpoint_path: Path
    aggregation: str = "mean"


def build_paired_outcome_preference_margin_report(
    config: PairedOutcomePreferenceMarginConfig,
) -> dict[str, Any]:
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
    pair_rows = preference_margin_rows_from_logps(
        selected_bundles=dataset.metadata.get("selected_bundles"),
        current_action_logp=current_logp,
        reference_action_logp=reference_logp,
        preference_pair_ids=_metadata_field(dataset, "preference_pair_id"),
        preference_roles=_metadata_field(dataset, "preference_role"),
        loss_mask=dataset.policy_train_mask,
        aggregation=str(config.aggregation),
    )
    margins = [float(row["dpo_margin"]) for row in pair_rows]
    group_summaries = _group_summaries(pair_rows)
    return {
        "kind": "paired_outcome_preference_margin_report_v1",
        "dataset_path": config.dataset_path.as_posix(),
        "stack_config_path": config.stack_config_path.as_posix(),
        "run_dir": config.run_dir.as_posix(),
        "checkpoint_path": config.checkpoint_path.as_posix(),
        "reference_checkpoint_path": config.reference_checkpoint_path.as_posix(),
        "aggregation": str(config.aggregation),
        "episode_count": int(dataset.episode_count),
        "train_rows": int(np.count_nonzero(dataset.policy_train_mask)),
        "pair_count": len(pair_rows),
        "current_context_episode_count": int(current_context_coverage["context_episode_count"]),
        "reference_context_episode_count": int(reference_context_coverage["context_episode_count"]),
        "current_context_coverage": current_context_coverage,
        "reference_context_coverage": reference_context_coverage,
        "dpo_margin_mean": None if not margins else float(np.mean(margins)),
        "dpo_margin_min": None if not margins else min(margins),
        "dpo_margin_max": None if not margins else max(margins),
        "satisfied_fraction": None if not margins else float(np.mean(np.asarray(margins) > 0.0)),
        "groups": group_summaries,
        "rows": pair_rows,
    }


def preference_margin_rows_from_logps(
    *,
    selected_bundles: object,
    current_action_logp: np.ndarray,
    reference_action_logp: np.ndarray,
    preference_pair_ids: np.ndarray,
    preference_roles: np.ndarray,
    loss_mask: np.ndarray,
    aggregation: str = "mean",
) -> list[dict[str, Any]]:
    normalized_aggregation = str(aggregation).strip().lower()
    if normalized_aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be one of: mean, sum")
    if current_action_logp.shape != reference_action_logp.shape:
        raise ValueError("current and reference logp arrays must have the same shape")
    if preference_pair_ids.shape != current_action_logp.shape:
        raise ValueError("preference_pair_ids must match logp shape")
    if preference_roles.shape != current_action_logp.shape:
        raise ValueError("preference_roles must match logp shape")
    if loss_mask.shape != current_action_logp.shape:
        raise ValueError("loss_mask must match logp shape")

    bundles = selected_bundles if isinstance(selected_bundles, list) else []
    valid = (
        loss_mask.astype(bool)
        & np.isfinite(current_action_logp)
        & np.isfinite(reference_action_logp)
        & (preference_pair_ids >= 0)
        & ((preference_roles == 0) | (preference_roles == 1))
    )
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(int(item) for item in np.unique(preference_pair_ids[valid])):
        pair_mask = valid & (preference_pair_ids == pair_id)
        preferred_mask = pair_mask & (preference_roles == 1)
        rejected_mask = pair_mask & (preference_roles == 0)
        if not np.any(preferred_mask) or not np.any(rejected_mask):
            continue
        cur_pref = _aggregate(current_action_logp[preferred_mask], normalized_aggregation)
        ref_pref = _aggregate(reference_action_logp[preferred_mask], normalized_aggregation)
        cur_rej = _aggregate(current_action_logp[rejected_mask], normalized_aggregation)
        ref_rej = _aggregate(reference_action_logp[rejected_mask], normalized_aggregation)
        preferred_episode = int(np.nonzero(np.any(preferred_mask, axis=0))[0][0])
        rejected_episode = int(np.nonzero(np.any(rejected_mask, axis=0))[0][0])
        preferred_bundle = (
            bundles[preferred_episode]
            if preferred_episode < len(bundles) and isinstance(bundles[preferred_episode], Mapping)
            else {}
        )
        rejected_bundle = (
            bundles[rejected_episode]
            if rejected_episode < len(bundles) and isinstance(bundles[rejected_episode], Mapping)
            else {}
        )
        rows.append(
            {
                "preference_pair_id": int(pair_id),
                "group_label": str(
                    preferred_bundle.get("merge_source_dataset_label")
                    or preferred_bundle.get("source_dataset_label")
                    or ""
                ),
                "preferred_label": str(preferred_bundle.get("preference_role_label") or ""),
                "rejected_label": str(rejected_bundle.get("preference_role_label") or ""),
                "opponent_policy_id": str(preferred_bundle.get("source_opponent_policy_id") or ""),
                "source_pair_index": _jsonable(preferred_bundle.get("source_pair_index")),
                "preferred_episode_index": preferred_episode,
                "rejected_episode_index": rejected_episode,
                "preferred_rows": int(np.count_nonzero(preferred_mask)),
                "rejected_rows": int(np.count_nonzero(rejected_mask)),
                "current_preferred_logp": float(cur_pref),
                "reference_preferred_logp": float(ref_pref),
                "current_rejected_logp": float(cur_rej),
                "reference_rejected_logp": float(ref_rej),
                "dpo_margin": float((cur_pref - ref_pref) - (cur_rej - ref_rej)),
                "current_raw_margin": float(cur_pref - cur_rej),
                "reference_raw_margin": float(ref_pref - ref_rej),
            }
        )
    return rows


def write_paired_outcome_preference_margin_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_action_logp(model: Any, dataset: Any) -> tuple[np.ndarray, dict[str, Any]]:
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    opponent_context_indices = np.asarray(
        model.opponent_context_indices_for_policy_ids(opponent_ids),
        dtype=np.int64,
    )
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
    from weiss_rl.experiments.paired_swing_context_margins import _contextual_packed_scores

    packed_scores = torch.as_tensor(_contextual_packed_scores(model, batch), dtype=torch.float32)
    action_logp = packed_selected_action_logp(
        packed_scores,
        torch.as_tensor(batch["legal_ids"], dtype=torch.long),
        torch.as_tensor(batch["legal_offsets"], dtype=torch.long),
        torch.as_tensor(batch["actions"], dtype=torch.long),
        pass_action_id=int(dataset.metadata.get("pass_action_id", 0)),
        strict=False,
    )
    return action_logp.detach().cpu().numpy().reshape(dataset.actions.shape), context_coverage


def _load_model_for_dataset(
    *,
    checkpoint_path: Path,
    stack_config_path: Path,
    run_dir: Path,
    observation_dim: int,
    spec_bundle: Mapping[str, Any],
) -> Any:
    stack = load_stack_config(stack_config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise ValueError(f"checkpoint is missing model_state_dict: {checkpoint_path}")
    model_config = stack.config.model
    if model_config is None:
        raise ValueError("stack config is missing model section")
    model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=GLOBAL_ACTION_SPACE_SIZE,
        observation_spec=_observation_spec_from_bundle(spec_bundle),
        spec_bundle=spec_bundle,
    )
    load_model_state_dict_with_context_compat(
        model,
        model_state_dict,
        context=f"paired outcome preference margin checkpoint {checkpoint_path}",
    )
    restore_model_guidance_from_payload(model, checkpoint)
    model.eval()
    return model


def _metadata_field(dataset: Any, field_name: str) -> np.ndarray:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        raise ValueError("dataset metadata must contain selected_bundles matching episode_count")
    values = []
    for bundle in bundles:
        if not isinstance(bundle, Mapping) or field_name not in bundle:
            raise ValueError(f"dataset selected_bundles missing {field_name}")
        values.append(int(bundle[field_name]))
    return np.broadcast_to(np.asarray(values, dtype=np.int64).reshape(1, -1), dataset.actions.shape).copy()


def _source_opponent_policy_ids_by_episode(dataset: Any) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        raise ValueError("dataset metadata must contain selected_bundles matching episode_count")
    return [
        str(bundle.get("source_opponent_policy_id") or "").strip() if isinstance(bundle, Mapping) else ""
        for bundle in bundles
    ]


def _aggregate(values: np.ndarray, aggregation: str) -> float:
    if aggregation == "sum":
        return float(np.sum(values))
    return float(np.mean(values))


def _group_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_label") or "")].append(row)
    summaries: dict[str, Any] = {}
    for label, group_rows in sorted(groups.items()):
        margins = [float(row["dpo_margin"]) for row in group_rows]
        summaries[label] = {
            "pair_count": len(group_rows),
            "dpo_margin_mean": float(np.mean(margins)) if margins else None,
            "dpo_margin_min": min(margins) if margins else None,
            "satisfied_fraction": float(np.mean(np.asarray(margins) > 0.0)) if margins else None,
        }
    return summaries


def _observation_spec_from_bundle(spec_bundle: Mapping[str, Any]) -> dict[str, Any] | None:
    observation = spec_bundle.get("observation")
    if observation is None:
        return None
    if not isinstance(observation, dict):
        raise ValueError("spec_bundle observation payload must be an object")
    return dict(observation)


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
    "PairedOutcomePreferenceMarginConfig",
    "build_paired_outcome_preference_margin_report",
    "preference_margin_rows_from_logps",
    "write_paired_outcome_preference_margin_report",
]
