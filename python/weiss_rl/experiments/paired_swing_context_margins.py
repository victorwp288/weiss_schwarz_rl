"""Mechanistic opponent-context margin checks for paired-swing replay rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import load_stack_config
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.experiments.opponent_context_coverage import summarize_opponent_context_coverage
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE, build_policy_value_model
from weiss_rl.models.loading import restore_model_guidance_from_payload
from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    replay_trajectory_bc_batch,
)
from weiss_rl.training.paired_swing_conflict_filter import (
    normalize_paired_swing_action_source as _normalize_action_source,
)
from weiss_rl.training.run_metadata import load_json_object


@dataclass(frozen=True, slots=True)
class PairedSwingContextMarginConfig:
    dataset_path: Path
    stack_config_path: Path
    run_dir: Path
    checkpoint_path: Path
    positive_action_source: str = "actions"
    negative_action_source: str = "teacher_action"
    report_action_ids: tuple[int, ...] = (104, 124)


def build_paired_swing_context_margin_report(config: PairedSwingContextMarginConfig) -> dict[str, Any]:
    """Return row-level positive-vs-negative log-prob margins under opponent context."""

    dataset = load_replay_trajectory_bc_dataset(Path(config.dataset_path))
    stack = load_stack_config(Path(config.stack_config_path))
    spec_bundle = load_json_object(Path(config.run_dir) / "spec_bundle.json", label="spec bundle")
    checkpoint = torch.load(Path(config.checkpoint_path), map_location="cpu", weights_only=True)
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise ValueError(f"checkpoint is missing model_state_dict: {config.checkpoint_path}")
    model_config = stack.config.model
    if model_config is None:
        raise ValueError("stack config is missing model section")
    model = build_policy_value_model(
        observation_dim=int(dataset.obs.shape[-1]),
        config=model_config,
        action_dim=GLOBAL_ACTION_SPACE_SIZE,
        observation_spec=_observation_spec_from_bundle(spec_bundle),
        spec_bundle=spec_bundle,
    )
    load_model_state_dict_with_context_compat(
        model,
        model_state_dict,
        context=f"paired-swing context margin checkpoint {config.checkpoint_path}",
    )
    restore_model_guidance_from_payload(model, checkpoint)
    model.eval()

    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    opponent_context_indices = np.asarray(
        model.opponent_context_indices_for_policy_ids(opponent_ids),
        dtype=np.int64,
    )
    opponent_context_indices_list = opponent_context_indices.tolist()
    context_coverage = summarize_opponent_context_coverage(opponent_ids, opponent_context_indices_list)
    hidden = model.initial_seat_hidden(
        int(dataset.episode_count),
        device=torch.device("cpu"),
        opponent_context_indices=opponent_context_indices_list,
    )
    batch = replay_trajectory_bc_batch(
        dataset,
        episode_indices=list(range(int(dataset.episode_count))),
        initial_hidden_state=hidden.detach().cpu().numpy(),
        opponent_context_indices=opponent_context_indices,
    )
    packed_scores = _contextual_packed_scores(model, batch)
    rows = paired_swing_margin_rows_from_packed_scores(
        dataset,
        packed_scores=packed_scores,
        positive_action_source=config.positive_action_source,
        negative_action_source=config.negative_action_source,
        opponent_context_indices=opponent_context_indices,
        report_action_ids=config.report_action_ids,
    )
    positive_margins = [float(row["positive_minus_negative_logp"]) for row in rows]
    return {
        "kind": "paired_swing_context_margin_report_v1",
        "dataset_path": Path(config.dataset_path).as_posix(),
        "stack_config_path": Path(config.stack_config_path).as_posix(),
        "run_dir": Path(config.run_dir).as_posix(),
        "checkpoint_path": Path(config.checkpoint_path).as_posix(),
        "positive_action_source": _normalize_action_source(
            config.positive_action_source,
            field_name="positive_action_source",
        ),
        "negative_action_source": _normalize_action_source(
            config.negative_action_source,
            field_name="negative_action_source",
        ),
        "report_action_ids": [int(action_id) for action_id in config.report_action_ids],
        "episode_count": int(dataset.episode_count),
        "row_count": len(rows),
        "context_episode_count": int(context_coverage["context_episode_count"]),
        "context_coverage": context_coverage,
        "positive_margin_min": None if not positive_margins else min(positive_margins),
        "positive_margin_mean": None if not positive_margins else float(np.mean(positive_margins)),
        "rows": rows,
    }


def paired_swing_margin_rows_from_packed_scores(
    dataset: ReplayTrajectoryDataset,
    *,
    packed_scores: Sequence[float] | np.ndarray | torch.Tensor,
    positive_action_source: str,
    negative_action_source: str,
    opponent_context_indices: Sequence[int] | np.ndarray | None = None,
    report_action_ids: Sequence[int] = (104, 124),
) -> list[dict[str, Any]]:
    """Summarize active paired-swing rows from packed candidate scores."""

    positive_source = _normalize_action_source(positive_action_source, field_name="positive_action_source")
    negative_source = _normalize_action_source(negative_action_source, field_name="negative_action_source")
    positive_actions = _dataset_actions(dataset, positive_source)
    negative_actions = _dataset_actions(dataset, negative_source)
    scores = torch.as_tensor(packed_scores, dtype=torch.float32)
    context_indices = (
        None if opponent_context_indices is None else np.asarray(opponent_context_indices, dtype=np.int64).reshape(-1)
    )
    if context_indices is not None and int(context_indices.shape[0]) != int(dataset.episode_count):
        raise ValueError("opponent_context_indices must match dataset episode_count")
    selected_bundles = dataset.metadata.get("selected_bundles")
    bundles = selected_bundles if isinstance(selected_bundles, list) else []
    rows: list[dict[str, Any]] = []
    valid = (
        dataset.policy_train_mask.astype(bool)
        & (positive_actions >= 0)
        & (negative_actions >= 0)
        & (positive_actions != negative_actions)
    )
    if positive_source == "teacher_action" or negative_source == "teacher_action":
        valid &= dataset.teacher_valid.astype(bool)

    for step_index, episode_index in zip(*np.nonzero(valid), strict=False):
        step = int(step_index)
        episode = int(episode_index)
        row_index = step * int(dataset.episode_count) + episode
        start = int(dataset.legal_offsets[row_index])
        stop = int(dataset.legal_offsets[row_index + 1])
        row_ids = np.asarray(dataset.legal_ids[start:stop], dtype=np.int64)
        row_scores = scores[start:stop]
        row_logp = torch.log_softmax(row_scores, dim=0)
        positive_action = int(positive_actions[step, episode])
        negative_action = int(negative_actions[step, episode])
        positive_logp = _logp_for_action(row_ids, row_logp, positive_action)
        negative_logp = _logp_for_action(row_ids, row_logp, negative_action)
        top_index = int(torch.argmax(row_logp).detach().cpu().item())
        top_action = int(row_ids[top_index])
        bundle = bundles[episode] if episode < len(bundles) and isinstance(bundles[episode], Mapping) else {}
        action_logps = {
            str(int(action_id)): _logp_for_action(row_ids, row_logp, int(action_id)) for action_id in report_action_ids
        }
        rows.append(
            {
                "step_index": step,
                "episode_index": episode,
                "source_dataset_label": str(bundle.get("source_dataset_label") or ""),
                "source_opponent_policy_id": str(bundle.get("source_opponent_policy_id") or ""),
                "source_pair_indices": _jsonable(bundle.get("source_pair_indices")),
                "source_pair_index": _jsonable(bundle.get("source_pair_index")),
                "opponent_context_index": None if context_indices is None else int(context_indices[episode]),
                "positive_action": positive_action,
                "negative_action": negative_action,
                "positive_logp": positive_logp,
                "negative_logp": negative_logp,
                "positive_minus_negative_logp": positive_logp - negative_logp,
                "top_action": top_action,
                "top_logp": float(row_logp[top_index].detach().cpu().item()),
                "positive_rank": _rank_for_action(row_ids, row_logp, positive_action),
                "negative_rank": _rank_for_action(row_ids, row_logp, negative_action),
                "reported_action_logps": action_logps,
            }
        )
    return rows


def write_paired_swing_context_margin_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contextual_packed_scores(model: Any, batch: Mapping[str, Any]) -> np.ndarray:
    obs = torch.as_tensor(np.asarray(batch["obs"], dtype=np.float32), dtype=torch.float32)
    expected_shape = obs.shape[:2]
    learner = ImpalaLearner(model=model, teacher_aux_mode="off")
    packed_legal = learner._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
    if packed_legal is None:
        raise ValueError("paired-swing context margin check requires packed legal actions")
    loss_mask = torch.as_tensor(np.asarray(batch["policy_train_mask"]), dtype=torch.float32)
    if bool(getattr(model, "supports_factorized_legal_policy", False)):
        packed_view, _target_logits = learner._factorized_public_heuristic_teacher_view(
            batch,
            obs=obs,
            loss_mask=loss_mask,
            packed_legal=packed_legal,
            score_public_target=False,
        )
        if packed_view is None:
            raise ValueError("factorized context margin check could not build packed scores")
        return packed_view.logits.detach().cpu().numpy().astype(np.float32, copy=False)

    legal_actions = LegalActionBatch.from_packed(
        np.asarray(batch["legal_ids"], dtype=np.uint32),
        np.asarray(batch["legal_offsets"], dtype=np.uint32),
        meta=np.asarray(batch["legal_action_meta"], dtype=np.uint16),
        action_space=GLOBAL_ACTION_SPACE_SIZE,
    )
    forward = learner._forward_time_major(
        obs,
        initial_hidden_state=batch.get("initial_hidden_state"),
        to_play_seat=batch.get("to_play_seat"),
        actor=batch.get("actor"),
        legal_actions=legal_actions,
        policy_train_mask=loss_mask,
        reset_before_step=batch.get("reset_before_step"),
        opponent_context_index=batch.get("opponent_context_index"),
    )
    if forward.packed_logits is None:
        raise ValueError("context margin check requires packed model scores")
    return forward.packed_logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _observation_spec_from_bundle(spec_bundle: Mapping[str, Any]) -> dict[str, Any] | None:
    observation = spec_bundle.get("observation")
    if observation is None:
        return None
    if not isinstance(observation, dict):
        raise ValueError("spec_bundle observation payload must be an object")
    return dict(observation)


def _dataset_actions(dataset: ReplayTrajectoryDataset, source: str) -> np.ndarray:
    normalized = _normalize_action_source(source, field_name="action_source")
    if normalized == "actions":
        return np.asarray(dataset.actions, dtype=np.int64)
    if normalized == "teacher_action":
        return np.asarray(dataset.teacher_action, dtype=np.int64)
    raise AssertionError(f"unreachable action source: {normalized}")


def _logp_for_action(row_ids: np.ndarray, row_logp: torch.Tensor, action_id: int) -> float:
    matches = np.nonzero(row_ids == int(action_id))[0]
    if matches.size == 0:
        return float("-inf")
    return float(row_logp[int(matches[0])].detach().cpu().item())


def _rank_for_action(row_ids: np.ndarray, row_logp: torch.Tensor, action_id: int) -> int | None:
    matches = np.nonzero(row_ids == int(action_id))[0]
    if matches.size == 0:
        return None
    value = row_logp[int(matches[0])]
    return int((row_logp > value).sum().detach().cpu().item()) + 1


def _source_opponent_policy_ids_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        raise ValueError("dataset metadata must contain selected_bundles matching episode_count")
    return [
        str(bundle.get("source_opponent_policy_id") or "").strip() if isinstance(bundle, Mapping) else ""
        for bundle in bundles
    ]


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
    "PairedSwingContextMarginConfig",
    "build_paired_swing_context_margin_report",
    "paired_swing_margin_rows_from_packed_scores",
    "write_paired_swing_context_margin_report",
]
