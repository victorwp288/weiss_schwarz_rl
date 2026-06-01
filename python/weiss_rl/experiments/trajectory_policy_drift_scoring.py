"""Policy scoring helpers for trajectory policy-drift diagnostics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.experiments.trajectory_policy_drift_reporting import (
    PolicyScores,
    PolicySpec,
    source_opponent_policy_ids_by_episode,
)
from weiss_rl.models.loading import load_snapshot_eval_model
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def configure_torch_determinism(*, torch_threads: int) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def action_catalog_from_stack_spec(spec_bundle: dict[str, Any]) -> Any:
    from weiss_rl.core.action_catalog import ActionCatalog

    return ActionCatalog.from_spec_bundle(spec_bundle)


def family_metadata(*, action_catalog: Any, action_dim: int) -> tuple[tuple[str, ...], np.ndarray]:
    family_names = tuple(str(family.name) for family in action_catalog.families)
    family_index = {name: index for index, name in enumerate(family_names)}
    family_by_action = np.full((int(action_dim),), -1, dtype=np.int64)
    for action_id in range(int(action_dim)):
        decoded = action_catalog.decode(action_id)
        family_by_action[action_id] = int(family_index.get(decoded.family, -1))
    return family_names, family_by_action


def score_policy(
    *,
    spec: PolicySpec,
    stack: Any,
    dataset: ReplayTrajectoryDataset,
    contract_spec_bundle: dict[str, Any],
    action_dim: int,
    family_by_action: np.ndarray,
    device: torch.device,
) -> PolicyScores:
    model = load_snapshot_eval_model(
        run_dir=spec.run_dir,
        snapshot_path=spec.checkpoint_relpath,
        stack=stack,
        observation_dim=int(dataset.obs.shape[-1]),
        action_dim=int(action_dim),
        observation_spec=contract_spec_bundle.get("observation"),
        spec_bundle=contract_spec_bundle,
    ).to(device)
    opponent_ids = source_opponent_policy_ids_by_episode(dataset)
    opponent_context_indices_list = model.opponent_context_indices_for_policy_ids(opponent_ids)
    opponent_context_indices = np.asarray(opponent_context_indices_list, dtype=np.int64)
    legal_actions = LegalActionBatch.from_packed(
        dataset.legal_ids,
        dataset.legal_offsets,
        meta=dataset.legal_action_meta,
        action_space=int(action_dim),
    )
    obs = torch.as_tensor(dataset.obs, device=device, dtype=torch.float32)
    acting_seat = torch.as_tensor(dataset.actor, device=device, dtype=torch.long)
    reset_before_step = torch.as_tensor(dataset.reset_before_step, device=device, dtype=torch.bool)
    initial_hidden = model.initial_seat_hidden(
        dataset.episode_count,
        device=device,
        opponent_context_indices=opponent_context_indices_list,
    )
    opponent_context_index = torch.as_tensor(
        np.broadcast_to(opponent_context_indices.reshape(1, -1), dataset.actions.shape).copy(),
        device=device,
        dtype=torch.long,
    )
    actions = torch.as_tensor(safe_actions_for_scoring(dataset), device=device, dtype=torch.long)
    eval_model = cast(Any, model)
    with torch.inference_mode():
        if bool(getattr(model, "supports_factorized_legal_policy", False)):
            result = eval_model.evaluate_factorized_sequence_packed_seat_aware(
                obs,
                acting_seat,
                initial_hidden,
                legal_actions=legal_actions,
                actions=actions,
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
            if result.top_action_ids is None or result.action_logp is None:
                raise RuntimeError(f"factorized policy did not return top actions/logp: {spec.label}")
            top_actions = result.top_action_ids.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            target_logp = result.action_logp.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            values = result.values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            top_action_tensor = torch.as_tensor(
                top_actions.reshape(dataset.actions.shape), device=device, dtype=torch.long
            )
            top_result = eval_model.evaluate_factorized_sequence_packed_seat_aware(
                obs,
                acting_seat,
                initial_hidden,
                legal_actions=legal_actions,
                actions=top_action_tensor,
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
            if top_result.action_logp is None:
                raise RuntimeError(f"factorized policy did not return top-action logp: {spec.label}")
            top_logp = top_result.action_logp.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        else:
            packed_logits, value_tensor, _hidden = eval_model.forward_sequence_packed_seat_aware(
                obs,
                acting_seat,
                initial_hidden,
                legal_actions=legal_actions,
                scoring_mode="learner",
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
            top_actions, target_logp, top_logp = dense_policy_scores_from_packed_logits(
                packed_logits.detach().cpu().numpy().astype(np.float64, copy=False),
                legal_ids=dataset.legal_ids,
                legal_offsets=dataset.legal_offsets,
                target_actions=dataset.actions.reshape(-1),
            )
            values = value_tensor.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    target_prob = np.exp(np.clip(target_logp, -80.0, 0.0))
    top_families = family_by_action[np.clip(top_actions, 0, family_by_action.shape[0] - 1)]
    return PolicyScores(
        label=spec.label,
        top_actions=top_actions,
        top_log_probs=top_logp,
        target_log_probs=target_logp,
        target_probabilities=target_prob,
        top_families=top_families,
        values=values,
        opponent_context_episode_count=int(np.count_nonzero(opponent_context_indices)),
    )


def dense_policy_scores_from_packed_logits(
    packed_logits: np.ndarray,
    *,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    target_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(legal_offsets, dtype=np.int64)
    ids = np.asarray(legal_ids, dtype=np.int64)
    row_count = int(offsets.shape[0] - 1)
    top_actions = np.full((row_count,), -1, dtype=np.int64)
    target_logp = np.full((row_count,), -np.inf, dtype=np.float64)
    top_logp = np.full((row_count,), -np.inf, dtype=np.float64)
    for row_index in range(row_count):
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        row_ids = ids[start:stop]
        row_logits = packed_logits[start:stop]
        if row_ids.size == 0:
            continue
        top_actions[row_index] = int(row_ids[int(np.argmax(row_logits))])
        finite = np.isfinite(row_logits)
        if not bool(np.any(finite)):
            log_probs = np.full_like(row_logits, -np.log(float(row_logits.size)))
        else:
            finite_logits = row_logits[finite]
            max_logit = float(np.max(finite_logits))
            log_z = max_logit + float(np.log(np.sum(np.exp(finite_logits - max_logit))))
            log_probs = np.full_like(row_logits, -np.inf)
            log_probs[finite] = row_logits[finite] - log_z
        target_positions = np.nonzero(row_ids == int(target_actions[row_index]))[0]
        if target_positions.size:
            target_logp[row_index] = float(log_probs[int(target_positions[0])])
        top_positions = np.nonzero(row_ids == int(top_actions[row_index]))[0]
        if top_positions.size:
            top_logp[row_index] = float(log_probs[int(top_positions[0])])
    return top_actions, target_logp, top_logp


def safe_actions_for_scoring(dataset: ReplayTrajectoryDataset) -> np.ndarray:
    """Use replay actions on train rows and legal placeholders elsewhere."""

    actions = np.asarray(dataset.actions, dtype=np.int64).reshape(-1).copy()
    train_mask = np.asarray(dataset.policy_train_mask, dtype=np.bool_).reshape(-1)
    offsets = np.asarray(dataset.legal_offsets, dtype=np.int64)
    legal_ids = np.asarray(dataset.legal_ids, dtype=np.int64)
    for row_index in np.nonzero(~train_mask)[0].tolist():
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        if stop > start:
            actions[row_index] = int(legal_ids[start])
    return actions.reshape(dataset.actions.shape)
