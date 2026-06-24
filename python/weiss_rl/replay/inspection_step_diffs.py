"""Policy-distribution step diffs for replay inspection."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.action_distribution_metrics import (
    family_probability_masses,
    gap_from_top_to_action,
    rank_of_action,
    same_family_margin_to_action,
    top_action_payload,
    top_margin,
)
from weiss_rl.core.masking import masked_log_softmax
from weiss_rl.replay.bundles import ReplayStep


def build_step_diff(
    *,
    step_index: int,
    expected_step: ReplayStep,
    raw_legal_ids: np.ndarray,
    legal_ids_a: np.ndarray,
    legal_ids_b: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    top_actions: int,
    action_catalog: ActionCatalog | None,
) -> dict[str, Any]:
    legal_mask_a = _legal_mask_from_ids(logits_a.shape[0], legal_ids_a)
    legal_mask_b = _legal_mask_from_ids(logits_b.shape[0], legal_ids_b)
    union_mask = legal_mask_a | legal_mask_b
    stacked_logits = np.stack((logits_a, logits_b), axis=0)
    stacked_mask = np.stack((legal_mask_a, legal_mask_b), axis=0)
    log_probs = masked_log_softmax(stacked_logits, stacked_mask)
    probs = np.zeros_like(log_probs, dtype=np.float64)
    probs[stacked_mask] = np.exp(log_probs[stacked_mask].astype(np.float64, copy=False))

    kl_divergence_ab = _kl_divergence(probs[0], probs[1])
    kl_divergence_ba = _kl_divergence(probs[1], probs[0])
    probability_delta = probs[1] - probs[0]
    total_variation = float(0.5 * np.sum(np.abs(probability_delta[union_mask]), dtype=np.float64))
    abs_probability_delta = np.abs(probability_delta)
    legal_action_indices_a = np.flatnonzero(legal_mask_a)
    legal_action_indices_b = np.flatnonzero(legal_mask_b)
    union_action_indices = np.flatnonzero(union_mask)
    ranked_action_indices = union_action_indices[np.argsort(abs_probability_delta[union_action_indices])[::-1]]
    policy_a_top_action = top_action_payload(
        probabilities=probs[0],
        legal_indices=legal_action_indices_a,
        action_catalog=action_catalog,
        descriptor_fn=action_descriptor,
        empty_error="Replay inspection requires at least one legal action per compared step",
    )
    policy_b_top_action = top_action_payload(
        probabilities=probs[1],
        legal_indices=legal_action_indices_b,
        action_catalog=action_catalog,
        descriptor_fn=action_descriptor,
        empty_error="Replay inspection requires at least one legal action per compared step",
    )
    policy_a_top_action_id = int(policy_a_top_action["action"])
    policy_b_top_action_id = int(policy_b_top_action["action"])
    raw_legal_action_count = int(np.asarray(raw_legal_ids).shape[0])
    policy_a_legal_action_count = int(legal_action_indices_a.shape[0])
    policy_b_legal_action_count = int(legal_action_indices_b.shape[0])
    policy_a_top_logit_margin = top_margin(values=logits_a, legal_indices=legal_action_indices_a)
    policy_a_top_probability_margin = top_margin(values=probs[0], legal_indices=legal_action_indices_a)
    policy_a_b_top_action_logit_gap = gap_from_top_to_action(
        values=logits_a,
        legal_indices=legal_action_indices_a,
        action_id=policy_b_top_action_id,
    )
    policy_a_b_top_action_same_family_logit_margin = same_family_margin_to_action(
        values=logits_a,
        legal_indices=legal_action_indices_a,
        action_id=policy_b_top_action_id,
        action_catalog=action_catalog,
    )
    policy_a_family_masses = family_probability_masses(
        probabilities=probs[0],
        legal_indices=legal_action_indices_a,
        action_catalog=action_catalog,
    )
    policy_b_family_masses = family_probability_masses(
        probabilities=probs[1],
        legal_indices=legal_action_indices_b,
        action_catalog=action_catalog,
    )
    policy_a_top_family = str(policy_a_top_action.get("family", "unknown"))
    policy_b_top_family = str(policy_b_top_action.get("family", "unknown"))

    return {
        "step_index": int(step_index),
        "decision_id": int(expected_step.decision_id),
        "actor": int(expected_step.actor),
        "recorded_action": int(expected_step.action),
        "recorded_action_detail": action_descriptor(int(expected_step.action), action_catalog=action_catalog),
        "raw_legal_action_count": raw_legal_action_count,
        "policy_a_legal_action_count": policy_a_legal_action_count,
        "policy_b_legal_action_count": policy_b_legal_action_count,
        "policy_a_legal_surface_removed_action_count": max(raw_legal_action_count - policy_a_legal_action_count, 0),
        "policy_b_legal_surface_removed_action_count": max(raw_legal_action_count - policy_b_legal_action_count, 0),
        "policy_a_legal_surface_is_filtered": policy_a_legal_action_count < raw_legal_action_count,
        "policy_b_legal_surface_is_filtered": policy_b_legal_action_count < raw_legal_action_count,
        "policy_b_top_action_legal_for_policy_a": bool(legal_mask_a[policy_b_top_action_id]),
        "policy_a_top_action_legal_for_policy_b": bool(legal_mask_b[policy_a_top_action_id]),
        "total_variation": total_variation,
        "kl_divergence_ab": kl_divergence_ab,
        "kl_divergence_ba": kl_divergence_ba,
        "max_abs_probability_delta": float(np.max(abs_probability_delta[union_action_indices], initial=0.0)),
        "policy_a_recorded_action_probability": float(probs[0, int(expected_step.action)]),
        "policy_b_recorded_action_probability": float(probs[1, int(expected_step.action)]),
        "policy_a_probability_on_policy_b_top_action": float(probs[0, policy_b_top_action_id]),
        "policy_a_probability_on_policy_b_top_action_family": float(
            policy_a_family_masses.get(policy_b_top_family, 0.0)
        ),
        "policy_a_top_logit_margin": policy_a_top_logit_margin,
        "policy_a_top_probability_margin": policy_a_top_probability_margin,
        "policy_a_gap_from_top_logit_to_policy_b_top_action": policy_a_b_top_action_logit_gap,
        "policy_a_policy_b_top_action_same_family_logit_margin": policy_a_b_top_action_same_family_logit_margin,
        "policy_a_top_action_family_probability": float(policy_a_family_masses.get(policy_a_top_family, 0.0)),
        "policy_b_top_action_family_probability": float(policy_b_family_masses.get(policy_b_top_family, 0.0)),
        "policy_a_rank_of_policy_b_top_action": rank_of_action(
            probabilities=probs[0],
            legal_indices=legal_action_indices_a,
            action_id=policy_b_top_action_id,
        ),
        "policy_a_matches_policy_b_top_action": policy_a_top_action_id == policy_b_top_action_id,
        "policy_a_matches_policy_b_top_action_family": (
            policy_a_top_action.get("family") == policy_b_top_action.get("family")
            if "family" in policy_a_top_action and "family" in policy_b_top_action
            else False
        ),
        "policy_a_top_action": policy_a_top_action,
        "policy_b_top_action": policy_b_top_action,
        "policy_a_family_probability_masses": policy_a_family_masses,
        "policy_b_family_probability_masses": policy_b_family_masses,
        "top_action_deltas": [
            {
                **action_descriptor(int(action_index), action_catalog=action_catalog),
                "probability_a": float(probs[0, action_index]),
                "probability_b": float(probs[1, action_index]),
                "probability_delta_b_minus_a": float(probability_delta[action_index]),
                "abs_probability_delta": float(abs_probability_delta[action_index]),
            }
            for action_index in ranked_action_indices.tolist()[:top_actions]
        ],
    }


def action_descriptor(action_id: int, *, action_catalog: ActionCatalog | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": int(action_id)}
    if action_catalog is None:
        return payload
    decoded = action_catalog.decode(int(action_id))
    payload["family"] = decoded.family
    if decoded.hand_index is not None:
        payload["hand_index"] = int(decoded.hand_index)
    if decoded.stage_slot is not None:
        payload["stage_slot"] = int(decoded.stage_slot)
    if decoded.from_slot is not None:
        payload["from_slot"] = int(decoded.from_slot)
    if decoded.to_slot is not None:
        payload["to_slot"] = int(decoded.to_slot)
    if decoded.slot is not None:
        payload["slot"] = int(decoded.slot)
    if decoded.attack_type is not None:
        payload["attack_type"] = str(decoded.attack_type)
    if decoded.index is not None:
        payload["index"] = int(decoded.index)
    return payload


def _legal_mask_from_ids(action_dim: int, legal_ids: np.ndarray) -> np.ndarray:
    legal_mask = np.zeros((int(action_dim),), dtype=bool)
    legal_ids_array = np.asarray(legal_ids, dtype=np.int64)
    if legal_ids_array.size:
        legal_mask[legal_ids_array] = True
    return legal_mask


def _kl_divergence(probs_p: np.ndarray, probs_q: np.ndarray) -> float:
    support = probs_p > 0.0
    if not bool(np.any(support)):
        return 0.0
    q = np.maximum(probs_q[support], np.finfo(np.float64).tiny)
    p = probs_p[support]
    return float(np.sum(p * (np.log(p) - np.log(q)), dtype=np.float64))


__all__ = [
    "action_descriptor",
    "build_step_diff",
]
