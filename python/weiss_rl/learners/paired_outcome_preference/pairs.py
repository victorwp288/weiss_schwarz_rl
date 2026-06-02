"""Preference-pair aggregation for paired outcome repair losses."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.paired_outcome_preference.edge_pairs import edge_mean_preference_pair_components
from weiss_rl.learners.paired_outcome_preference.pair_components import (
    PreferencePairComponents,
    balanced_pair_loss,
    weighted_pair_loss,
)
from weiss_rl.learners.paired_outcome_preference.span_pairs import span_preference_pair_components


def preference_pair_components(
    *,
    current_action_logp: Tensor,
    reference_action_logp: Tensor,
    current: Tensor,
    reference: Tensor,
    pair_ids: Tensor,
    roles: Tensor,
    valid: Tensor,
    group_ids: Tensor | None,
    pair_weight_rows: Tensor | None,
    preference_pair_ids: Tensor,
    preference_role: Tensor,
    preference_group_ids: Tensor | None,
    preference_pair_weights: Tensor | None,
    unique_pair_ids: Tensor,
    aggregation: str,
    beta: float,
    dtype: torch.dtype,
) -> PreferencePairComponents:
    if aggregation == "edge_mean":
        return edge_mean_preference_pair_components(
            current_action_logp=current_action_logp,
            reference_action_logp=reference_action_logp,
            pair_ids=preference_pair_ids,
            roles=preference_role,
            valid=valid.reshape(current_action_logp.shape),
            group_ids=preference_group_ids,
            pair_weight_rows=preference_pair_weights,
            unique_pair_ids=unique_pair_ids,
            beta=float(beta),
            dtype=dtype,
        )
    return span_preference_pair_components(
        current=current,
        reference=reference,
        pair_ids=pair_ids,
        roles=roles,
        valid=valid,
        group_ids=group_ids,
        pair_weight_rows=pair_weight_rows,
        unique_pair_ids=unique_pair_ids,
        aggregation=aggregation,
        beta=float(beta),
        dtype=dtype,
    )


__all__ = [
    "PreferencePairComponents",
    "balanced_pair_loss",
    "preference_pair_components",
    "weighted_pair_loss",
]
