"""Row-level action comparisons for paired-swing replay losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.action_logp import packed_selected_action_logp
from weiss_rl.learners.paired_swing.rows import positive_vs_top_other_margin_by_row


@dataclass(frozen=True)
class PairedSwingComparisonRows:
    margin_by_row: Tensor
    supported: Tensor
    positive_logp_by_row: Tensor
    negative_logp_by_row: Tensor


def paired_swing_margin_comparison_rows(
    *,
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    flat_positive_actions: Tensor,
    flat_negative_actions: Tensor,
    positive_actions: Tensor,
    negative_actions: Tensor,
    active_rows: Tensor,
    pass_action_id: int | None,
    compare_to: str,
) -> PairedSwingComparisonRows:
    """Compute per-row positive-vs-comparator margins for active paired-swing rows."""

    if compare_to == "top_other":
        margin_by_row, supported, positive_logp_by_row, negative_logp_by_row = positive_vs_top_other_margin_by_row(
            packed_logits=packed_logits,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            flat_positive_actions=flat_positive_actions,
            active_rows=active_rows,
        )
        return PairedSwingComparisonRows(
            margin_by_row=margin_by_row,
            supported=supported,
            positive_logp_by_row=positive_logp_by_row,
            negative_logp_by_row=negative_logp_by_row,
        )

    positive_logp = packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        flat_positive_actions.reshape_as(positive_actions),
        pass_action_id=pass_action_id,
        strict=False,
    ).reshape(-1)
    negative_logp = packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        flat_negative_actions.reshape_as(negative_actions),
        pass_action_id=pass_action_id,
        strict=False,
    ).reshape(-1)
    supported = active_rows & torch.isfinite(positive_logp) & torch.isfinite(negative_logp)
    margin_by_row = (positive_logp - negative_logp).to(dtype=packed_logits.dtype)
    return PairedSwingComparisonRows(
        margin_by_row=margin_by_row,
        supported=supported,
        positive_logp_by_row=positive_logp,
        negative_logp_by_row=negative_logp,
    )


__all__ = ["PairedSwingComparisonRows", "paired_swing_margin_comparison_rows"]
