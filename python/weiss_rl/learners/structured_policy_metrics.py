"""Structured policy metric summaries for learner logging."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView,
    packed_structured_legal_view,
    structured_catalog_metadata,
)
from weiss_rl.learners.tensor_ops import segment_group_sum, segment_max


def summarize_structured_policy_metrics(
    logits: Tensor | None,
    legal_mask: Tensor | None,
    *,
    action_catalog: ActionCatalog,
    packed_ids: Tensor | None = None,
    packed_offsets: Tensor | None = None,
    packed_meta: Tensor | None = None,
    packed_view: PackedStructuredLegalView | None = None,
    factorized_family_log_probs: Tensor | None = None,
) -> dict[str, float]:
    catalog_metadata = structured_catalog_metadata(action_catalog)
    family_names = catalog_metadata.family_names
    family_index = {name: index for index, name in enumerate(family_names)}
    main_move_02_action_id = catalog_metadata.main_move_02_action_id
    if factorized_family_log_probs is not None:
        family_probs = torch.exp(factorized_family_log_probs.detach().to(dtype=torch.float32))
        move_family_id = family_index.get("main_move", -1)
        play_family_id = family_index.get("main_play_character", -1)
        pass_family_id = family_index.get("pass", -1)
        metrics = {
            "structured_exact_action_concentration": float(family_probs.max(dim=-1).values.mean().item()),
            "structured_main_play_character_mass": float(
                family_probs[..., play_family_id].mean().item() if play_family_id >= 0 else 0.0
            ),
            "structured_main_move_mass": float(
                family_probs[..., move_family_id].mean().item() if move_family_id >= 0 else 0.0
            ),
            "structured_pass_mass": float(
                family_probs[..., pass_family_id].mean().item() if pass_family_id >= 0 else 0.0
            ),
        }
        return metrics

    packed_view = (
        packed_view
        if packed_view is not None
        else packed_structured_legal_view(
            logits=logits,
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
    )
    if packed_view is not None and bool(packed_view.row_has_candidates.any().item()):
        row_log_z = packed_view.row_log_z.index_select(0, packed_view.row_indices)
        probs = torch.exp(packed_view.logits - row_log_z)
        top_logits = segment_max(packed_view.logits, packed_view.row_indices, packed_view.row_count)
        non_empty = packed_view.row_has_candidates
        top1_confidence = torch.exp(top_logits[non_empty] - packed_view.row_log_z[non_empty])
        packed_family_mass = segment_group_sum(
            probs,
            packed_view.row_indices,
            packed_view.family_ids,
            row_count=packed_view.row_count,
            group_count=len(family_names),
        )
        play_family_id = family_index.get("main_play_character", -1)
        move_family_id = family_index.get("main_move", -1)
        pass_family_id = family_index.get("pass", -1)
        play_mass = (
            packed_family_mass[:, play_family_id]
            if play_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=probs.dtype, device=probs.device)
        )
        move_mass = (
            packed_family_mass[:, move_family_id]
            if move_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=probs.dtype, device=probs.device)
        )
        pass_mass = (
            packed_family_mass[:, pass_family_id]
            if pass_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=probs.dtype, device=probs.device)
        )
        legal_play_available = (
            segment_group_sum(
                torch.ones_like(probs),
                packed_view.row_indices,
                packed_view.family_ids,
                row_count=packed_view.row_count,
                group_count=len(family_names),
            )[:, play_family_id]
            > 0
            if play_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=torch.bool, device=probs.device)
        )

        metrics = {
            "structured_exact_action_concentration": float(top1_confidence.mean().item()),
            "structured_main_play_character_mass": float(play_mass[non_empty].mean().item()),
            "structured_main_move_mass": float(move_mass[non_empty].mean().item()),
            "structured_pass_mass": float(pass_mass[non_empty].mean().item()),
        }
        legal_play_rows = non_empty & legal_play_available
        if bool(legal_play_rows.any().item()):
            metrics["structured_main_move_share_when_play_available"] = float(move_mass[legal_play_rows].mean().item())
        if main_move_02_action_id is not None:
            mm_mask = packed_view.action_ids == int(main_move_02_action_id)
            mm_top_rows = torch.zeros((packed_view.row_count,), dtype=torch.bool, device=packed_view.logits.device)
            if bool(mm_mask.any().item()):
                mm_rows = packed_view.row_indices[mm_mask]
                mm_is_top = packed_view.logits[mm_mask] >= top_logits.index_select(0, mm_rows) - 1.0e-6
                mm_top_rows[mm_rows[mm_is_top]] = True
            metrics["structured_main_move_0_2_top1_rate"] = float(mm_top_rows[non_empty].float().mean().item())
        return metrics

    if logits is None or legal_mask is None:
        return {}
    flat_logits = logits.detach().to(dtype=torch.float32).reshape(-1, logits.shape[-1])
    flat_mask = legal_mask.detach().to(dtype=torch.bool).reshape(-1, legal_mask.shape[-1])
    non_empty = flat_mask.any(dim=1)
    if not bool(non_empty.any().item()):
        return {}

    masked_logits = torch.where(
        flat_mask[non_empty],
        flat_logits[non_empty],
        torch.full_like(flat_logits[non_empty], -1.0e9),
    )
    probs = torch.softmax(masked_logits, dim=1)
    top1_ids = probs.argmax(dim=1)
    top1_confidence = probs.max(dim=1).values

    family_ids = torch.as_tensor(catalog_metadata.family_ids, dtype=torch.long, device=flat_logits.device)

    def family_mass(name: str) -> Tensor:
        family_id = family_index.get(name, -1)
        if family_id < 0:
            return torch.zeros((probs.shape[0],), dtype=probs.dtype, device=probs.device)
        mask = family_ids == family_id
        if not bool(mask.any().item()):
            return torch.zeros((probs.shape[0],), dtype=probs.dtype, device=probs.device)
        return probs[:, mask].sum(dim=1)

    play_mass = family_mass("main_play_character")
    move_mass = family_mass("main_move")
    pass_mass = family_mass("pass")
    legal_play_available = flat_mask[non_empty][:, family_ids == family_index.get("main_play_character", -1)].any(dim=1)

    metrics = {
        "structured_exact_action_concentration": float(top1_confidence.mean().item()),
        "structured_main_play_character_mass": float(play_mass.mean().item()),
        "structured_main_move_mass": float(move_mass.mean().item()),
        "structured_pass_mass": float(pass_mass.mean().item()),
    }
    if bool(legal_play_available.any().item()):
        metrics["structured_main_move_share_when_play_available"] = float(move_mass[legal_play_available].mean().item())
    if main_move_02_action_id is not None:
        metrics["structured_main_move_0_2_top1_rate"] = float(
            (top1_ids == int(main_move_02_action_id)).to(dtype=torch.float32).mean().item()
        )
    return metrics
