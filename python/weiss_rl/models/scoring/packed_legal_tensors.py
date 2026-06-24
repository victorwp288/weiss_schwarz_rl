"""Torch tensor view of packed legal-action payloads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch


@dataclass(frozen=True, slots=True)
class PackedLegalTensors:
    ids: Tensor
    offsets: Tensor
    meta: Tensor | None

    @property
    def candidate_count(self) -> int:
        return int(self.ids.numel())


def require_packed_legal_tensors(
    legal_actions: LegalActionBatch,
    *,
    device: torch.device,
    row_count: int,
    require_meta: bool = False,
    missing_message: str = "packed legal actions require ids and offsets",
) -> PackedLegalTensors:
    if legal_actions.ids is None or legal_actions.offsets is None:
        raise ValueError(missing_message)
    if require_meta and legal_actions.meta is None:
        raise ValueError(missing_message)
    ids = torch.as_tensor(legal_actions.ids, device=device, dtype=torch.long)
    offsets = torch.as_tensor(legal_actions.offsets, device=device, dtype=torch.long)
    meta = (
        None
        if legal_actions.meta is None
        else torch.as_tensor(legal_actions.meta, device=device, dtype=torch.long)
    )
    validate_packed_offsets(ids=ids, offsets=offsets, row_count=row_count)
    return PackedLegalTensors(ids=ids, offsets=offsets, meta=meta)


def validate_packed_offsets(*, ids: Tensor, offsets: Tensor, row_count: int) -> None:
    if offsets.ndim != 1 or offsets.numel() != int(row_count) + 1:
        raise ValueError(f"packed legal offsets must have shape ({int(row_count) + 1},)")
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != int(ids.numel()):
        raise ValueError("packed legal offsets must be a valid prefix sum")


__all__ = ["PackedLegalTensors", "require_packed_legal_tensors", "validate_packed_offsets"]
