"""Dense legal-mask resolution for IMPALA loss inputs."""

from __future__ import annotations

from typing import Any

from torch import Tensor


def resolve_impala_dense_legal_mask(
    *,
    learner: Any,
    batch: Any,
    obs: Tensor,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    logits: Tensor | None,
) -> Tensor | None:
    if packed_legal is not None:
        return None
    if logits is None:
        raise ValueError("dense learner path requires dense logits")
    legal_mask = learner._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
    if legal_mask.shape != logits.shape:
        raise ValueError("legal_mask must match learner logits on time, batch, and action dimensions")
    return legal_mask


__all__ = ["resolve_impala_dense_legal_mask"]
