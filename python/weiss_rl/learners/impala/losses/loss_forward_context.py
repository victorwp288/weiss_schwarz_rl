"""IMPALA policy-forward context and finite checks."""

from __future__ import annotations

from typing import Any


def build_impala_forward_context(
    *,
    learner: Any,
    batch: Any,
    forward_result: Any,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "logits": None if forward_result.logits is None else forward_result.logits.detach(),
        "packed_logits": None if forward_result.packed_logits is None else forward_result.packed_logits.detach(),
        "values": forward_result.values.detach(),
    }
    if forward_result.logits is not None:
        learner._ensure_finite_tensor("forward_logits", forward_result.logits, batch=batch, context=context)
    if forward_result.packed_logits is not None:
        learner._ensure_finite_tensor(
            "forward_packed_logits",
            forward_result.packed_logits,
            batch=batch,
            context=context,
        )
    learner._ensure_finite_tensor("forward_values", forward_result.values, batch=batch, context=context)
    return context


__all__ = ["build_impala_forward_context"]
