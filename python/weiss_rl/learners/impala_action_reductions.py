"""IMPALA learner action log-probability and entropy reductions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.action_logp import (
    masked_action_logp_and_entropy,
    packed_action_logp_and_entropy,
    packed_scores_action_logp_and_entropy,
    packed_scores_family_entropy,
)

TimingRecorder = Callable[[str, float], None]


@dataclass(frozen=True, slots=True)
class ImpalaActionReductions:
    action_logp: Tensor
    entropy: Tensor


def resolve_impala_action_reductions(
    *,
    factorized_result: Any,
    logits: Tensor | None,
    packed_logits: Tensor | None,
    legal_mask: Tensor | None,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    actions: Tensor,
    entropy_scope: str,
    pass_action_id: int | None,
    action_catalog: Any,
    record_timing_ms: TimingRecorder,
) -> ImpalaActionReductions:
    if factorized_result is not None:
        if factorized_result.action_logp is None or factorized_result.entropy is None:
            raise ValueError("factorized learner path requires action_logp and entropy")
        return ImpalaActionReductions(action_logp=factorized_result.action_logp, entropy=factorized_result.entropy)

    if packed_legal is not None:
        packed_reductions_started = time.perf_counter()
        packed_ids, packed_offsets, packed_meta = packed_legal
        if packed_logits is not None:
            action_logp, entropy = packed_scores_action_logp_and_entropy(
                packed_logits,
                packed_ids,
                packed_offsets,
                actions,
                pass_action_id=pass_action_id,
            )
            if entropy_scope == "family":
                if not isinstance(action_catalog, ActionCatalog) or packed_meta is None:
                    raise ValueError("family entropy requires packed legal-action metadata and action_catalog")
                entropy = packed_scores_family_entropy(
                    packed_logits,
                    packed_offsets,
                    packed_meta,
                    row_shape=actions.shape,
                    family_count=len(action_catalog.families),
                )
        else:
            assert logits is not None
            if entropy_scope == "family":
                raise ValueError("family entropy requires packed candidate logits")
            action_logp, entropy = packed_action_logp_and_entropy(
                logits,
                packed_ids,
                packed_offsets,
                actions,
                pass_action_id=pass_action_id,
            )
        record_timing_ms("learner_packed_reductions", time.perf_counter() - packed_reductions_started)
        return ImpalaActionReductions(action_logp=action_logp, entropy=entropy)

    assert legal_mask is not None
    assert logits is not None
    action_logp, entropy = masked_action_logp_and_entropy(
        logits,
        legal_mask,
        actions,
        pass_action_id=pass_action_id,
    )
    return ImpalaActionReductions(action_logp=action_logp, entropy=entropy)


__all__ = ["ImpalaActionReductions", "resolve_impala_action_reductions"]
