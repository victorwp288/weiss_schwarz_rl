"""Structured-head public heuristic logit-bias adapter."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from torch import Tensor

from weiss_rl.models.public_heuristic.public_heuristic_bias import apply_public_heuristic_bias


class StructuredPublicHeuristicBiasMixin:
    """Resolve and apply the optional public-heuristic logit bias."""

    def _public_heuristic_logit_bias_scale_for(self, scoring_mode: str) -> float:
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "actor":
            return float(self._public_heuristic_actor_logit_bias_scale)
        return float(self._public_heuristic_logit_bias_scale)

    def _apply_public_heuristic_bias(
        self,
        scores: Tensor,
        raw_scores: Tensor,
        *,
        scale: float,
        family_ids: Tensor | None = None,
    ) -> Tensor:
        return apply_public_heuristic_bias(
            scores,
            raw_scores,
            scale=scale,
            family_ids=family_ids,
            bias_family_ids=self._public_heuristic_bias_family_ids,
        )


__all__ = ["StructuredPublicHeuristicBiasMixin"]
