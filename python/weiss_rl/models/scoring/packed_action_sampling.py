"""Packed legal-action sampling bridge used by model facades."""

from __future__ import annotations

from torch import Tensor


def sample_packed_action_scores(
    packed_scores: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    sample_seeds: Tensor,
    *,
    pass_action_id: int,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    # Resolve lazily through weiss_rl.model so the compatibility wrapper remains monkeypatchable.
    from weiss_rl import model as model_module

    return model_module._sample_packed_action_scores(
        packed_scores,
        packed_ids,
        packed_offsets,
        sample_seeds,
        pass_action_id=pass_action_id,
        temperature=temperature,
    )


__all__ = ["sample_packed_action_scores"]
