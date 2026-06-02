"""Compatibility surface for minimal training checkpoint promotion."""

from __future__ import annotations

from weiss_rl.training.checkpointing.snapshot_promotion import (
    TrainingCheckpointPromotionHooks as TrainingCheckpointPromotionHooks,
)
from weiss_rl.training.checkpointing.snapshot_promotion import (
    league_reference_update_from_metrics as _league_reference_update_from_metrics,
)
from weiss_rl.training.checkpointing.snapshot_promotion import (
    maybe_checkpoint_and_promote_snapshot as _maybe_checkpoint_and_promote_snapshot,
)

__all__ = [
    "TrainingCheckpointPromotionHooks",
    "_league_reference_update_from_metrics",
    "_maybe_checkpoint_and_promote_snapshot",
]
