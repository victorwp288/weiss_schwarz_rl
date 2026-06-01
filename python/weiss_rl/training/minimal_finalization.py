"""Compatibility surface for minimal training final checkpoint selection."""

from __future__ import annotations

from weiss_rl.training.checkpoint_finalization import (
    TrainingFinalCheckpointHooks as TrainingFinalCheckpointHooks,
)
from weiss_rl.training.checkpoint_finalization import (
    final_dev_eval_summary_for_update as _final_dev_eval_summary_for_update,
)
from weiss_rl.training.checkpoint_finalization import (
    finalize_training_checkpoint_selection as _finalize_training_checkpoint_selection,
)

__all__ = [
    "TrainingFinalCheckpointHooks",
    "_final_dev_eval_summary_for_update",
    "_finalize_training_checkpoint_selection",
]
