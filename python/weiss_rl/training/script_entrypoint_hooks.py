"""Script-level callback facade for the path-based training entrypoint."""

from __future__ import annotations

from weiss_rl.training.script_entrypoint_best_checkpoint_hooks import (
    FinalizeFromBestCheckpointRequest,
    RollbackToBestCheckpointRequest,
    maybe_finalize_from_best_checkpoint_with_script_hooks,
    maybe_rollback_to_best_checkpoint_with_script_hooks,
)
from weiss_rl.training.script_entrypoint_current_checkpoint_hooks import (
    EnsureCurrentCheckpointRequest,
    ensure_current_checkpoint_with_script_hooks,
)
from weiss_rl.training.script_entrypoint_dev_eval_hooks import (
    PeriodicDevEvalOpponentsRequest,
    PeriodicDevEvalRequest,
    StallMonitorRequest,
    periodic_dev_eval_opponents_with_script_hooks,
    run_periodic_dev_eval_with_script_hooks,
    update_stall_monitor_with_script_hooks,
)
from weiss_rl.training.script_entrypoint_promotion_hooks import (
    SnapshotPromotionGateRequest,
    run_snapshot_promotion_gate_with_script_hooks,
)

__all__ = [
    "EnsureCurrentCheckpointRequest",
    "FinalizeFromBestCheckpointRequest",
    "PeriodicDevEvalOpponentsRequest",
    "PeriodicDevEvalRequest",
    "RollbackToBestCheckpointRequest",
    "SnapshotPromotionGateRequest",
    "StallMonitorRequest",
    "ensure_current_checkpoint_with_script_hooks",
    "maybe_finalize_from_best_checkpoint_with_script_hooks",
    "maybe_rollback_to_best_checkpoint_with_script_hooks",
    "periodic_dev_eval_opponents_with_script_hooks",
    "run_periodic_dev_eval_with_script_hooks",
    "run_snapshot_promotion_gate_with_script_hooks",
    "update_stall_monitor_with_script_hooks",
]
