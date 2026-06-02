"""Central policy-routing phase for batched actor collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.runtime.components.central_dense_policy_phase import (
    run_dense_central_policy_phase as _run_dense_central_policy_phase,
)
from weiss_rl.runtime.components.central_structured_policy_phase import (
    run_structured_central_policy_phase as _run_structured_central_policy_phase,
)
from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.policy_inference.central_policy_outputs import (
    CentralPolicyPhaseOutputs as CentralPolicyPhaseOutputs,
)

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def run_central_policy_phase(
    *,
    actors: Sequence[_ActorState],
    batches: Sequence[Any],
    obs_steps: Sequence[np.ndarray],
    actor_steps: Sequence[np.ndarray],
    states_by_actor: Mapping[int, CollectorUnrollState],
    batch_size: int,
    action_dim: int,
    structured_central_packed: bool,
    disable_mirror_policy_fusion: bool,
    opponent_heuristic_policy_ids: Sequence[str],
    record_batch_timer_ms: Callable[[str, float], None],
    central_sample_policy_rows_ids: Callable[..., None],
    central_advance_actor_rows: Callable[..., None],
    should_track_heuristic_actor_hidden_state: Callable[[], bool],
    apply_opponent_rows_ids: Callable[..., None],
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
    central_forward_all_rows: Callable[..., None],
    overwrite_central_outputs_with_configured_opponents: Callable[..., None],
) -> CentralPolicyPhaseOutputs:
    if structured_central_packed:
        return _run_structured_central_policy_phase(
            actors=actors,
            batches=batches,
            obs_steps=obs_steps,
            actor_steps=actor_steps,
            states_by_actor=states_by_actor,
            batch_size=batch_size,
            opponent_heuristic_policy_ids=opponent_heuristic_policy_ids,
            fuse_mirror_policy_rows=not bool(disable_mirror_policy_fusion),
            record_batch_timer_ms=record_batch_timer_ms,
            central_sample_policy_rows_ids=central_sample_policy_rows_ids,
            central_advance_actor_rows=central_advance_actor_rows,
            should_track_heuristic_actor_hidden_state=should_track_heuristic_actor_hidden_state,
            apply_opponent_rows_ids=apply_opponent_rows_ids,
            ensure_legal_action_meta=ensure_legal_action_meta,
        )
    return _run_dense_central_policy_phase(
        actors=actors,
        batches=batches,
        obs_steps=obs_steps,
        actor_steps=actor_steps,
        states_by_actor=states_by_actor,
        batch_size=batch_size,
        action_dim=action_dim,
        record_batch_timer_ms=record_batch_timer_ms,
        central_forward_all_rows=central_forward_all_rows,
        overwrite_central_outputs_with_configured_opponents=overwrite_central_outputs_with_configured_opponents,
    )
