from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from weiss_rl.runtime.components.batching.learner_batch_inputs import (
    ImpalaLearnerBatchInputs,
    prepare_impala_learner_batch_inputs,
)
from weiss_rl.runtime.components.field_assembly import base_runtime_learner_payload


def impala_algorithm_payload(
    prepared: ImpalaLearnerBatchInputs,
    *,
    vtrace_rho_bar: float,
    vtrace_c_bar: float,
) -> dict[str, Any]:
    return {
        "bootstrap_obs": prepared.bootstrap.obs,
        "bootstrap_actor": prepared.bootstrap.actor,
        "final_hidden_state": prepared.bootstrap.final_hidden_state,
        "behavior_logp": prepared.fields.behavior_logp,
        "behavior_values": prepared.fields.values,
        "bootstrap_value": prepared.bootstrap.value,
        "vtrace_rho_bar": float(vtrace_rho_bar),
        "vtrace_c_bar": float(vtrace_c_bar),
        "terminal_outcome_backfill_count": prepared.backfill_metrics.outcome_count,
        "terminal_outcome_backfill_total_micros": prepared.backfill_metrics.outcome_total_micros,
        "terminal_outcome_trace_backfill_count": prepared.backfill_metrics.trace_count,
        "terminal_outcome_trace_backfill_total_micros": prepared.backfill_metrics.trace_total_micros,
    }


def impala_learner_payload(
    prepared: ImpalaLearnerBatchInputs,
    *,
    vtrace_rho_bar: float,
    vtrace_c_bar: float,
) -> dict[str, Any]:
    return {
        **base_runtime_learner_payload(
            fields=prepared.fields,
            rewards=prepared.rewards,
            discounts=prepared.discounts,
            reset_before_step=prepared.reset_before_step,
        ),
        **impala_algorithm_payload(
            prepared,
            vtrace_rho_bar=vtrace_rho_bar,
            vtrace_c_bar=vtrace_c_bar,
        ),
    }


def build_impala_learner_batch(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    gamma: float,
    truncation_reward: float,
    truncation_bootstrap_value: bool,
    vtrace_rho_bar: float,
    vtrace_c_bar: float,
    terminal_outcome_backfill_reward: float = 0.0,
    terminal_outcome_trace_backfill_reward: float = 0.0,
    record_batch_timer_ms: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    del truncation_reward, truncation_bootstrap_value
    prepared = prepare_impala_learner_batch_inputs(
        unrolls,
        action_dim=action_dim,
        gamma=gamma,
        terminal_outcome_backfill_reward=terminal_outcome_backfill_reward,
        terminal_outcome_trace_backfill_reward=terminal_outcome_trace_backfill_reward,
        record_batch_timer_ms=record_batch_timer_ms,
    )
    return impala_learner_payload(prepared, vtrace_rho_bar=vtrace_rho_bar, vtrace_c_bar=vtrace_c_bar)


__all__ = [
    "build_impala_learner_batch",
    "impala_algorithm_payload",
    "impala_learner_payload",
]
