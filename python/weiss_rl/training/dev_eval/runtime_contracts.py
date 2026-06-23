"""Runtime contract checks for periodic dev-eval."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.core.masking import assert_strictly_increasing_legal_ids


def evaluation_config_or_raise(stack: Any) -> Any:
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise RuntimeError("The locked stack is missing the evaluation config block")
    return evaluation


def validate_periodic_dev_eval_contract(stack: Any) -> Any:
    evaluation = evaluation_config_or_raise(stack)
    if not evaluation.seat_swap:
        raise RuntimeError("Periodic dev eval requires evaluation.seat_swap=true")
    if evaluation.eval_device != "cpu":
        raise RuntimeError(f"Periodic dev eval requires evaluation.eval_device='cpu', got {evaluation.eval_device!r}")
    if not evaluation.eval_inference_mode:
        raise RuntimeError("Periodic dev eval requires evaluation.eval_inference_mode=true")
    if evaluation.eval_sampling_algorithm not in {"pinned_cdf_pcg_v1", "model_argmax_pinned_v1"}:
        raise RuntimeError(
            "Periodic dev eval requires evaluation.eval_sampling_algorithm='pinned_cdf_pcg_v1' "
            "or 'model_argmax_pinned_v1', "
            f"got {evaluation.eval_sampling_algorithm!r}"
        )
    return evaluation


def legal_ids_for_env_row(
    *,
    batch: Any,
    env_index: int,
    require_sorted: bool,
) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError("Expected ids_offsets legality during periodic dev eval")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[env_index])
    end = int(legal_offsets[env_index + 1])
    row = np.asarray(legal_ids[start:end], dtype=np.uint32)
    if require_sorted:
        assert_strictly_increasing_legal_ids(row)
    return row


def should_run_periodic_dev_eval(stack: Any, *, update_count: int) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    interval = int(evaluation.periodic_dev_eval_interval_updates)
    return interval > 0 and update_count % interval == 0


__all__ = [
    "evaluation_config_or_raise",
    "legal_ids_for_env_row",
    "should_run_periodic_dev_eval",
    "validate_periodic_dev_eval_contract",
]
