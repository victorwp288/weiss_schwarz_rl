from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.training.dev_eval import (
    legal_ids_for_env_row,
    periodic_dev_eval_bootstrap_seed,
    periodic_dev_eval_rng_seed,
    periodic_dev_eval_summaries_path,
    promotion_gate_bootstrap_seed,
    promotion_gate_rng_seed,
    should_run_periodic_dev_eval,
    stall_monitor_state_path,
    validate_periodic_dev_eval_contract,
)

from .training_dev_eval_test_support import make_dev_eval_stack


def test_legal_ids_for_env_row_slices_packed_legality_rows() -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([2, 4, 9, 11, 13], dtype=np.uint32),
            np.asarray([0, 2, 5], dtype=np.int64),
        )
    )

    row = legal_ids_for_env_row(batch=batch, env_index=1, require_sorted=True)

    assert row.dtype == np.uint32
    assert row.tolist() == [9, 11, 13]


def test_legal_ids_for_env_row_rejects_missing_offsets() -> None:
    batch = SimpleNamespace(ids_offsets=None)

    with pytest.raises(RuntimeError, match="Expected ids_offsets legality during periodic dev eval"):
        legal_ids_for_env_row(batch=batch, env_index=0, require_sorted=False)


def test_legal_ids_for_env_row_can_enforce_sorted_ids() -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([5, 3], dtype=np.uint32),
            np.asarray([0, 2], dtype=np.int64),
        )
    )

    assert legal_ids_for_env_row(batch=batch, env_index=0, require_sorted=False).tolist() == [5, 3]
    with pytest.raises(ValueError, match="strictly increasing"):
        legal_ids_for_env_row(batch=batch, env_index=0, require_sorted=True)


def test_periodic_dev_eval_contract_preserves_public_failures(tmp_path) -> None:
    stack = make_dev_eval_stack(tmp_path)
    validate_periodic_dev_eval_contract(stack)
    stack.config.evaluation.eval_device = "cuda"

    with pytest.raises(RuntimeError, match="evaluation.eval_device='cpu'"):
        validate_periodic_dev_eval_contract(stack)


def test_periodic_dev_eval_contract_accepts_model_argmax_sampling(tmp_path) -> None:
    stack = make_dev_eval_stack(tmp_path)
    stack.config.evaluation.eval_sampling_algorithm = "model_argmax_pinned_v1"

    validate_periodic_dev_eval_contract(stack)


def test_periodic_and_promotion_rng_seed_helpers_are_stable_and_distinct() -> None:
    scheduled_game = SimpleNamespace(
        pair_index=3,
        swap_index=1,
        episode_seed=123456,
        seat0_policy_id="a",
        seat1_policy_id="b",
    )

    first = periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=0)
    assert periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=0) == first
    assert periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=1) != first
    assert promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=0) != first
    assert periodic_dev_eval_bootstrap_seed(update_count=10, policy_version=2) != promotion_gate_bootstrap_seed(
        update_count=10,
        policy_version=2,
    )


def test_periodic_dev_eval_paths_and_interval_predicate(tmp_path) -> None:
    stack = make_dev_eval_stack(tmp_path)
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")

    assert should_run_periodic_dev_eval(stack, update_count=40)
    assert not should_run_periodic_dev_eval(stack, update_count=41)
    stack.config.evaluation.periodic_dev_eval_interval_updates = 0
    assert not should_run_periodic_dev_eval(stack, update_count=40)
    assert periodic_dev_eval_summaries_path(paths) == tmp_path / "logs" / "periodic_dev_eval_summaries.json"
    assert stall_monitor_state_path(paths) == tmp_path / "logs" / "stall_monitor.json"
