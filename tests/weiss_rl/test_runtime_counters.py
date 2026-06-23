from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.runtime.components.counters import (
    accumulate_actor_role_row_counters,
    accumulate_timeout_counters,
    collector_counter_template,
    merge_simulator_timing_counters,
    optional_int,
    packed_legal_views_from_step_out,
    timeout_limits_for_env,
)


def test_collector_counter_template_contains_timeout_and_simulator_keys() -> None:
    first = collector_counter_template()
    second = collector_counter_template()

    assert first is not second
    assert first["engine_fault_done_rows"] == 0
    assert first["no_progress_timeout_rows"] == 0
    assert first["natural_timeout_rows"] == 0
    assert first["simulator_step_sample_from_logits_with_logp_into_i16_legal_ids_count"] == 0

    first["engine_fault_done_rows"] = 3
    assert second["engine_fault_done_rows"] == 0


def test_accumulate_actor_role_row_counters_tracks_focal_and_opponent_rows() -> None:
    counters = collector_counter_template()

    focal_rows, opponent_rows = accumulate_actor_role_row_counters(
        counters=counters,
        actor_step=np.asarray([0, 1, 1, 0, 0], dtype=np.int64),
        focal_seat_by_env=np.asarray([0, 0, 1, 1, 0], dtype=np.int64),
    )

    assert focal_rows == 3
    assert opponent_rows == 2
    assert counters["focal_row_count"] == 3
    assert counters["opponent_row_count"] == 2


def test_accumulate_actor_role_row_counters_requires_matching_shapes() -> None:
    counters = collector_counter_template()

    with pytest.raises(ValueError, match="matching shapes"):
        accumulate_actor_role_row_counters(
            counters=counters,
            actor_step=np.asarray([0, 1], dtype=np.int64),
            focal_seat_by_env=np.asarray([0], dtype=np.int64),
        )


def test_timeout_limits_for_env_casts_present_limits_and_preserves_missing_limits() -> None:
    env = SimpleNamespace(max_decisions="12", max_ticks=np.int64(77), max_no_progress_decisions=None)

    assert optional_int(None) is None
    assert optional_int("9") == 9
    assert timeout_limits_for_env(env) == {
        "max_decisions": 12,
        "max_ticks": 77,
        "max_no_progress_decisions": None,
    }
    assert timeout_limits_for_env(SimpleNamespace()) == {
        "max_decisions": None,
        "max_ticks": None,
        "max_no_progress_decisions": None,
    }


def test_merge_simulator_timing_counters_accumulates_prefixed_drained_values() -> None:
    class TimingEnv:
        def drain_timing_counters(self) -> dict[str, int | str]:
            return {"select_actions_from_logits_count": 7, "new_counter": "4"}

    counters = {"simulator_select_actions_from_logits_count": 5}

    merge_simulator_timing_counters(counters, TimingEnv())
    merge_simulator_timing_counters(counters, SimpleNamespace(drain_timing_counters=None))

    assert counters["simulator_select_actions_from_logits_count"] == 12
    assert counters["simulator_new_counter"] == 4


def test_accumulate_timeout_counters_classifies_done_rows() -> None:
    counters = collector_counter_template()
    batch = SimpleNamespace(
        terminated=np.array([True, False, False, False, False], dtype=np.bool_),
        truncated=np.array([False, True, True, True, True], dtype=np.bool_),
        engine_status=np.array([0, 9, 0, 0, 0], dtype=np.int32),
        decision_count=np.array([0, 0, 10, 0, 0], dtype=np.int32),
        tick_count=np.array([0, 0, 0, 20, 0], dtype=np.int32),
        no_progress_count=np.array([0, 0, 0, 0, 3], dtype=np.int32),
    )

    accumulate_timeout_counters(
        counters=counters,
        batch=batch,
        done=np.array([True, True, True, True, True], dtype=np.bool_),
        timeout_limits={"max_decisions": 10, "max_ticks": 20, "max_no_progress_decisions": 3},
    )

    assert counters["engine_fault_done_rows"] == 1
    assert counters["no_progress_timeout_rows"] == 1
    assert counters["natural_timeout_rows"] == 2
    assert counters["decision_limit_timeout_rows"] == 1
    assert counters["tick_limit_timeout_rows"] == 1
    assert counters["timeout_unknown_rows"] == 0


def test_accumulate_timeout_counters_preserves_no_done_and_defaults_missing_counts() -> None:
    counters = collector_counter_template()
    batch = SimpleNamespace(
        terminated=np.array([False], dtype=np.bool_),
        truncated=np.array([True], dtype=np.bool_),
        engine_status=np.array([0], dtype=np.int32),
    )

    accumulate_timeout_counters(
        counters=counters,
        batch=batch,
        done=np.array([False], dtype=np.bool_),
        timeout_limits={"max_decisions": None, "max_ticks": None, "max_no_progress_decisions": None},
    )
    assert counters["natural_timeout_rows"] == 0

    accumulate_timeout_counters(
        counters=counters,
        batch=batch,
        done=np.array([True], dtype=np.bool_),
        timeout_limits={"max_decisions": None, "max_ticks": None, "max_no_progress_decisions": None},
    )
    assert counters["natural_timeout_rows"] == 1
    assert counters["timeout_unknown_rows"] == 1


def test_packed_legal_views_from_step_out_trims_to_last_offset_and_casts_dtypes() -> None:
    step_out = SimpleNamespace(
        legal_ids=np.array([9, 8, 7, 6], dtype=np.int64),
        legal_offsets=np.array([0, 1, 3], dtype=np.int64),
        legal_action_meta=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64),
    )

    legal_ids, legal_offsets, legal_action_meta = packed_legal_views_from_step_out(step_out)

    assert legal_ids.dtype == np.uint32
    assert legal_ids.tolist() == [9, 8, 7]
    assert legal_offsets.dtype == np.uint32
    assert legal_offsets.tolist() == [0, 1, 3]
    assert legal_action_meta is not None
    assert legal_action_meta.dtype == np.uint16
    assert legal_action_meta.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_packed_legal_views_from_step_out_keeps_missing_meta_none() -> None:
    step_out = SimpleNamespace(
        legal_ids=np.array([1, 2], dtype=np.uint32),
        legal_offsets=np.array([], dtype=np.uint32),
    )

    legal_ids, legal_offsets, legal_action_meta = packed_legal_views_from_step_out(step_out)

    assert legal_ids.tolist() == []
    assert legal_offsets.tolist() == []
    assert legal_action_meta is None
