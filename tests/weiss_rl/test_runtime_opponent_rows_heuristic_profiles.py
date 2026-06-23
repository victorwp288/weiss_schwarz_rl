from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_CONTROL_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID

from .runtime_opponent_rows_test_support import (
    BaseNativeForbiddenPool,
    RecordingHeuristicPolicy,
    apply_fixed_opponent_rows,
    assert_heuristic_rows_written,
    make_fixed_opponent_rows,
    make_fixed_opponent_runtime_actor,
)


def test_apply_opponent_rows_ids_falls_back_for_profile_when_profile_native_hook_is_unavailable() -> None:
    heuristic_policy = RecordingHeuristicPolicy(actions=(10, 20))
    runtime, actor = make_fixed_opponent_runtime_actor(
        policy_id=HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
        heuristic_policy=heuristic_policy,
        pool=BaseNativeForbiddenPool(),
    )
    rows = make_fixed_opponent_rows()

    apply_fixed_opponent_rows(runtime, actor=actor, rows=rows)

    assert len(heuristic_policy.calls) == 1
    assert_heuristic_rows_written(actor, rows, [10, 20])


def test_apply_opponent_rows_ids_falls_back_when_simulator_native_pool_hook_is_unavailable() -> None:
    heuristic_policy = RecordingHeuristicPolicy()
    runtime, actor = make_fixed_opponent_runtime_actor(
        policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        heuristic_policy=heuristic_policy,
        pool=SimpleNamespace(),
    )
    rows = make_fixed_opponent_rows()

    apply_fixed_opponent_rows(runtime, actor=actor, rows=rows)

    assert len(heuristic_policy.calls) == 1
    call_obs, call_ids, call_offsets, call_meta = heuristic_policy.calls[0]
    assert np.array_equal(call_obs, np.asarray([[1, 0], [2, 0]], dtype=np.int32))
    assert np.array_equal(call_ids, np.asarray([10, 11, 12, 20, 21], dtype=np.uint32))
    assert np.array_equal(call_offsets, np.asarray([0, 3, 5], dtype=np.uint32))
    assert call_meta is not None
    assert np.array_equal(call_meta, rows.legal_action_meta)
    assert_heuristic_rows_written(actor, rows, [10, 20])
