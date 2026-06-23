from __future__ import annotations

import numpy as np
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_AGGRO_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID

from .runtime_opponent_rows_test_support import (
    FailHeuristicPolicy,
    NativeBasePool,
    ProfileNativeForbiddenPool,
    RecordingHeuristicPolicy,
    apply_fixed_opponent_rows,
    assert_heuristic_rows_written,
    make_fixed_opponent_rows,
    make_fixed_opponent_runtime_actor,
)


def test_apply_opponent_rows_ids_uses_simulator_native_backend_for_heuristic_public() -> None:
    pool = NativeBasePool(actions=(11, 20))
    runtime, actor = make_fixed_opponent_runtime_actor(
        policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        heuristic_policy=FailHeuristicPolicy(),
        pool=pool,
    )
    rows = make_fixed_opponent_rows()

    apply_fixed_opponent_rows(runtime, actor=actor, rows=rows)

    assert len(pool.calls) == 1
    assert np.array_equal(pool.calls[0], np.asarray([0, 1], dtype=np.uint32))
    assert_heuristic_rows_written(actor, rows, [11, 20])


def test_apply_opponent_rows_ids_uses_python_profile_oracle_for_b3_b4_native_backend() -> None:
    pool = ProfileNativeForbiddenPool()
    heuristic_policy = RecordingHeuristicPolicy(actions=(11, 20))
    runtime, actor = make_fixed_opponent_runtime_actor(
        policy_id=HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
        heuristic_policy=heuristic_policy,
        pool=pool,
    )
    rows = make_fixed_opponent_rows()

    apply_fixed_opponent_rows(runtime, actor=actor, rows=rows)

    assert pool.profile_calls == []
    assert len(heuristic_policy.calls) == 1
    assert_heuristic_rows_written(actor, rows, [11, 20])
