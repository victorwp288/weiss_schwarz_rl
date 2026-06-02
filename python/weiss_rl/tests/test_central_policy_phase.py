from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.runtime.components.central_policy_phase import (
    CentralPolicyPhaseOutputs,
    run_central_policy_phase,
)
from weiss_rl.runtime.components.collector_state import allocate_collector_unroll_state
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID
from weiss_rl.runtime.components.policy_inference.central_policy_outputs import (
    CentralPolicyPhaseOutputs as CanonicalOutputs,
)


def _state() -> Any:
    return allocate_collector_unroll_state(
        time_steps=1,
        batch_size=3,
        observation_dim=2,
        obs_dtype=np.float32,
        seat_hidden=torch.zeros((3, 2), dtype=torch.float32),
        trajectory_retention_enabled=False,
    )


def _actor(*, policies: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        actor_id=0,
        focal_seat_by_env=np.asarray([0, 0, 0], dtype=np.int64),
        opponent_policy_id_by_env=np.asarray(policies, dtype=object),
        rng=np.random.default_rng(3),
    )


def _packed_batch() -> SimpleNamespace:
    return SimpleNamespace(
        ids_offsets=(
            np.asarray([1, 2, 3, 4, 5, 6], dtype=np.uint32),
            np.asarray([0, 2, 4, 6], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray([[1], [2], [3], [4], [5], [6]], dtype=np.uint16),
    )


def test_central_policy_phase_reexports_canonical_output_contract() -> None:
    assert CentralPolicyPhaseOutputs is CanonicalOutputs


def test_run_central_policy_phase_structured_routes_focal_mirror_and_heuristic_rows() -> None:
    actor = _actor(policies=["focal", MIRROR_OPPONENT_POLICY_ID, "B2 HeuristicPublic"])
    state = _state()
    timers: list[str] = []
    sampled_rows: list[list[int]] = []
    advanced_rows: list[list[int]] = []
    opponent_calls: list[dict[str, Any]] = []

    def sample_policy(**kwargs: Any) -> None:
        rows = kwargs["row_indices_by_actor"][0]
        sampled_rows.append(rows.tolist())
        kwargs["actions_outs"][0][rows] = np.asarray([10 + int(row) for row in rows], dtype=np.int64)
        kwargs["logp_outs"][0][rows] = np.asarray([-0.1 * (int(row) + 1) for row in rows], dtype=np.float32)
        kwargs["values_outs"][0][rows] = np.asarray([1.0 + int(row) for row in rows], dtype=np.float32)

    def advance_rows(**kwargs: Any) -> None:
        advanced_rows.append(kwargs["row_indices_by_actor"][0].tolist())

    def apply_opponent(**kwargs: Any) -> None:
        rows = np.asarray(kwargs["row_indices"], dtype=np.int64)
        opponent_calls.append({"rows": rows.tolist(), "hidden": kwargs.get("heuristic_rows_hidden_already_advanced")})
        kwargs["actions_out"][rows] = np.asarray([40 + int(row) for row in rows], dtype=np.int64)
        kwargs["logp_out"][rows] = np.asarray([-4.0 - int(row) for row in rows], dtype=np.float32)
        kwargs["values_out"][rows] = np.asarray([4.0 + int(row) for row in rows], dtype=np.float32)

    outputs = run_central_policy_phase(
        actors=[actor],
        batches=[_packed_batch()],
        obs_steps=[np.ones((3, 2), dtype=np.float32)],
        actor_steps=[np.asarray([0, 1, 1], dtype=np.int64)],
        states_by_actor={0: state},
        batch_size=3,
        action_dim=8,
        structured_central_packed=True,
        disable_mirror_policy_fusion=False,
        opponent_heuristic_policy_ids=["B2 HeuristicPublic"],
        record_batch_timer_ms=lambda name, _elapsed: timers.append(name),
        central_sample_policy_rows_ids=sample_policy,
        central_advance_actor_rows=advance_rows,
        should_track_heuristic_actor_hidden_state=lambda: True,
        apply_opponent_rows_ids=apply_opponent,
        ensure_legal_action_meta=lambda legal_ids, legal_action_meta: legal_action_meta,
        central_forward_all_rows=lambda **_: None,
        overwrite_central_outputs_with_configured_opponents=lambda **_: None,
    )

    assert sampled_rows == [[0, 1]]
    assert advanced_rows == [[2]]
    assert opponent_calls == [{"rows": [2], "hidden": True}]
    assert outputs.logits_steps == [None]
    assert outputs.action_steps is not None
    assert outputs.logp_steps is not None
    assert outputs.action_steps[0].tolist() == [10, 11, 42]
    np.testing.assert_allclose(outputs.logp_steps[0], np.asarray([-0.1, -0.2, -6.0], dtype=np.float32))
    np.testing.assert_allclose(outputs.value_steps[0], np.asarray([1.0, 2.0, 6.0], dtype=np.float32))
    assert state.counters["focal_row_count"] == 1
    assert state.counters["opponent_row_count"] == 2
    assert state.counters["actor_policy_forward_ms"] >= 0
    assert state.counters["fixed_opponent_routing_ms"] >= 0
    assert timers == ["central_focal_policy", "central_fixed_opponent_overwrite"]


def test_run_central_policy_phase_dense_forwards_then_overwrites_outputs() -> None:
    actor = _actor(policies=["focal", "snapshot", "snapshot"])
    state = _state()
    calls: list[str] = []
    timers: list[str] = []

    def central_forward(**kwargs: Any) -> None:
        calls.append("forward")
        logits = kwargs["logits_outs"][0]
        logits[:, :] = np.arange(12, dtype=np.float32).reshape(3, 4)
        kwargs["values_outs"][0][:] = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)

    def overwrite(**kwargs: Any) -> None:
        calls.append("overwrite")
        logits = kwargs["logits_outs"][0]
        logits[1:, :] = -1.0
        kwargs["values_outs"][0][1:] = np.asarray([2.0, 3.0], dtype=np.float32)

    outputs = run_central_policy_phase(
        actors=[actor],
        batches=[_packed_batch()],
        obs_steps=[np.ones((3, 2), dtype=np.float32)],
        actor_steps=[np.asarray([0, 1, 1], dtype=np.int64)],
        states_by_actor={0: state},
        batch_size=3,
        action_dim=4,
        structured_central_packed=False,
        disable_mirror_policy_fusion=False,
        opponent_heuristic_policy_ids=[],
        record_batch_timer_ms=lambda name, _elapsed: timers.append(name),
        central_sample_policy_rows_ids=lambda **_: None,
        central_advance_actor_rows=lambda **_: None,
        should_track_heuristic_actor_hidden_state=lambda: False,
        apply_opponent_rows_ids=lambda **_: None,
        ensure_legal_action_meta=lambda legal_ids, legal_action_meta: legal_action_meta,
        central_forward_all_rows=central_forward,
        overwrite_central_outputs_with_configured_opponents=overwrite,
    )

    assert calls == ["forward", "overwrite"]
    assert outputs.action_steps is None
    assert outputs.logp_steps is None
    assert outputs.logits_steps[0] is not None
    np.testing.assert_allclose(outputs.logits_steps[0][0], np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(outputs.logits_steps[0][1:], -np.ones((2, 4), dtype=np.float32))
    np.testing.assert_allclose(outputs.value_steps[0], np.asarray([0.25, 2.0, 3.0], dtype=np.float32))
    assert state.counters["focal_row_count"] == 1
    assert state.counters["opponent_row_count"] == 2
    assert state.counters["actor_policy_forward_ms"] >= 0
    assert state.counters["fixed_opponent_routing_ms"] >= 0
    assert timers == ["central_focal_policy", "central_fixed_opponent_overwrite"]
