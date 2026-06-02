from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.runtime.components.central_finalization import (
    build_central_runtime_unrolls,
    compute_central_bootstrap_values,
)
from weiss_rl.runtime.components.central_unroll_assembly import (
    build_central_runtime_unrolls as build_central_runtime_unrolls_canonical,
)
from weiss_rl.runtime.components.collector_state import allocate_collector_unroll_state


class _TimingEnv:
    def __init__(self) -> None:
        self.drained = False

    def drain_timing_counters(self) -> dict[str, int]:
        self.drained = True
        return {"python_step": 7}


def _state() -> Any:
    state = allocate_collector_unroll_state(
        time_steps=1,
        batch_size=2,
        observation_dim=2,
        obs_dtype=np.float32,
        seat_hidden=torch.zeros((2, 2), dtype=torch.float32),
        trajectory_retention_enabled=False,
    )
    state.packed_ids.append(np.asarray([4, 5], dtype=np.uint32))
    state.packed_offsets.append(np.asarray([1, 2], dtype=np.uint32))
    return state


def _actor(*, actor_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        actor_id=actor_id,
        next_unroll_seq=3,
        snapshot_version=9,
        layout_name="i16_legal_ids",
        seat_hidden=torch.ones((2, 2), dtype=torch.float32),
        current_batch=None,
        env=_TimingEnv(),
    )


def _batch() -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.asarray([[1.25, 2.25], [3.25, 4.25]], dtype=np.float64),
        actor=np.asarray([0, 1], dtype=np.int32),
    )


def test_central_finalization_reexports_canonical_unroll_assembly() -> None:
    assert build_central_runtime_unrolls is build_central_runtime_unrolls_canonical
    assert build_central_runtime_unrolls.__module__ == "weiss_rl.runtime.components.central_unroll_assembly"


def test_compute_central_bootstrap_values_uses_structured_value_rows_without_overwrite() -> None:
    actor = _actor()
    state = _state()
    calls: list[str] = []

    def central_value_actor_rows(**kwargs: Any) -> None:
        calls.append("value")
        assert kwargs["row_indices_by_actor"][0].tolist() == [0, 1]
        kwargs["values_outs"][0][:] = np.asarray([0.5, 0.75], dtype=np.float32)

    def central_forward_all_rows(**_: Any) -> None:
        calls.append("forward")

    def overwrite(**_: Any) -> None:
        calls.append("overwrite")

    obs_steps, actor_steps, values = compute_central_bootstrap_values(
        actors=[actor],
        batches=[_batch()],
        states_by_actor={0: state},
        batch_size=2,
        action_dim=4,
        structured_central_packed=True,
        values_required=True,
        central_value_actor_rows=central_value_actor_rows,
        central_forward_all_rows=central_forward_all_rows,
        overwrite_central_outputs_with_configured_opponents=overwrite,
    )

    assert calls == ["value"]
    assert obs_steps[0].dtype == np.float32
    assert actor_steps[0].dtype == np.int64
    np.testing.assert_allclose(values[0], np.asarray([0.5, 0.75], dtype=np.float32))
    assert state.counters["actor_bootstrap_ms"] >= 0
    assert state.counters["fixed_opponent_routing_ms"] == 0


def test_compute_central_bootstrap_values_for_dense_path_forwards_then_overwrites() -> None:
    actor = _actor()
    state = _state()
    calls: list[str] = []

    def central_value_actor_rows(**_: Any) -> None:
        calls.append("value")

    def central_forward_all_rows(**kwargs: Any) -> None:
        calls.append("forward")
        assert kwargs["logits_outs"][0].shape == (2, 5)
        kwargs["values_outs"][0][:] = np.asarray([1.5, 1.75], dtype=np.float32)

    def overwrite(**kwargs: Any) -> None:
        calls.append("overwrite")
        assert kwargs["logits_outs"] == [None]
        kwargs["values_outs"][0][:] += np.asarray([2.0, 3.0], dtype=np.float32)

    _obs_steps, _actor_steps, values = compute_central_bootstrap_values(
        actors=[actor],
        batches=[_batch()],
        states_by_actor={0: state},
        batch_size=2,
        action_dim=5,
        structured_central_packed=False,
        values_required=True,
        central_value_actor_rows=central_value_actor_rows,
        central_forward_all_rows=central_forward_all_rows,
        overwrite_central_outputs_with_configured_opponents=overwrite,
    )

    assert calls == ["forward", "overwrite"]
    np.testing.assert_allclose(values[0], np.asarray([3.5, 4.75], dtype=np.float32))
    assert state.counters["actor_bootstrap_ms"] >= 0
    assert state.counters["fixed_opponent_routing_ms"] >= 0


def test_compute_central_bootstrap_values_skips_model_calls_when_values_not_required() -> None:
    actor = _actor()
    state = _state()
    calls: list[str] = []

    _obs_steps, _actor_steps, values = compute_central_bootstrap_values(
        actors=[actor],
        batches=[_batch()],
        states_by_actor={0: state},
        batch_size=2,
        action_dim=5,
        structured_central_packed=False,
        values_required=False,
        central_value_actor_rows=lambda **_: calls.append("value"),
        central_forward_all_rows=lambda **_: calls.append("forward"),
        overwrite_central_outputs_with_configured_opponents=lambda **_: calls.append("overwrite"),
    )

    assert calls == []
    np.testing.assert_allclose(values[0], np.zeros((2,), dtype=np.float32))
    assert state.counters["actor_bootstrap_ms"] == 0
    assert state.counters["fixed_opponent_routing_ms"] == 0


def test_build_central_runtime_unrolls_updates_actor_and_packages_copied_fields() -> None:
    actor = _actor(actor_id=4)
    state = _state()
    batch = _batch()

    unrolls = build_central_runtime_unrolls(
        actors=[actor],
        batches=[batch],
        bootstrap_values=[np.asarray([0.25, 0.5], dtype=np.float64)],
        states_by_actor={4: state},
        action_dim=8,
        central_started=time.perf_counter(),
    )

    assert len(unrolls) == 1
    unroll = unrolls[0]
    assert actor.current_batch is batch
    assert actor.next_unroll_seq == 4
    assert actor.env.drained is True
    assert state.counters["simulator_python_step"] == 7
    assert state.counters["collect_actor_unroll_ms"] >= 0
    assert state.counters["copied_bytes_estimate"] > 0
    assert unroll.actor_id == 4
    assert unroll.unroll_seq == 3
    assert unroll.behavior_policy_version == 9
    assert unroll.bootstrap_obs.dtype == np.float32
    assert unroll.bootstrap_actor.dtype == np.int64
    assert unroll.bootstrap_value.dtype == np.float32
    assert unroll.legal_actions.ids is not None
    assert unroll.legal_actions.ids.tolist() == [4, 5]
    assert unroll.counters is not state.counters
    assert unroll.counters is not None
    assert unroll.counters["simulator_python_step"] == 7

    actor.seat_hidden.fill_(99.0)
    np.testing.assert_allclose(unroll.final_hidden_state, np.ones((2, 2), dtype=np.float32))
