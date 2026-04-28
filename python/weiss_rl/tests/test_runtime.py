from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest
import torch

import weiss_rl.model as model_module
import weiss_rl.runtime as runtime_module
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
)
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.runtime import (
    _MIRROR_OPPONENT_POLICY_ID,
    _NOLEAGUE_BASELINE_POLICY_ID,
    QueueRuntime,
    QueueRuntimeConfig,
    RuntimeUnroll,
    _concat_optional_time_major_field,
    _concatenate_legal_actions,
    _create_shared_collector_slot_config,
    _gae_advantages,
    _handle_collector_commands,
    _maybe_compile_runtime_actor_model,
    _open_shared_collector_slot,
    _read_unroll_from_shared_slot,
    _resolve_actor_topology,
    _shared_unroll_metadata,
    _SharedPendingUnroll,
    _write_unroll_to_shared_slot,
    build_runtime_config,
    resolve_actor_device_layout,
)
from weiss_rl.residual_policy import FrozenStoredLogitResidual, LiveFrozenB1Residual, TrainableLiveFrozenB1Residual


def _make_runtime_unroll(
    *,
    actor_id: int,
    unroll_seq: int,
    behavior_policy_version: int,
    counters: dict[str, int] | None = None,
) -> RuntimeUnroll:
    return RuntimeUnroll(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
        unroll_hash=f"{actor_id}:{unroll_seq}:{behavior_policy_version}",
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.int64),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int64),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 1), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
        b1_opponent_mask=np.zeros((1, 1), dtype=np.bool_),
        behavior_logits=None,
        counters=counters,
    )


class _CompileStructuredActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.compile_calls = 0
        self.compile_mode: str | None = None

    def enable_trunk_compile(self, *, mode: str = "reduce-overhead") -> _CompileStructuredActorModel:
        self.compile_calls += 1
        self.compile_mode = mode
        return self


def test_maybe_compile_runtime_actor_model_uses_structured_trunk_compile_hook() -> None:
    model = _CompileStructuredActorModel()

    compiled = _maybe_compile_runtime_actor_model(model, enabled=True)

    assert compiled is model
    assert model.compile_calls == 1
    assert model.compile_mode == "reduce-overhead"


class _FactorizedRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.supports_factorized_legal_policy = True
        self.factorized_calls = 0
        self.packed_calls = 0

    def sample_factorized_packed_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: torch.Tensor,
        pass_action_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del acting_seat, legal_actions, sample_seeds, pass_action_id
        self.factorized_calls += 1
        batch = int(obs.shape[0])
        next_hidden = (
            torch.zeros((batch, 2, 3), dtype=obs.dtype, device=obs.device)
            if seat_hidden_state is None
            else seat_hidden_state + 1.0
        )
        return (
            torch.full((batch,), 7, dtype=torch.long, device=obs.device),
            torch.full((batch,), -0.25, dtype=obs.dtype, device=obs.device),
            torch.full((batch,), 0.5, dtype=obs.dtype, device=obs.device),
            next_hidden,
        )

    def sample_packed_seat_aware(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del args, kwargs
        self.packed_calls += 1
        raise AssertionError("factorized runtime path should bypass packed sampler")


def test_apply_policy_rows_ids_prefers_factorized_structured_sampler() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any.config = SimpleNamespace(pass_action_id=8)

    model = _FactorizedRuntimeActorModel()
    hidden_state = torch.zeros((2, 2, 3), dtype=torch.float32)
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.asarray([0, 1], dtype=np.int64)
    legal_ids = np.asarray([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    legal_meta = np.asarray(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.zeros((2,), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._apply_policy_rows_ids(
        runtime,
        model=model,
        hidden_state=hidden_state,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(123),
        sample_actions=True,
    )

    assert model.factorized_calls == 1
    assert model.packed_calls == 0
    assert actions_out.tolist() == [7, 7]
    assert np.allclose(logp_out, -0.25)
    assert np.allclose(values_out, 0.5)
    assert torch.allclose(hidden_state, torch.ones_like(hidden_state))


def test_sync_actor_batch_from_step_out_updates_env_last_batch() -> None:
    runtime = object.__new__(QueueRuntime)
    stale_batch = SimpleNamespace(
        ids_offsets=(
            np.array([51], dtype=np.uint32),
            np.array([0, 0, 1], dtype=np.uint32),
        )
    )
    actor = SimpleNamespace(
        current_batch=stale_batch,
        env=SimpleNamespace(_last_batch=stale_batch),
    )
    step_out = SimpleNamespace(
        obs=np.zeros((2, 3), dtype=np.float32),
        rewards=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=np.bool_),
        truncated=np.zeros((2,), dtype=np.bool_),
        actor=np.array([0, 1], dtype=np.int64),
        decision_kind=np.zeros((2,), dtype=np.int32),
        decision_id=np.array([11, 12], dtype=np.uint32),
        engine_status=np.zeros((2,), dtype=np.uint32),
        decision_count=np.zeros((2,), dtype=np.uint32),
        tick_count=np.zeros((2,), dtype=np.uint32),
        no_progress_count=np.zeros((2,), dtype=np.uint32),
        episode_seed=np.array([101, 202], dtype=np.uint64),
        episode_key=np.array([301, 402], dtype=np.uint64),
        legal_ids=np.array([51, 474, 51, 102], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4], dtype=np.uint32),
    )
    pool = SimpleNamespace(action_space=512)

    batch = runtime._sync_actor_batch_from_step_out(
        actor=actor,
        step_out=step_out,
        pool=pool,
    )

    assert actor.current_batch is batch
    assert actor.env._last_batch is batch
    assert actor.current_batch is not stale_batch
    npt.assert_array_equal(batch.ids_offsets[0], np.array([51, 474, 51, 102], dtype=np.uint32))
    npt.assert_array_equal(batch.ids_offsets[1], np.array([0, 2, 4], dtype=np.uint32))


class _HeuristicRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.forward_legal_flags: list[bool] = []

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del acting_seat
        self.forward_legal_flags.append(legal_actions is not None)
        batch = int(obs.shape[0])
        next_hidden = (
            torch.zeros((batch, 2, 3), dtype=obs.dtype, device=obs.device)
            if seat_hidden_state is None
            else seat_hidden_state + 1.0
        )
        return (
            torch.zeros((batch, 8), dtype=obs.dtype, device=obs.device),
            torch.full((batch,), 0.25, dtype=obs.dtype, device=obs.device),
            next_hidden,
        )


class _HeuristicAdvanceOnlyRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.advance_calls = 0
        self.forward_calls = 0

    def advance_seat_hidden(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del obs, acting_seat
        self.advance_calls += 1
        if seat_hidden_state is None:
            return torch.zeros((0, 2, 3), dtype=self.bias.dtype, device=self.bias.device)
        return seat_hidden_state + 1.0

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del obs, acting_seat, seat_hidden_state, legal_actions
        self.forward_calls += 1
        raise AssertionError(
            "heuristic actor rows should use advance_seat_hidden when behavior values are not required"
        )


def test_fill_policy_outputs_ids_can_use_heuristic_actor_backend_for_focal_rows() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._teacher_policy = object()
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    runtime_any._heuristic_public_actions_from_ids = lambda **_kwargs: np.array([7, 8], dtype=np.int64)

    model = _HeuristicRuntimeActorModel()
    actor = cast(
        Any,
        SimpleNamespace(
            model=model,
            compiled_model=None,
            seat_hidden=torch.zeros((2, 2, 3), dtype=torch.float32),
            focal_seat_by_env=np.array([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.array(["mirror", "mirror"], dtype=object),
            rng=np.random.default_rng(123),
        ),
    )
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.array([0, 1], dtype=np.int64)
    legal_ids = np.array([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6], dtype=np.uint32)
    legal_meta = np.array(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.zeros((2,), dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._fill_policy_outputs_ids(
        runtime,
        actor=actor,
        obs_step=obs_step,
        actor_step=actor_step,
        focal_rows=np.array([True, True], dtype=np.bool_),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(321),
        sample_actions=True,
    )

    assert model.forward_legal_flags == [False]
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.25)
    assert torch.allclose(actor.seat_hidden, torch.ones_like(actor.seat_hidden))


def test_fill_policy_outputs_ids_heuristic_actor_backend_can_skip_behavior_values() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_behavior_values_required = False
    runtime_any._teacher_policy = object()
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    runtime_any._heuristic_public_actions_from_ids = lambda **_kwargs: np.array([7, 8], dtype=np.int64)

    model = _HeuristicAdvanceOnlyRuntimeActorModel()
    actor = cast(
        Any,
        SimpleNamespace(
            model=model,
            compiled_model=None,
            seat_hidden=torch.zeros((2, 2, 3), dtype=torch.float32),
            focal_seat_by_env=np.array([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.array(["mirror", "mirror"], dtype=object),
            rng=np.random.default_rng(123),
        ),
    )
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.array([0, 1], dtype=np.int64)
    legal_ids = np.array([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6], dtype=np.uint32)
    legal_meta = np.array(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.full((2,), 9.0, dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._fill_policy_outputs_ids(
        runtime,
        actor=actor,
        obs_step=obs_step,
        actor_step=actor_step,
        focal_rows=np.array([True, True], dtype=np.bool_),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(321),
        sample_actions=True,
    )

    assert model.advance_calls == 1
    assert model.forward_calls == 0
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.0)
    assert torch.allclose(actor.seat_hidden, torch.ones_like(actor.seat_hidden))


def test_fill_policy_outputs_ids_heuristic_actor_backend_can_skip_hidden_tracking() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = False
    runtime_any._teacher_policy = object()
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    runtime_any._heuristic_public_actions_from_ids = lambda **_kwargs: np.array([7, 8], dtype=np.int64)

    model = _HeuristicAdvanceOnlyRuntimeActorModel()
    actor = cast(
        Any,
        SimpleNamespace(
            model=model,
            compiled_model=None,
            seat_hidden=torch.zeros((2, 2, 3), dtype=torch.float32),
            focal_seat_by_env=np.array([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.array(["mirror", "mirror"], dtype=object),
            rng=np.random.default_rng(123),
        ),
    )
    obs_step = np.zeros((2, 4), dtype=np.float32)
    actor_step = np.array([0, 1], dtype=np.int64)
    legal_ids = np.array([0, 7, 8, 1, 7, 8], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 6], dtype=np.uint32)
    legal_meta = np.array(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
    values_out = np.full((2,), 9.0, dtype=np.float32)
    actions_out = np.zeros((2,), dtype=np.int64)
    logp_out = np.zeros((2,), dtype=np.float32)

    QueueRuntime._fill_policy_outputs_ids(
        runtime,
        actor=actor,
        obs_step=obs_step,
        actor_step=actor_step,
        focal_rows=np.array([True, True], dtype=np.bool_),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(321),
        sample_actions=True,
    )

    assert model.advance_calls == 0
    assert model.forward_calls == 0
    assert np.array_equal(actions_out, np.array([7, 8], dtype=np.int64))
    assert np.allclose(logp_out, 0.0)
    assert np.allclose(values_out, 0.0)
    assert torch.allclose(actor.seat_hidden, torch.zeros_like(actor.seat_hidden))


def test_collect_all_heuristic_ids_fast_requires_all_heuristic_linux_frontier_conditions() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(choose_heuristic_public_actions_into=lambda *args, **kwargs: None)
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", HEURISTIC_PUBLIC_POLICY_ID], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_fast(runtime, actor) is True


def test_collect_all_heuristic_ids_fast_rejects_nonheuristic_opponent_assignments() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(choose_heuristic_public_actions_into=lambda *args, **kwargs: None)
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, "latest_policy_mirror"],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", ""], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_fast(runtime, actor) is False


def test_collect_all_heuristic_ids_native_rollout_requires_stateless_heuristic_actor() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())
    runtime_any._heuristic_native_rollout_enabled = True
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = False

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(
                    choose_heuristic_public_actions_into=lambda *args, **kwargs: None,
                    rollout_heuristic_public_into_i16_legal_ids=lambda *args, **kwargs: None,
                    reset_done_into_i16_legal_ids=lambda *args, **kwargs: None,
                )
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", HEURISTIC_PUBLIC_POLICY_ID], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_native_rollout(runtime, actor) is True


def test_collect_all_heuristic_ids_native_rollout_rejects_hidden_tracking() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any._teacher_policy = object()
    runtime_any._league_config = SimpleNamespace(sampling=SimpleNamespace(heuristic_public_mix_fraction=1.0))
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: bool(str(policy_id).strip())
    runtime_any._heuristic_native_rollout_enabled = True
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = True

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            env=SimpleNamespace(
                pool=SimpleNamespace(
                    choose_heuristic_public_actions_into=lambda *args, **kwargs: None,
                    rollout_heuristic_public_into_i16_legal_ids=lambda *args, **kwargs: None,
                    reset_done_into_i16_legal_ids=lambda *args, **kwargs: None,
                )
            ),
            opponent_policy_id_by_env=np.array(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
                dtype=object,
            ),
            fixed_opponent_policy_id_by_env=np.array(["", HEURISTIC_PUBLIC_POLICY_ID], dtype=object),
        ),
    )

    assert QueueRuntime._can_collect_all_heuristic_ids_native_rollout(runtime, actor) is False


def _teacher_test_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 41,
                "pass_action_id": 40,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 9},
                    {"name": "main_move", "base": 19, "count": 20},
                    {"name": "climax_play", "base": 39, "count": 1},
                    {"name": "pass", "base": 40, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def _teacher_mulligan_test_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 7,
                "pass_action_id": 6,
                "constants": [["MAX_HAND", 5], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "families": [
                    {"name": "mulligan_confirm", "base": 0, "count": 1},
                    {"name": "mulligan_select", "base": 1, "count": 5},
                    {"name": "pass", "base": 6, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_select_pending_unrolls_train_ordered_keeps_same_behavior_version() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=2,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=3,
        queue_capacity_unrolls=3,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._pending_unrolls = deque(
        [
            _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=1, behavior_policy_version=1),
            _make_runtime_unroll(actor_id=1, unroll_seq=1, behavior_policy_version=1),
        ]
    )

    selected = QueueRuntime._select_pending_unrolls(runtime)

    assert [(item.behavior_policy_version, item.unroll_seq, item.actor_id) for item in selected] == [
        (0, 0, 0),
        (0, 0, 1),
    ]


def test_select_pending_unrolls_reserves_diverse_lane_quota_in_async_mode() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=6,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=4,
        queue_capacity_unrolls=8,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._diverse_opponent_actor_count = 2
    runtime_any._diverse_opponent_batch_fraction = 0.5
    runtime_any._pending_unrolls = deque(
        [
            _make_runtime_unroll(actor_id=4, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=5, unroll_seq=1, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=2, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=1, unroll_seq=3, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=3, unroll_seq=4, behavior_policy_version=0),
        ]
    )

    selected = QueueRuntime._select_pending_unrolls(runtime)

    assert len(selected) == 4
    assert sum(1 for item in selected if item.actor_id in {0, 1}) == 2
    assert [item.actor_id for item in selected[:2]] == [0, 1]


def test_split_focal_actor_rows_forces_model_policy_on_diverse_model_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._teacher_policy = object()
    runtime_any._active_actor_heuristic_fraction = lambda: 1.0

    model_rows, heuristic_rows = QueueRuntime._split_focal_actor_rows(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=True)),
        focal_indices=np.asarray([0, 2, 4], dtype=np.int64),
        rng=np.random.default_rng(7),
    )

    npt.assert_array_equal(model_rows, np.asarray([0, 2, 4], dtype=np.int64))
    assert heuristic_rows.size == 0


def test_actor_id_force_model_policy_lane_uses_global_actor_id() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._diverse_model_actor_count = 2

    assert QueueRuntime._actor_id_force_model_policy_lane(runtime, 0)
    assert QueueRuntime._actor_id_force_model_policy_lane(runtime, 1)
    assert not QueueRuntime._actor_id_force_model_policy_lane(runtime, 2)


def test_split_focal_actor_rows_rejects_mixed_impala_without_hidden_tracking() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._teacher_policy = object()
    runtime_any._active_actor_heuristic_fraction = lambda: 0.5
    runtime_any._actor_behavior_values_required = False
    runtime_any._heuristic_actor_hidden_state_tracking = False

    with pytest.raises(RuntimeError, match="mixed heuristic/model actor rows require"):
        QueueRuntime._split_focal_actor_rows(
            runtime,
            actor=cast(Any, SimpleNamespace(force_model_policy_lane=False)),
            focal_indices=np.asarray([0, 2, 4, 6], dtype=np.int64),
            rng=np.random.default_rng(7),
        )


def test_policy_train_mask_for_actor_can_exclude_pure_heuristic_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._train_on_heuristic_actor_rows = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._active_actor_heuristic_fraction = lambda: 1.0

    mask = QueueRuntime._policy_train_mask_for_actor(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=False)),
        focal_rows=np.asarray([True, False, True, False], dtype=np.bool_),
    )

    npt.assert_array_equal(mask, np.asarray([False, False, False, False], dtype=np.bool_))


def test_policy_train_mask_for_actor_excludes_only_known_heuristic_rows() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._train_on_heuristic_actor_rows = False
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._active_actor_heuristic_fraction = lambda: 0.75
    counters = {
        "policy_train_model_rows": 0,
        "policy_train_heuristic_rows": 0,
        "policy_excluded_heuristic_rows": 0,
    }

    mask = QueueRuntime._policy_train_mask_for_actor(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=False)),
        focal_rows=np.asarray([True, True, False, True, True], dtype=np.bool_),
        model_focal_indices=np.asarray([1, 2, 4], dtype=np.int64),
        counters=counters,
    )

    npt.assert_array_equal(mask, np.asarray([False, True, False, False, True], dtype=np.bool_))
    assert counters["policy_train_model_rows"] == 2
    assert counters["policy_train_heuristic_rows"] == 0
    assert counters["policy_excluded_heuristic_rows"] == 2


def test_reset_done_rows_fallback_reinitializes_full_actor_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=2,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._device = torch.device("cpu")

    assign_calls: list[tuple[np.ndarray, bool]] = []

    def fake_assign_episode_roles(actor, done, *, initial: bool = False) -> None:
        assign_calls.append((np.asarray(done, dtype=np.bool_).copy(), initial))
        actor.focal_seat_by_env[:] = np.array([0, 1], dtype=np.int64)
        actor.opponent_policy_id_by_env[:] = np.asarray(["mirror", "policy_000007"], dtype=object)

    runtime_any._assign_episode_roles = fake_assign_episode_roles

    class _FakeEnv:
        def reset_done(self, done) -> None:
            raise RuntimeError("reset_done unsupported")

        def reset(self, *, seed: int):
            return {"seed": seed}

    class _FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 3), device=device)

    actor = cast(
        Any,
        SimpleNamespace(
            env=_FakeEnv(),
            model=_FakeModel(),
            rng=np.random.default_rng(7),
            seat_hidden=torch.ones((2, 3)),
            opponent_hidden=torch.full((2, 3), 2.0),
            focal_seat_by_env=np.array([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["stale_a", "stale_b"], dtype=object),
        ),
    )

    batch = cast(Any, QueueRuntime._reset_done_rows(runtime, actor, np.array([True, False], dtype=np.bool_)))

    assert isinstance(batch, dict)
    assert isinstance(batch["seed"], int)
    assert torch.count_nonzero(actor.seat_hidden) == 0
    assert torch.count_nonzero(actor.opponent_hidden) == 0
    assert len(assign_calls) == 1
    assert np.array_equal(assign_calls[0][0], np.array([True, True], dtype=np.bool_))
    assert assign_calls[0][1] is True
    assert actor.focal_seat_by_env.tolist() == [0, 1]
    assert actor.opponent_policy_id_by_env.tolist() == ["mirror", "policy_000007"]


def test_advance_hidden_only_prefers_hidden_only_model_path() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False

    class _FakeModel:
        def __init__(self) -> None:
            self.advance_calls = 0
            self.forward_calls = 0

        def advance_seat_hidden(self, obs, acting_seat, hidden_state):
            self.advance_calls += 1
            assert tuple(obs.shape) == (2, 3)
            assert tuple(acting_seat.shape) == (2,)
            return hidden_state + 5.0

        def forward_seat_aware(self, obs, acting_seat, hidden_state):
            self.forward_calls += 1
            raise AssertionError("forward_seat_aware should not run when advance_seat_hidden exists")

    hidden = torch.zeros((4, 2), dtype=torch.float32)
    model = _FakeModel()

    QueueRuntime._advance_hidden_only(
        runtime,
        model=model,
        hidden_state=hidden,
        row_indices=np.array([1, 3], dtype=np.int64),
        obs_step=np.zeros((4, 3), dtype=np.float32),
        actor_step=np.array([0, 1, 0, 1], dtype=np.int64),
    )

    assert model.advance_calls == 1
    assert model.forward_calls == 0
    npt.assert_array_equal(hidden.numpy(), np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 0.0], [5.0, 5.0]], dtype=np.float32))


def test_bootstrap_values_prefers_value_only_model_path() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._bootstrap_models = None
    runtime_any._actors = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.value_calls = 0
            self.forward_calls = 0

        def value_seat_aware(self, obs, acting_seat, hidden_state):
            self.value_calls += 1
            return torch.full((obs.shape[0],), 7.0, dtype=torch.float32, device=obs.device)

        def forward_seat_aware(self, obs, acting_seat, hidden_state):
            self.forward_calls += 1
            raise AssertionError("forward_seat_aware should not run when value_seat_aware exists")

    model = _FakeModel()
    runtime_any._actors[0] = cast(Any, SimpleNamespace(model=model))
    unroll = _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0)
    unroll = replace(
        unroll,
        bootstrap_obs=np.zeros((3, 2), dtype=np.float32),
        bootstrap_actor=np.array([0, 2, 1], dtype=np.int64),
        final_hidden_state=np.zeros((3, 4), dtype=np.float32),
    )

    values = QueueRuntime._bootstrap_values(runtime, unroll)

    assert model.value_calls == 1
    assert model.forward_calls == 0
    npt.assert_array_equal(values, np.array([7.0, 0.0, 7.0], dtype=np.float32))


def test_sample_packed_action_scores_falls_back_to_last_candidate_when_cdf_undershoots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_scores = torch.tensor([0.0, -0.1, -0.2], dtype=torch.float32)
    packed_ids = torch.tensor([4, 7, 9], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3], dtype=torch.long)
    sample_seeds = torch.tensor([123], dtype=torch.long)

    monkeypatch.setattr(
        model_module,
        "_uniform_from_seeds",
        lambda sample_seeds, *, dtype: torch.tensor([0.99999994], dtype=dtype, device=sample_seeds.device),
    )
    monkeypatch.setattr(
        model_module,
        "_packed_local_cdf",
        lambda probabilities, offsets: torch.tensor(
            [0.4, 0.8, 0.9999999], dtype=probabilities.dtype, device=probabilities.device
        ),
    )

    actions, logp = model_module._sample_packed_action_scores(
        packed_scores,
        packed_ids,
        packed_offsets,
        sample_seeds,
        pass_action_id=51,
    )

    assert actions.tolist() == [9]
    assert torch.isfinite(logp).all()


def test_structured_warmstart_source_mix_balances_sources_and_restores_actor_slots() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=8,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = None
    runtime_any._teacher_policy = object()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._device = torch.device("cpu")
    runtime_any._forced_fixed_opponent_policy_ids = ()

    def fake_assign_episode_roles(actor, done, *, initial: bool = False) -> None:
        fixed = actor.fixed_opponent_policy_id_by_env
        assigned = np.full(actor.focal_seat_by_env.shape, _MIRROR_OPPONENT_POLICY_ID, dtype=object)
        if fixed is not None:
            fixed_array = np.asarray(fixed, dtype=object)
            non_empty = np.asarray([bool(str(value).strip()) for value in fixed_array.tolist()], dtype=np.bool_)
            assigned[non_empty] = fixed_array[non_empty]
        actor.opponent_policy_id_by_env[:] = assigned

    runtime_any._assign_episode_roles = fake_assign_episode_roles

    class _FakeEnv:
        def reset_done(self, done):
            return {"reset_done": np.asarray(done, dtype=np.bool_).copy()}

    class _FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 3), device=device)

    actor = cast(
        Any,
        SimpleNamespace(
            env=_FakeEnv(),
            model=_FakeModel(),
            rng=np.random.default_rng(7),
            seat_hidden=torch.ones((8, 3)),
            opponent_hidden=torch.ones((8, 3)),
            focal_seat_by_env=np.zeros((8,), dtype=np.int64),
            opponent_policy_id_by_env=np.full((8,), _MIRROR_OPPONENT_POLICY_ID, dtype=object),
            fixed_opponent_policy_id_by_env=None,
            current_batch=None,
        ),
    )
    runtime_any._actors = [actor]

    with QueueRuntime.structured_warmstart_source_mix(runtime) as metrics:
        assert metrics["structured_warmstart_source_count"] == 3.0
        assert metrics["structured_warmstart_self_play_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b1_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b2_envs_per_actor"] == 2.0
        assert actor.fixed_opponent_policy_id_by_env is not None
        assert actor.opponent_policy_id_by_env.tolist().count(_NOLEAGUE_BASELINE_POLICY_ID) == 3
        assert actor.opponent_policy_id_by_env.tolist().count(HEURISTIC_PUBLIC_POLICY_ID) == 2
        assert actor.opponent_policy_id_by_env.tolist().count(_MIRROR_OPPONENT_POLICY_ID) == 3
        assert HEURISTIC_PUBLIC_POLICY_ID in runtime_any._opponent_heuristic_policies
        assert runtime_any._forced_fixed_opponent_policy_ids == (
            _NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )

    assert actor.fixed_opponent_policy_id_by_env is None
    assert actor.opponent_policy_id_by_env.tolist() == [_MIRROR_OPPONENT_POLICY_ID] * 8
    assert HEURISTIC_PUBLIC_POLICY_ID not in runtime_any._opponent_heuristic_policies
    assert runtime_any._forced_fixed_opponent_policy_ids == ()


def test_structured_warmstart_source_mix_process_collectors_pushes_fixed_sources() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=2,
        envs_per_actor=8,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=11,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = object()
    runtime_any._teacher_policy = object()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: torch.nn.Linear(1, 1)}
    runtime_any._forced_fixed_opponent_policy_ids = ()
    runtime_any._actors = []

    class _FakeControlQueue:
        def __init__(self) -> None:
            self.commands: list[dict[str, Any]] = []

        def put(self, payload: dict[str, Any]) -> None:
            self.commands.append(payload)

    control_queues = [_FakeControlQueue(), _FakeControlQueue()]
    runtime_any._collector_control_queues = control_queues

    with QueueRuntime.structured_warmstart_source_mix(runtime) as metrics:
        assert metrics["structured_warmstart_source_count"] == 3.0
        assert metrics["structured_warmstart_self_play_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b1_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b2_envs_per_actor"] == 2.0
        assert runtime_any._forced_fixed_opponent_policy_ids == (
            _NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )

    for control_queue in control_queues:
        assert len(control_queue.commands) == 2
        apply_payload, restore_payload = control_queue.commands
        assert apply_payload["kind"] == "set_fixed_opponents"
        assert apply_payload["restore_defaults"] is False
        assert apply_payload["activate_teacher_heuristic"] is True
        assert tuple(apply_payload["forced_policy_ids"]) == (
            _NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )
        assert apply_payload["fixed_opponent_policy_id_by_env"].count(_NOLEAGUE_BASELINE_POLICY_ID) == 3
        assert apply_payload["fixed_opponent_policy_id_by_env"].count(HEURISTIC_PUBLIC_POLICY_ID) == 2
        assert apply_payload["fixed_opponent_policy_id_by_env"].count("") == 3
        assert isinstance(apply_payload["noleague_baseline_state_dict"], dict)
        assert restore_payload == {"kind": "set_fixed_opponents", "restore_defaults": True}


def test_disable_mirror_policy_fusion_context_restores_previous_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._disable_mirror_policy_fusion = False

    with QueueRuntime.disable_mirror_policy_fusion(runtime):
        assert runtime_any._disable_mirror_policy_fusion is True

    assert runtime_any._disable_mirror_policy_fusion is False

    runtime_any._disable_mirror_policy_fusion = True
    with QueueRuntime.disable_mirror_policy_fusion(runtime):
        assert runtime_any._disable_mirror_policy_fusion is True
    assert runtime_any._disable_mirror_policy_fusion is True


def test_handle_collector_commands_tracks_update_and_refreshes_pool() -> None:
    refresh_calls: list[int] = []

    runtime = cast(
        Any,
        SimpleNamespace(
            _current_learner_update=0,
            _effective_learner_update=0,
            refresh_opponent_pool=lambda: refresh_calls.append(1),
            _teacher_policy=None,
            _opponent_heuristic_policies={},
            _opponent_models={},
            _opponent_model_locks={},
            _forced_fixed_opponent_policy_ids=(),
            _reset_actor_state_for_fixed_opponents=lambda actor: None,
        ),
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.loaded = 0
            self.evaluated = 0
            self.learner_scale = 0.0
            self.actor_scale = 0.0

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.loaded += len(state_dict)

        def eval(self) -> _FakeModel:
            self.evaluated += 1
            return self

        def set_public_heuristic_logit_bias_scale(
            self,
            value: float,
            *,
            actor_value: float | None = None,
        ) -> None:
            self.learner_scale = float(value)
            if actor_value is not None:
                self.actor_scale = float(actor_value)

    actor = cast(Any, SimpleNamespace(model=_FakeModel(), snapshot_version=0, fixed_opponent_policy_id_by_env=None))

    class _Queue:
        def __init__(self, commands: list[dict[str, Any]]) -> None:
            self._commands = list(commands)

        def get_nowait(self) -> dict[str, Any]:
            if not self._commands:
                raise queue.Empty
            return self._commands.pop(0)

    queue_obj = _Queue(
        [
            {"kind": "set_update", "update": 7, "refresh_opponent_pool": True},
            {
                "kind": "reload",
                "model_state_dict": {"w": torch.tensor([1.0])},
                "public_heuristic_logit_bias_scale": 0.75,
                "public_heuristic_actor_logit_bias_scale": 0.5,
                "update": 8,
                "effective_update": 5,
            },
            {"kind": "refresh_opponent_pool", "update": 9},
        ]
    )

    should_stop = _handle_collector_commands(
        runtime=runtime,
        actor=actor,
        control_queue=queue_obj,
        default_fixed_slots=None,
        default_forced_policy_ids=(),
        default_teacher_active=False,
        default_has_noleague_baseline=False,
    )

    assert should_stop is False
    assert actor.snapshot_version == 8
    assert runtime._current_learner_update == 9
    assert runtime._effective_learner_update == 5
    assert actor.model.loaded == 1
    assert actor.model.evaluated == 1
    assert actor.model.learner_scale == pytest.approx(0.75)
    assert actor.model.actor_scale == pytest.approx(0.5)
    assert len(refresh_calls) == 2


def test_runtime_snapshot_opponent_load_restores_model_guidance_payload(tmp_path: Path, monkeypatch) -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.learner_scale = -1.0
            self.actor_scale = -1.0
            self.loaded_state: dict[str, Any] | None = None
            self.evaluated = 0

        def to(self, _device: torch.device) -> _FakeModel:
            return self

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.loaded_state = dict(state_dict)

        def set_public_heuristic_logit_bias_scale(
            self,
            value: float,
            *,
            actor_value: float | None = None,
        ) -> None:
            self.learner_scale = float(value)
            if actor_value is not None:
                self.actor_scale = float(actor_value)

        def eval(self) -> _FakeModel:
            self.evaluated += 1
            return self

    built_models: list[_FakeModel] = []

    def fake_build_policy_value_model(**_kwargs: Any) -> _FakeModel:
        model = _FakeModel()
        built_models.append(model)
        return model

    monkeypatch.setattr(runtime_module, "build_policy_value_model", fake_build_policy_value_model)
    snapshot_path = tmp_path / "training" / "snapshots" / "policy_000001" / "weights.pt"
    snapshot_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model_state_dict": {"w": torch.tensor([1.0])},
            "public_heuristic_logit_bias_scale": 3.0,
            "public_heuristic_actor_logit_bias_scale": 1.0,
        },
        snapshot_path,
    )

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._run_dir = tmp_path
    runtime_any.stack = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace()))
    runtime_any.observation_dim = 4
    runtime_any.action_dim = 5
    runtime_any._observation_spec = None
    runtime_any._spec_bundle = None
    runtime_any._device = torch.device("cpu")

    loaded = QueueRuntime._load_snapshot_model(
        runtime,
        "training/snapshots/policy_000001/weights.pt",
    )

    assert loaded is built_models[0]
    assert loaded.loaded_state is not None
    assert loaded.learner_scale == pytest.approx(3.0)
    assert loaded.actor_scale == pytest.approx(1.0)
    assert loaded.evaluated == 1


def test_load_residual_opponent_model_wraps_frozen_base(tmp_path: Path) -> None:
    class DummyBase(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.learner_scale: float | None = None
            self.actor_scale: float | None = None

        def set_public_heuristic_logit_bias_scale(self, value: float, *, scoring_mode: str = "learner") -> None:
            if scoring_mode == "learner":
                self.learner_scale = float(value)
            elif scoring_mode == "actor":
                self.actor_scale = float(value)

        def initial_seat_hidden(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            return torch.zeros((batch_size, 1), device=device, dtype=dtype or torch.float32)

        def forward_seat_aware(self, obs, acting_seat, seat_hidden_state=None, *, scoring_mode="learner"):
            return (
                torch.zeros((obs.shape[0], 5), device=obs.device),
                torch.zeros((obs.shape[0],), device=obs.device),
                self.initial_seat_hidden(obs.shape[0], device=obs.device),
            )

    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"base")
    residual_path = tmp_path / "residual_state.pt"
    residual = FrozenStoredLogitResidual(obs_dim=4, action_dim=5, hidden_dim=8, alpha=0.1)
    torch.save(
        {
            "obs_dim": 4,
            "action_dim": 5,
            "hidden_dim": 8,
            "alpha": 0.1,
            "model_state_dict": residual.state_dict(),
        },
        residual_path,
    )
    base_model = DummyBase()
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._run_dir = tmp_path
    runtime_any._device = torch.device("cpu")
    runtime_any._load_snapshot_model_from_path = lambda path, display_path=None: base_model

    loaded = QueueRuntime._load_residual_opponent_model(
        runtime,
        SimpleNamespace(
            policy_id="b1_residual_test",
            base_snapshot_path="base.pt",
            residual_state_path="residual_state.pt",
            public_heuristic_bias_scale=1.0,
        ),
    )

    assert isinstance(loaded, LiveFrozenB1Residual)
    assert base_model.learner_scale == pytest.approx(1.0)
    assert base_model.actor_scale == pytest.approx(1.0)
    assert all(not parameter.requires_grad for parameter in loaded.base_model.parameters())


def test_trainable_live_residual_freezes_base_but_keeps_residual_gradients() -> None:
    class DummyBase(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def initial_seat_hidden(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            return torch.zeros((batch_size, 1), device=device, dtype=dtype or torch.float32)

        def forward_seat_aware(
            self,
            obs,
            acting_seat,
            seat_hidden_state=None,
            *,
            scoring_mode="learner",
            legal_actions=None,
        ):
            logits = torch.zeros((obs.shape[0], 5), device=obs.device) + self.weight
            return (
                logits,
                torch.zeros((obs.shape[0],), device=obs.device),
                self.initial_seat_hidden(obs.shape[0], device=obs.device),
            )

    base_model = DummyBase()
    residual = FrozenStoredLogitResidual(obs_dim=4, action_dim=5, hidden_dim=8, alpha=0.1)
    wrapper = TrainableLiveFrozenB1Residual(base_model=base_model, residual_probe=residual)
    obs = torch.ones((1, 4))
    hidden = wrapper.initial_seat_hidden(1)
    logits, _value, _next_hidden = wrapper.forward_seat_aware(obs, torch.tensor([0]), hidden)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2]))
    loss.backward()
    wrapper.set_public_heuristic_logit_bias_scale(1.0, actor_value=1.0)
    legal_logits, _value, _next_hidden = wrapper.forward_seat_aware(
        obs,
        torch.tensor([0]),
        hidden,
        legal_actions=object(),
    )

    assert all(not parameter.requires_grad for parameter in wrapper.base_model.parameters())
    assert base_model.weight.grad is None
    assert legal_logits.shape == (1, 5)
    assert wrapper.get_public_heuristic_logit_bias_scale(scoring_mode="learner") == pytest.approx(0.0)
    residual_grads = [parameter.grad for parameter in wrapper.residual_probe.parameters() if parameter.requires_grad]
    assert any(grad is not None and torch.any(grad != 0) for grad in residual_grads)


def test_configured_resident_opponent_policy_ids_include_residual_specs() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 0
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._active_heuristic_public_variant_mix_fraction = lambda: 0.0
    runtime_any._active_noleague_baseline_mix_fraction = lambda: 0.0
    runtime_any._residual_opponent_policy_specs = (SimpleNamespace(policy_id="b1_residual_test"),)

    assert QueueRuntime._configured_resident_opponent_policy_ids(runtime) == ("b1_residual_test",)


def test_diverse_opponent_actor_count_minus_one_marks_all_actor_ids_diverse() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._diverse_opponent_actor_count = -1
    runtime_any._diverse_model_actor_count = 0

    assert QueueRuntime._actor_id_is_diverse_lane(runtime, 0) is True
    assert QueueRuntime._actor_id_is_diverse_lane(runtime, 999) is True
    assert QueueRuntime._actor_id_force_model_policy_lane(runtime, 0) is False


def test_build_actor_state_uses_minus_one_diverse_lane_sentinel(monkeypatch) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = SimpleNamespace(base_seed=1, envs_per_actor=2)
    runtime_any._device = torch.device("cpu")
    runtime_any._compile_actor_inference = False
    runtime_any._shared_actor_model = None
    runtime_any._shared_compiled_actor_model = None
    runtime_any._current_learner_update = 0
    runtime_any._diverse_opponent_actor_count = -1
    runtime_any._diverse_model_actor_count = 0
    runtime_any._fixed_opponent_policy_slots = lambda: None
    runtime_any._assign_episode_roles = lambda *_args, **_kwargs: None

    class _FakeEnv:
        def reset(self, *, seed: int | None = None) -> Any:
            return SimpleNamespace(seed=seed)

    class _FakeModel:
        def to(self, _device: torch.device) -> _FakeModel:
            return self

        def eval(self) -> None:
            return None

        def initial_seat_hidden(self, env_count: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((env_count, 1), device=device)

    runtime_any._build_env = lambda **_kwargs: (_FakeEnv(), "i16_legal_ids")

    def fake_compile(model: Any, *, enabled: bool) -> None:
        return None

    monkeypatch.setattr(runtime_module, "_maybe_compile_runtime_actor_model", fake_compile)
    actor = QueueRuntime._build_actor_state(runtime, model=_FakeModel(), actor_id=37)

    assert actor.diverse_opponent_lane is True
    assert actor.force_model_policy_lane is False


def test_refresh_opponent_pool_excludes_fixed_b1_anchor(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
    )
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 0
    runtime_any._pfsp_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._pfsp_pool_size == 1
    assert runtime_any._opponent_models == {"policy_000007": "loaded::training/snapshots/policy_000007/weights.pt"}


def test_refresh_opponent_pool_keeps_reserved_b1_anchor_resident(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
    )
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 1
    runtime_any._pfsp_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
        "b1_noleague_baseline": "loaded::training/snapshots/b1_noleague_baseline/weights.pt",
    }


def test_refresh_opponent_pool_keeps_mixed_b1_anchor_resident(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
        sampling=SimpleNamespace(
            noleague_baseline_mix_fraction=0.15,
            noleague_baseline_mix_end_updates=-1,
        ),
    )
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 0
    runtime_any._pfsp_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
        "b1_noleague_baseline": "loaded::training/snapshots/b1_noleague_baseline/weights.pt",
    }


def test_refresh_opponent_pool_keeps_small_recent_reservoir_when_promotion_gate_enabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.add_champion("policy_000007")
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ("policy_000007",)
    assert runtime_any._opponent_recent_ids == ("policy_000008",)
    assert runtime_any._opponent_candidate_ids == ("policy_000007", "policy_000008")
    assert runtime_any._pfsp_pool_size == 2
    assert runtime_any._pfsp_recent_pool_size == 1
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
        "policy_000008": "loaded::training/snapshots/policy_000008/weights.pt",
    }


def test_refresh_opponent_pool_excludes_rejected_recent_when_promotion_gate_enabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_champion("policy_000007")
    registry.reject_snapshot("policy_000008")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ("policy_000007",)
    assert runtime_any._opponent_recent_ids == ()
    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
    }


def test_refresh_opponent_pool_uses_probationary_recent_pool_before_first_champion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ()
    assert runtime_any._opponent_recent_ids == ("policy_000007", "policy_000008")
    assert runtime_any._opponent_candidate_ids == ("policy_000007", "policy_000008")
    assert runtime_any._pfsp_pool_size == 2
    assert runtime_any._pfsp_recent_pool_size == 2
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
        "policy_000008": "loaded::training/snapshots/policy_000008/weights.pt",
    }


def test_refresh_opponent_pool_keeps_models_for_inflight_stale_assignments(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._actors = [SimpleNamespace(opponent_policy_id_by_env=np.asarray(["policy_000007"], dtype=object))]
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000008",)
    assert runtime_any._opponent_models == {
        "policy_000008": "loaded::training/snapshots/policy_000008/weights.pt",
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
    }


def test_maybe_publish_snapshot_tracks_effective_update_for_reused_weights() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=1,
        queue_capacity_unrolls=1,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._device = torch.device("cpu")
    runtime_any._collector_result_queue = None
    runtime_any._collector_control_queues = []
    runtime_any._collector_free_queues = []
    runtime_any._shared_actor_model = None
    runtime_any._bootstrap_models = None
    runtime_any._actors = [SimpleNamespace(model=torch.nn.Linear(2, 2), snapshot_version=0)]
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=200),
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        promotion_gate_enabled=True,
    )
    runtime_any._opponent_sampler = object()
    runtime_any._opponent_candidate_ids = ("policy_000007",)
    runtime_any._opponent_models = {"policy_000007": object()}
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0
    runtime_any._published_snapshot_update_by_fingerprint = {}
    runtime_any._last_published_snapshot_version = 0

    learner_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        learner_model.weight.fill_(1.0)
        learner_model.bias.fill_(0.5)

    QueueRuntime.maybe_publish_snapshot(runtime, learner_model=learner_model, learner_update_count=20, force=True)
    assert runtime_any._current_learner_update == 20
    assert runtime_any._effective_learner_update == 20
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False

    restored_model = torch.nn.Linear(2, 2)
    restored_model.load_state_dict(learner_model.state_dict())

    QueueRuntime.maybe_publish_snapshot(runtime, learner_model=restored_model, learner_update_count=220, force=True)
    assert runtime_any._current_learner_update == 220
    assert runtime_any._effective_learner_update == 20
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False


def test_refresh_opponent_pool_uses_effective_update_for_champion_age(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000120",
        update=120,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000120/weights.pt",
    )
    registry.add_champion("policy_000120")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        pool=SimpleNamespace(champion_max_age_updates=50),
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._current_learner_update = 220
    runtime_any._effective_learner_update = 20
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ("policy_000120",)
    assert runtime_any._opponent_candidate_ids == ("policy_000120",)


def test_refresh_opponent_pool_quarantines_timeout_heavy_champions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_champion("policy_000007")
    registry.add_champion("policy_000008")
    registry.save(registry_path)

    outcomes = OnlineOutcomeTracker(window_size=128)
    for _ in range(40):
        outcomes.update("policy_000007", "t")
        outcomes.update("policy_000008", "w")

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = outcomes
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007", "policy_000008")
    assert runtime_any._pfsp_pool_size == 2
    assert runtime_any._pfsp_quarantined_opponents == 1
    assert runtime_any._opponent_models == {
        "policy_000008": "loaded::training/snapshots/policy_000008/weights.pt",
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
    }


def test_refresh_opponent_pool_keeps_small_recent_reservoir_when_champions_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000190",
        update=190,
        weights_sha256="a" * 64,
        path="training/snapshots/policy_000190/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000191",
        update=191,
        weights_sha256="b" * 64,
        path="training/snapshots/policy_000191/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000192",
        update=192,
        weights_sha256="c" * 64,
        path="training/snapshots/policy_000192/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000193",
        update=193,
        weights_sha256="d" * 64,
        path="training/snapshots/policy_000193/weights.pt",
    )
    registry.add_champion("policy_000190")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ("policy_000190",)
    assert runtime_any._opponent_recent_ids == ("policy_000192", "policy_000193")
    assert runtime_any._opponent_candidate_ids == ("policy_000190", "policy_000192", "policy_000193")
    assert runtime_any._pfsp_pool_size == 3
    assert runtime_any._pfsp_recent_pool_size == 2
    assert runtime_any._opponent_models == {
        "policy_000190": "loaded::training/snapshots/policy_000190/weights.pt",
        "policy_000192": "loaded::training/snapshots/policy_000192/weights.pt",
        "policy_000193": "loaded::training/snapshots/policy_000193/weights.pt",
    }


def test_refresh_opponent_pool_can_exclude_seed_imports_after_pfsp_handoff(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="seed_source_policy_000450",
        update=450,
        weights_sha256="a" * 64,
        path="training/snapshots/seed_source_policy_000450/weights.pt",
        source_kind="seed_import",
    )
    registry.add_snapshot(
        policy_id="policy_000480",
        update=480,
        weights_sha256="b" * 64,
        path="training/snapshots/policy_000480/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000500",
        update=500,
        weights_sha256="c" * 64,
        path="training/snapshots/policy_000500/weights.pt",
    )
    registry.add_champion("seed_source_policy_000450")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._effective_learner_update = 530
    runtime_any._league_eval_warmup_gate_open = True
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        warmup=SimpleNamespace(first_updates=520, eval_gate_enabled=True),
        sampling=SimpleNamespace(
            exclude_seed_snapshots_from_pfsp=True,
            hard_negative_min_samples=16,
            hard_negative_max_win_rate=0.45,
        ),
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ()
    assert runtime_any._opponent_recent_ids == ("policy_000480", "policy_000500")
    assert runtime_any._opponent_candidate_ids == ("policy_000480", "policy_000500")
    assert "seed_source_policy_000450" not in runtime_any._opponent_models


def test_refresh_opponent_pool_keeps_seed_imports_before_pfsp_handoff(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="seed_source_policy_000450",
        update=450,
        weights_sha256="a" * 64,
        path="training/snapshots/seed_source_policy_000450/weights.pt",
        source_kind="seed_import",
    )
    registry.add_snapshot(
        policy_id="policy_000480",
        update=480,
        weights_sha256="b" * 64,
        path="training/snapshots/policy_000480/weights.pt",
    )
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._effective_learner_update = 500
    runtime_any._league_eval_warmup_gate_open = False
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        warmup=SimpleNamespace(first_updates=520, eval_gate_enabled=True),
        sampling=SimpleNamespace(
            exclude_seed_snapshots_from_pfsp=True,
            hard_negative_min_samples=16,
            hard_negative_max_win_rate=0.45,
        ),
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_recent_ids == ("policy_000480",)
    assert runtime_any._opponent_warmup_snapshot_ids == ("seed_source_policy_000450",)
    assert runtime_any._pfsp_warmup_snapshot_pool_size == 1
    assert "seed_source_policy_000450" in runtime_any._opponent_models


def test_refresh_opponent_pool_never_treats_seed_history_as_active_champion_or_recent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="seed_source_policy_000450",
        update=450,
        weights_sha256="a" * 64,
        path="training/snapshots/seed_source_policy_000450/weights.pt",
        source_kind="seed_import",
    )
    registry.add_snapshot(
        policy_id="policy_000480",
        update=480,
        weights_sha256="b" * 64,
        path="training/snapshots/policy_000480/weights.pt",
        source_kind="league_import",
    )
    registry.add_champion("seed_source_policy_000450")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._effective_learner_update = 530
    runtime_any._league_eval_warmup_gate_open = True
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        warmup=SimpleNamespace(first_updates=520, eval_gate_enabled=True),
        sampling=SimpleNamespace(
            exclude_seed_snapshots_from_pfsp=True,
            hard_negative_min_samples=16,
            hard_negative_max_win_rate=0.45,
        ),
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ()
    assert runtime_any._opponent_recent_ids == ("policy_000480",)
    assert runtime_any._opponent_warmup_snapshot_ids == ()
    assert runtime_any._opponent_candidate_ids == ("policy_000480",)


def test_refresh_opponent_pool_keeps_champions_out_of_recent_lane(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    for update in (780, 800, 820, 840, 860):
        registry.add_snapshot(
            policy_id=f"policy_{update:06d}",
            update=update,
            weights_sha256=f"{update:064x}"[-64:],
            path=f"training/snapshots/policy_{update:06d}/weights.pt",
        )
    for update in (800, 820, 840, 860):
        registry.add_champion(f"policy_{update:06d}")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == (
        "policy_000800",
        "policy_000820",
        "policy_000840",
        "policy_000860",
    )
    assert runtime_any._opponent_recent_ids == ("policy_000780",)
    assert runtime_any._pfsp_recent_pool_size == 1
    assert runtime_any._opponent_candidate_ids == (
        "policy_000800",
        "policy_000820",
        "policy_000840",
        "policy_000860",
        "policy_000780",
    )


def test_refresh_opponent_pool_demotes_stale_champions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000010",
        update=10,
        weights_sha256="a" * 64,
        path="training/snapshots/policy_000010/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000190",
        update=190,
        weights_sha256="b" * 64,
        path="training/snapshots/policy_000190/weights.pt",
    )
    registry.add_champion("policy_000010")
    registry.add_champion("policy_000190")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._current_learner_update = 220
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        pool=SimpleNamespace(champion_max_age_updates=40),
        sampling=SimpleNamespace(
            champion_mix_fraction=0.35,
            hard_negative_mix_fraction=0.2,
            hard_negative_min_samples=16,
            hard_negative_max_win_rate=0.45,
        ),
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    refreshed = SnapshotRegistry.load(registry_path)
    assert refreshed.champion_snapshots == ["policy_000190"]
    assert runtime_any._opponent_champion_ids == ("policy_000190",)
    assert runtime_any._opponent_recent_ids == ("policy_000010",)
    assert runtime_any._opponent_candidate_ids == ("policy_000190", "policy_000010")


def test_sample_opponent_policy_ids_can_force_hard_negative_bucket() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ("policy_hard", "policy_recent")
    runtime_any._opponent_hard_negative_ids = ("policy_hard",)
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ("policy_recent",)
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=1.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {"policy_hard": object(), "policy_recent": object()}
    runtime_any._pfsp_sampling_ready = lambda: True

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == ("policy_hard", "policy_hard", "policy_hard", "policy_hard")
    assert runtime_any._pfsp_last_hard_negative_envs == 4
    assert runtime_any._pfsp_last_recent_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 0


def test_opponent_sampling_weights_reassign_inactive_league_mass_to_recent() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._current_learner_update = 780
    runtime_any._effective_learner_update = 780
    runtime_any._opponent_candidate_ids = ("policy_recent",)
    runtime_any._opponent_recent_ids = ("policy_recent",)
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_heuristic_policies = {
        HEURISTIC_PUBLIC_POLICY_ID: object(),
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID: object(),
        HEURISTIC_PUBLIC_CONTROL_POLICY_ID: object(),
    }
    runtime_any._opponent_models = {"policy_recent": object()}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.25,
            heuristic_public_final_mix_fraction=0.25,
            heuristic_public_mix_end_updates=400,
            heuristic_public_variant_mix_fraction=0.25,
            heuristic_public_variant_final_mix_fraction=0.25,
            heuristic_public_variant_mix_end_updates=400,
            noleague_baseline_mix_fraction=0.25,
            noleague_baseline_mix_end_updates=400,
            champion_mix_fraction=0.25,
            hard_negative_mix_fraction=0.20,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._pfsp_sampling_ready = lambda: True

    metrics = QueueRuntime._opponent_sampling_group_weight_metrics(runtime)

    assert metrics["pfsp_sampling_weight_heuristic_public"] == pytest.approx(0.25)
    assert metrics["pfsp_sampling_weight_heuristic_public_variant"] == pytest.approx(0.25)
    assert metrics["pfsp_sampling_weight_recent"] == pytest.approx(0.5)
    assert metrics["pfsp_sampling_weight_champion"] == pytest.approx(0.0)
    assert metrics["pfsp_sampling_weight_hard_negative"] == pytest.approx(0.0)
    assert metrics["pfsp_sampling_weight_noleague_baseline"] == pytest.approx(0.0)


def test_sample_opponent_policy_ids_can_force_heuristic_public_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {"B2 HeuristicPublic": object()}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
    )
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 4


def test_sample_opponent_policy_ids_can_force_noleague_baseline_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            noleague_baseline_mix_fraction=1.0,
            noleague_baseline_mix_end_updates=-1,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        _NOLEAGUE_BASELINE_POLICY_ID,
        _NOLEAGUE_BASELINE_POLICY_ID,
        _NOLEAGUE_BASELINE_POLICY_ID,
        _NOLEAGUE_BASELINE_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_noleague_baseline_envs == 4


def test_sample_opponent_policy_ids_disables_noleague_mix_after_end_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            noleague_baseline_mix_fraction=1.0,
            noleague_baseline_mix_end_updates=1,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {_NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 1

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 0
    assert runtime_any._pfsp_last_mirror_envs == 4
    assert runtime_any._pfsp_last_noleague_baseline_envs == 0


def test_sample_opponent_policy_ids_can_force_warmup_snapshot_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._opponent_candidate_ids = ("seed_recent_a", "seed_recent_b")
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        warmup=SimpleNamespace(first_updates=200000),
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            noleague_baseline_mix_fraction=0.0,
            noleague_baseline_mix_end_updates=-1,
            warmup_snapshot_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {"seed_recent_a": object(), "seed_recent_b": object()}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert set(sampled) <= {"seed_recent_a", "seed_recent_b"}
    assert len(sampled) == 4
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_recent_envs == 0
    assert runtime_any._pfsp_last_warmup_snapshot_envs == 4


def test_sample_opponent_policy_ids_can_use_configured_mirror_lane_after_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._opponent_candidate_ids = ("recent_a",)
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ("recent_a",)
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            heuristic_public_variant_mix_fraction=0.0,
            noleague_baseline_mix_fraction=0.0,
            noleague_baseline_mix_end_updates=-1,
            warmup_snapshot_mix_fraction=0.0,
            mirror_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {"recent_a": object()}
    runtime_any._pfsp_sampling_ready = lambda: True
    runtime_any._league_reference_update = lambda: 100

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
    )
    assert runtime_any._pfsp_last_mirror_envs == 4
    assert runtime_any._pfsp_last_recent_envs == 0
    metrics = QueueRuntime._opponent_sampling_group_weight_metrics(runtime)
    assert metrics["pfsp_sampling_weight_mirror"] == pytest.approx(1.0)
    assert metrics["pfsp_sampling_weight_recent"] == pytest.approx(0.0)


def test_active_heuristic_public_mix_fraction_linearly_anneals_with_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
        )
    )
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_heuristic_public_mix_fraction(runtime) == pytest.approx(1.0)

    runtime_any._effective_learner_update = 3

    assert QueueRuntime._active_heuristic_public_mix_fraction(runtime) == pytest.approx(0.55)

    runtime_any._effective_learner_update = 5

    assert QueueRuntime._active_heuristic_public_mix_fraction(runtime) == pytest.approx(0.25)


def test_active_heuristic_public_variant_mix_fraction_linearly_anneals_with_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_variant_mix_fraction=0.4,
            heuristic_public_variant_mix_end_updates=4,
            heuristic_public_variant_final_mix_fraction=0.1,
        )
    )
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0
    runtime_any._league_reference_update = lambda: runtime_any._effective_learner_update

    assert QueueRuntime._active_heuristic_public_variant_mix_fraction(runtime) == pytest.approx(0.4)

    runtime_any._effective_learner_update = 2

    assert QueueRuntime._active_heuristic_public_variant_mix_fraction(runtime) == pytest.approx(0.25)

    runtime_any._effective_learner_update = 4

    assert QueueRuntime._active_heuristic_public_variant_mix_fraction(runtime) == pytest.approx(0.1)


def test_heuristic_opponent_policy_falls_back_to_teacher_for_b2() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    teacher_policy = object()
    runtime_any._teacher_policy = teacher_policy
    runtime_any._opponent_heuristic_policies = {
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID: object(),
    }

    assert QueueRuntime._heuristic_opponent_policy(runtime, HEURISTIC_PUBLIC_POLICY_ID) is teacher_policy
    assert (
        QueueRuntime._heuristic_opponent_policy(runtime, HEURISTIC_PUBLIC_AGGRO_POLICY_ID)
        is runtime_any._opponent_heuristic_policies[HEURISTIC_PUBLIC_AGGRO_POLICY_ID]
    )


def test_active_warmup_snapshot_mix_fraction_turns_off_after_league_warmup() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=5),
        sampling=SimpleNamespace(warmup_snapshot_mix_fraction=0.4),
    )
    runtime_any._opponent_candidate_ids = ("seed_a",)
    runtime_any._opponent_models = {"seed_a": object()}
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_warmup_snapshot_mix_fraction(runtime) == pytest.approx(0.4)

    runtime_any._effective_learner_update = 5

    assert QueueRuntime._active_warmup_snapshot_mix_fraction(runtime) == pytest.approx(0.0)


def test_eval_gated_warmup_keeps_snapshot_lane_and_blocks_pfsp_after_update_threshold() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=5, eval_gate_enabled=True),
        sampling=SimpleNamespace(warmup_snapshot_mix_fraction=0.4),
    )
    runtime_any._opponent_sampler = object()
    runtime_any._opponent_candidate_ids = ("seed_a",)
    runtime_any._opponent_models = {"seed_a": object()}
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 5
    runtime_any._league_eval_warmup_gate_open = False

    assert QueueRuntime._active_warmup_snapshot_mix_fraction(runtime) == pytest.approx(0.4)
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False

    runtime_any._league_eval_warmup_gate_open = True

    assert QueueRuntime._active_warmup_snapshot_mix_fraction(runtime) == pytest.approx(0.0)
    assert QueueRuntime._pfsp_sampling_ready(runtime) is True


def test_assign_episode_roles_uses_weighted_sampler_on_diverse_warmup_lane() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace()
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._active_warmup_snapshot_mix_fraction = lambda: 0.5
    runtime_any._opponent_candidate_ids = ("seed_recent_a", "seed_recent_b")
    runtime_any._sample_warmup_snapshot_policy_ids = lambda *, count, rng: ("unexpected",) * count
    runtime_any._sample_opponent_policy_ids = lambda *, count, rng: (_NOLEAGUE_BASELINE_POLICY_ID,) * count
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: False

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID], dtype=object
            ),
            fixed_opponent_policy_id_by_env=None,
            diverse_opponent_lane=True,
        ),
    )

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.asarray([True, True], dtype=np.bool_),
        initial=False,
    )

    assert actor.opponent_policy_id_by_env.tolist() == [
        _NOLEAGUE_BASELINE_POLICY_ID,
        _NOLEAGUE_BASELINE_POLICY_ID,
    ]


def test_assign_episode_roles_can_force_b1_baseline_focal_seat() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(noleague_baseline_force_focal_seat=1)
    )
    runtime_any._sample_opponent_policy_ids = lambda *, count, rng: (
        _NOLEAGUE_BASELINE_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
    )
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: False

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID], dtype=object
            ),
            fixed_opponent_policy_id_by_env=None,
            diverse_opponent_lane=True,
        ),
    )

    QueueRuntime._assign_episode_roles(
        runtime,
        actor,
        np.asarray([True, True], dtype=np.bool_),
        initial=False,
    )

    assert actor.opponent_policy_id_by_env.tolist() == [
        _NOLEAGUE_BASELINE_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
    ]
    assert actor.focal_seat_by_env.tolist() == [1, 0]


def test_sample_opponent_policy_ids_respects_heuristic_public_mix_anneal_end_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {"B2 HeuristicPublic": object()}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=1,
            heuristic_public_final_mix_fraction=0.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 1

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
        _MIRROR_OPPONENT_POLICY_ID,
    )
    assert runtime_any._pfsp_last_sampled_envs == 0
    assert runtime_any._pfsp_last_mirror_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_envs == 0


def test_sample_opponent_policy_ids_can_force_heuristic_public_variant_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_heuristic_public_variant_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            heuristic_public_variant_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
        pfsp_power=1.5,
        pfsp_epsilon_uniform=0.3,
    )
    runtime_any._opponent_heuristic_policies = {
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID: object(),
        HEURISTIC_PUBLIC_CONTROL_POLICY_ID: object(),
    }
    runtime_any._opponent_models = {}
    runtime_any._opponent_candidate_ids = ()
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=8,
        rng=np.random.default_rng(7),
    )

    assert set(sampled).issubset({HEURISTIC_PUBLIC_AGGRO_POLICY_ID, HEURISTIC_PUBLIC_CONTROL_POLICY_ID})
    assert runtime_any._pfsp_last_heuristic_public_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_variant_envs == 8


def test_active_actor_heuristic_fraction_linearly_anneals_with_update() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_start_updates = 0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(1.0)

    runtime_any._effective_learner_update = 3

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.55)

    runtime_any._effective_learner_update = 5

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.25)


def test_active_actor_heuristic_fraction_respects_delayed_anneal_start() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_start_updates = 4
    runtime_any._actor_heuristic_end_updates = 8
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(1.0)

    runtime_any._effective_learner_update = 6

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.625)

    runtime_any._effective_learner_update = 8

    assert QueueRuntime._active_actor_heuristic_fraction(runtime) == pytest.approx(0.25)


def test_split_focal_actor_rows_respects_actor_heuristic_anneal_endpoint() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._teacher_policy = object()
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.0
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 5

    model_rows, heuristic_rows = QueueRuntime._split_focal_actor_rows(
        runtime,
        actor=cast(Any, SimpleNamespace(force_model_policy_lane=False)),
        focal_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        rng=np.random.default_rng(7),
    )

    assert model_rows.tolist() == [0, 1, 2, 3]
    assert heuristic_rows.tolist() == []


def test_assign_episode_roles_prioritizes_fixed_anchor_lanes() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: True

    def fake_sample(*, count: int, rng) -> tuple[str, ...]:
        runtime_any._pfsp_last_sampled_envs = count
        runtime_any._pfsp_last_mirror_envs = 0
        runtime_any._pfsp_last_heuristic_public_envs = 0
        runtime_any._pfsp_last_noleague_baseline_envs = 0
        runtime_any._pfsp_last_champion_envs = 0
        runtime_any._pfsp_last_recent_envs = count
        runtime_any._pfsp_last_hard_negative_envs = 0
        return tuple(f"recent_{index}" for index in range(count))

    runtime_any._sample_opponent_policy_ids = fake_sample

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1, 0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["old0", "old1", "old2", "old3"], dtype=object),
            fixed_opponent_policy_id_by_env=np.asarray(
                ["B2 HeuristicPublic", "b1_noleague_baseline", "", ""],
                dtype=object,
            ),
        ),
    )

    QueueRuntime._assign_episode_roles(
        runtime, actor, np.asarray([True, True, True, True], dtype=np.bool_), initial=False
    )

    assert actor.opponent_policy_id_by_env.tolist() == [
        "B2 HeuristicPublic",
        "b1_noleague_baseline",
        "recent_0",
        "recent_1",
    ]
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_envs == 1
    assert runtime_any._pfsp_last_noleague_baseline_envs == 1
    assert runtime_any._pfsp_last_recent_envs == 2


def test_assign_episode_roles_cycles_fixed_diverse_opponent_policy_ids() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._diverse_opponent_policy_ids = (
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )
    runtime_any._opponent_heuristic_policies = {
        "B3 HeuristicPublicAggro": object(),
        "B4 HeuristicPublicControl": object(),
    }
    runtime_any._opponent_models = {}
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_heuristic_public_variant_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_last_warmup_snapshot_envs = 0

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            diverse_opponent_lane=True,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1, 0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["old0", "old1", "old2", "old3"], dtype=object),
            fixed_opponent_policy_id_by_env=None,
        ),
    )

    QueueRuntime._assign_episode_roles(
        runtime, actor, np.asarray([True, True, True, True], dtype=np.bool_), initial=True
    )

    assigned = actor.opponent_policy_id_by_env.tolist()
    assert sorted(assigned) == [
        "B3 HeuristicPublicAggro",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
        "B4 HeuristicPublicControl",
    ]
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_variant_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0


def test_overwrite_central_outputs_with_opponents_only_touches_non_mirror_rows() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._opponent_model_locks = {"policy_000007": threading.Lock()}
    runtime_any._opponent_heuristic_policies = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
            actor_np = actor_tensor.detach().cpu().numpy()
            self.calls.append((obs_np.copy(), actor_np.copy()))
            logits = torch.full((obs_tensor.shape[0], 5), -7.0, dtype=torch.float32)
            values = torch.full((obs_tensor.shape[0],), 42.0, dtype=torch.float32)
            return logits, values, hidden_tensor + 1.0

    model = _FakeModel()
    runtime_any._opponent_models = {"policy_000007": model}

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            focal_seat_by_env=np.asarray([0, 0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, "policy_000007", _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            opponent_hidden=torch.zeros((3, 4)),
            rng=np.random.default_rng(7),
        ),
    )
    batch = cast(
        Any,
        SimpleNamespace(
            obs=np.zeros((3, 8), dtype=np.float32),
            actor=np.asarray([1, 1, 1], dtype=np.int64),
            ids_offsets=(
                np.asarray([0, 1, 2], dtype=np.uint32),
                np.asarray([0, 1, 2, 3], dtype=np.uint32),
            ),
            mask=None,
        ),
    )
    logits = np.zeros((3, 5), dtype=np.float32)
    values = np.zeros((3,), dtype=np.float32)

    QueueRuntime._overwrite_central_outputs_with_opponents(
        runtime,
        actor=actor,
        batch=batch,
        obs_step=np.asarray(batch.obs, dtype=np.float32),
        actor_step=np.asarray(batch.actor, dtype=np.int64),
        logits_out=logits,
        values_out=values,
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.zeros((1, 8), dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1], dtype=np.int64))
    assert values.tolist() == [0.0, 42.0, 0.0]
    assert np.all(logits[0] == 0.0)
    assert np.all(logits[1] == -7.0)
    assert np.all(logits[2] == 0.0)
    assert actor.opponent_hidden[1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert actor.opponent_hidden[0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_overwrite_central_outputs_with_batched_opponents_groups_rows_across_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._opponent_model_locks = {"policy_000007": threading.Lock()}

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
            actor_np = actor_tensor.detach().cpu().numpy()
            self.calls.append((obs_np.copy(), actor_np.copy()))
            logits = torch.arange(obs_tensor.shape[0] * 5, dtype=torch.float32).reshape(obs_tensor.shape[0], 5)
            values = torch.arange(obs_tensor.shape[0], dtype=torch.float32) + 10.0
            return logits, values, hidden_tensor + 2.0

    model = _FakeModel()
    runtime_any._opponent_models = {"policy_000007": model}

    actor_a = cast(
        Any,
        SimpleNamespace(
            focal_seat_by_env=np.asarray([0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["policy_000007", _MIRROR_OPPONENT_POLICY_ID], dtype=object),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    actor_b = cast(
        Any,
        SimpleNamespace(
            focal_seat_by_env=np.asarray([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray([_MIRROR_OPPONENT_POLICY_ID, "policy_000007"], dtype=object),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    logits_a = np.zeros((2, 5), dtype=np.float32)
    logits_b = np.zeros((2, 5), dtype=np.float32)
    values_a = np.zeros((2,), dtype=np.float32)
    values_b = np.zeros((2,), dtype=np.float32)
    obs_a = np.asarray([[1.0, 0.0], [9.0, 9.0]], dtype=np.float32)
    obs_b = np.asarray([[8.0, 8.0], [2.0, 0.0]], dtype=np.float32)
    actor_step_a = np.asarray([1, 0], dtype=np.int64)
    actor_step_b = np.asarray([1, 0], dtype=np.int64)

    QueueRuntime._overwrite_central_outputs_with_batched_opponents(
        runtime,
        actors=[actor_a, actor_b],
        batches=[cast(Any, SimpleNamespace()), cast(Any, SimpleNamespace())],
        obs_steps=[obs_a, obs_b],
        actor_steps=[actor_step_a, actor_step_b],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1, 0], dtype=np.int64))
    assert np.all(logits_a[0] == np.asarray([0, 1, 2, 3, 4], dtype=np.float32))
    assert np.all(logits_b[1] == np.asarray([5, 6, 7, 8, 9], dtype=np.float32))
    assert values_a.tolist() == [10.0, 0.0]
    assert values_b.tolist() == [0.0, 11.0]
    assert actor_a.opponent_hidden[0].tolist() == [2.0, 2.0, 2.0]
    assert actor_b.opponent_hidden[1].tolist() == [2.0, 2.0, 2.0]


def test_overwrite_central_outputs_with_batched_opponents_batches_heuristic_public_rows_across_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._profile_timers = False
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _FakeHeuristicPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []

        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            obs_array = np.asarray(obs_rows, dtype=np.int32)
            ids_array = np.asarray(legal_ids, dtype=np.uint32)
            offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
            meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
            self.calls.append(
                (
                    obs_array.copy(),
                    ids_array.copy(),
                    offsets_array.copy(),
                    None if meta_array is None else meta_array.copy(),
                )
            )
            return np.asarray(
                [int(ids_array[int(offsets_array[row_index])]) for row_index in range(offsets_array.shape[0] - 1)],
                dtype=np.int64,
            )

    heuristic_policy = _FakeHeuristicPolicy()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_POLICY_ID: heuristic_policy}

    shared_model = _AdvanceOnlyModel()
    actor_a = cast(
        Any,
        SimpleNamespace(
            model=shared_model,
            compiled_model=None,
            focal_seat_by_env=np.asarray([0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID], dtype=object
            ),
            seat_hidden=torch.zeros((2, 3)),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    actor_b = cast(
        Any,
        SimpleNamespace(
            model=shared_model,
            compiled_model=None,
            focal_seat_by_env=np.asarray([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID], dtype=object
            ),
            seat_hidden=torch.zeros((2, 3)),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    batch_a = cast(
        Any,
        SimpleNamespace(
            ids_offsets=(
                np.asarray([10, 11, 12], dtype=np.uint32),
                np.asarray([0, 2, 3], dtype=np.uint32),
            ),
            legal_action_meta=np.asarray(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [2, 2, 2, 0],
                ],
                dtype=np.uint16,
            ),
            mask=None,
        ),
    )
    batch_b = cast(
        Any,
        SimpleNamespace(
            ids_offsets=(
                np.asarray([20, 21, 22], dtype=np.uint32),
                np.asarray([0, 1, 3], dtype=np.uint32),
            ),
            legal_action_meta=np.asarray(
                [
                    [3, 0, 0, 0],
                    [4, 1, 0, 0],
                    [5, 2, 0, 0],
                ],
                dtype=np.uint16,
            ),
            mask=None,
        ),
    )
    obs_a = np.asarray([[1, 0, 0], [9, 9, 9]], dtype=np.float32)
    obs_b = np.asarray([[8, 8, 8], [2, 0, 0]], dtype=np.float32)
    actor_step_a = np.asarray([1, 0], dtype=np.int64)
    actor_step_b = np.asarray([1, 0], dtype=np.int64)
    logits_a = np.full((2, 32), -5.0, dtype=np.float32)
    logits_b = np.full((2, 32), -5.0, dtype=np.float32)
    values_a = np.ones((2,), dtype=np.float32)
    values_b = np.ones((2,), dtype=np.float32)

    QueueRuntime._overwrite_central_outputs_with_batched_opponents(
        runtime,
        actors=[actor_a, actor_b],
        batches=[batch_a, batch_b],
        obs_steps=[obs_a, obs_b],
        actor_steps=[actor_step_a, actor_step_b],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(heuristic_policy.calls) == 1
    call_obs, call_ids, call_offsets, call_meta = heuristic_policy.calls[0]
    assert np.array_equal(call_obs, np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.int32))
    assert np.array_equal(call_ids, np.asarray([10, 11, 21, 22], dtype=np.uint32))
    assert np.array_equal(call_offsets, np.asarray([0, 2, 4], dtype=np.uint32))
    assert call_meta is not None
    assert np.array_equal(
        call_meta,
        np.asarray(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [4, 1, 0, 0],
                [5, 2, 0, 0],
            ],
            dtype=np.uint16,
        ),
    )
    assert actor_a.seat_hidden[0].tolist() == [1.0, 1.0, 1.0]
    assert actor_b.seat_hidden[1].tolist() == [1.0, 1.0, 1.0]
    assert values_a.tolist() == [0.0, 1.0]
    assert values_b.tolist() == [1.0, 0.0]
    assert logits_a[0, 10] == pytest.approx(0.0)
    assert logits_a[0, 11] < 0.0
    assert logits_b[1, 21] == pytest.approx(0.0)
    assert logits_b[1, 22] < 0.0


def test_apply_opponent_rows_ids_uses_simulator_native_backend_for_heuristic_public() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _FailHeuristicPolicy:
        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            raise AssertionError("python heuristic batch path should not be used when simulator_native is enabled")

    class _FakePool:
        def __init__(self) -> None:
            self.calls: list[np.ndarray] = []

        def choose_heuristic_public_actions_into(self, env_indices: np.ndarray, actions_out: np.ndarray) -> None:
            indices = np.asarray(env_indices, dtype=np.uint32)
            self.calls.append(indices.copy())
            actions_out[...] = np.asarray([11, 20], dtype=np.uint16)

    fake_pool = _FakePool()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_POLICY_ID: _FailHeuristicPolicy()}

    actor = cast(
        Any,
        SimpleNamespace(
            model=_AdvanceOnlyModel(),
            compiled_model=None,
            env=SimpleNamespace(pool=fake_pool),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            seat_hidden=torch.zeros((3, 2)),
            opponent_hidden=torch.zeros((3, 2)),
        ),
    )
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
    actor_step = np.asarray([1, 1, 0], dtype=np.int64)
    legal_ids = np.asarray([10, 11, 12, 20, 21], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 5, 5], dtype=np.uint32)
    legal_action_meta = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    values_out = np.ones((3,), dtype=np.float32)
    actions_out = np.full((3,), 99, dtype=np.int64)
    logp_out = np.full((3,), -1.0, dtype=np.float32)

    QueueRuntime._apply_opponent_rows_ids(
        runtime,
        actor=actor,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(7),
        sample_actions=True,
    )

    assert len(fake_pool.calls) == 1
    assert np.array_equal(fake_pool.calls[0], np.asarray([0, 1], dtype=np.uint32))
    assert actor.seat_hidden[0].tolist() == [1.0, 1.0]
    assert actor.seat_hidden[1].tolist() == [1.0, 1.0]
    assert values_out.tolist() == [0.0, 0.0, 1.0]
    assert actions_out.tolist() == [11, 20, 99]
    assert logp_out.tolist() == [0.0, 0.0, -1.0]


def test_apply_opponent_rows_ids_falls_back_when_simulator_native_pool_hook_is_unavailable() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}

    class _AdvanceOnlyModel:
        def advance_seat_hidden(self, obs_tensor, actor_tensor, hidden_tensor):
            return hidden_tensor + 1.0

    class _RecordingHeuristicPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []

        def choose_actions_from_meta_batch(self, obs_rows, legal_ids, legal_offsets, legal_action_meta):
            obs_array = np.asarray(obs_rows, dtype=np.int32)
            ids_array = np.asarray(legal_ids, dtype=np.uint32)
            offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
            meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
            self.calls.append(
                (
                    obs_array.copy(),
                    ids_array.copy(),
                    offsets_array.copy(),
                    None if meta_array is None else meta_array.copy(),
                )
            )
            return np.asarray(
                [int(ids_array[int(offsets_array[row_index])]) for row_index in range(offsets_array.shape[0] - 1)],
                dtype=np.int64,
            )

    heuristic_policy = _RecordingHeuristicPolicy()
    runtime_any._opponent_heuristic_policies = {HEURISTIC_PUBLIC_POLICY_ID: heuristic_policy}

    actor = cast(
        Any,
        SimpleNamespace(
            model=_AdvanceOnlyModel(),
            compiled_model=None,
            env=SimpleNamespace(pool=SimpleNamespace()),
            opponent_policy_id_by_env=np.asarray(
                [HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID, _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            seat_hidden=torch.zeros((3, 2)),
            opponent_hidden=torch.zeros((3, 2)),
        ),
    )
    row_indices = np.asarray([0, 1], dtype=np.int64)
    obs_step = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
    actor_step = np.asarray([1, 1, 0], dtype=np.int64)
    legal_ids = np.asarray([10, 11, 12, 20, 21], dtype=np.uint32)
    legal_offsets = np.asarray([0, 3, 5, 5], dtype=np.uint32)
    legal_action_meta = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [4, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    values_out = np.ones((3,), dtype=np.float32)
    actions_out = np.full((3,), 99, dtype=np.int64)
    logp_out = np.full((3,), -1.0, dtype=np.float32)

    QueueRuntime._apply_opponent_rows_ids(
        runtime,
        actor=actor,
        row_indices=row_indices,
        obs_step=obs_step,
        actor_step=actor_step,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        logits_out=None,
        values_out=values_out,
        actions_out=actions_out,
        logp_out=logp_out,
        rng=np.random.default_rng(7),
        sample_actions=True,
    )

    assert len(heuristic_policy.calls) == 1
    call_obs, call_ids, call_offsets, call_meta = heuristic_policy.calls[0]
    assert np.array_equal(call_obs, np.asarray([[1, 0], [2, 0]], dtype=np.int32))
    assert np.array_equal(call_ids, np.asarray([10, 11, 12, 20, 21], dtype=np.uint32))
    assert np.array_equal(call_offsets, np.asarray([0, 3, 5], dtype=np.uint32))
    assert call_meta is not None
    assert np.array_equal(call_meta, legal_action_meta)
    assert actor.seat_hidden[0].tolist() == [1.0, 1.0]
    assert actor.seat_hidden[1].tolist() == [1.0, 1.0]
    assert values_out.tolist() == [0.0, 0.0, 1.0]
    assert actions_out.tolist() == [10, 20, 99]
    assert logp_out.tolist() == [0.0, 0.0, -1.0]


def test_build_runtime_config_minimal_batch_uses_one_unroll_per_actor() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            system=SimpleNamespace(
                actor_process_count=12,
                envs_per_actor=8,
                actor_queue_capacity_unrolls=256,
            ),
            training=SimpleNamespace(
                batch_unrolls_per_update=128,
                actor_reload_interval_updates=1000,
            ),
        )
    )

    small = build_runtime_config(
        stack=stack,
        num_envs=1,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert small.actor_count == 1
    assert small.envs_per_actor == 1
    assert small.batch_unrolls_per_update == 1
    assert small.queue_capacity_unrolls == 1

    full = build_runtime_config(
        stack=stack,
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert full.actor_count == 12
    assert full.envs_per_actor == 8
    assert full.batch_unrolls_per_update == 12
    assert full.queue_capacity_unrolls == 12

    default = build_runtime_config(
        stack=stack,
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
    )
    assert default.batch_unrolls_per_update == 128
    assert default.queue_capacity_unrolls == 256


def test_concatenate_legal_actions_keeps_packed_ids_fast_path() -> None:
    packed = LegalActionBatch.from_packed(
        np.array([0, 2, 1, 2], dtype=np.uint32),
        np.array([0, 2, 4], dtype=np.uint32),
        action_space=64,
    )
    unroll_a = replace(_make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)
    unroll_b = replace(_make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)

    combined = _concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.action_space == 64
    assert combined.offsets.tolist() == [0, 2, 4]
    assert combined.ids.tolist() == [0, 2, 0, 2]


def test_concatenate_legal_actions_reorders_packed_rows_to_match_time_major_batch_layout() -> None:
    packed_a = LegalActionBatch.from_packed(
        np.array([10, 11, 20, 21], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    packed_b = LegalActionBatch.from_packed(
        np.array([30, 31, 40, 41], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    unroll_a = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_a,
    )
    unroll_b = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_b,
    )

    combined = _concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.action_space == 64
    assert combined.offsets.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert combined.ids.tolist() == [10, 11, 30, 31, 20, 21, 40, 41]


def test_legal_action_batch_uses_metadata_to_expand_packed_payloads() -> None:
    packed = LegalActionBatch.from_packed(
        np.array([1, 3], dtype=np.uint32),
        np.array([0, 1, 2], dtype=np.uint32),
        action_space=5,
    )

    mask = packed.to_mask(expected_shape=(1, 2))

    assert packed.action_space == 5
    npt.assert_array_equal(
        mask,
        np.array([[[False, True, False, False, False], [False, False, False, True, False]]], dtype=np.bool_),
    )


def test_gae_advantages_matches_manual_discounted_deltas() -> None:
    rewards = np.asarray([[1.0], [0.5]], dtype=np.float32)
    values = np.asarray([[0.2], [0.3]], dtype=np.float32)
    discounts = np.asarray([[1.0], [0.0]], dtype=np.float32)
    bootstrap = np.asarray([0.4], dtype=np.float32)

    advantages = _gae_advantages(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        discounts=discounts,
        gae_lambda=0.95,
    )

    expected_last = 0.5 - 0.3
    expected_first = (1.0 + 0.3 - 0.2) + (0.95 * expected_last)
    assert advantages[:, 0].tolist() == pytest.approx([expected_first, expected_last])


def test_build_learner_batch_does_not_double_apply_truncation_reward() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 2), dtype=np.bool_)),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=-0.25,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert batch["discounts"][:, 0].tolist() == pytest.approx([0.99, 0.0])


def test_build_learner_batch_can_penalize_pass_with_nonpass_available() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 3
    runtime_any.config = SimpleNamespace(pass_action_id=0)
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((3, 1, 1), dtype=np.float32),
        actions=np.array([[0], [0], [1]], dtype=np.int64),
        rewards=np.zeros((3, 1), dtype=np.float32),
        terminated=np.zeros((3, 1), dtype=np.bool_),
        truncated=np.zeros((3, 1), dtype=np.bool_),
        to_play_seat=np.zeros((3, 1), dtype=np.int64),
        behavior_logp=np.zeros((3, 1), dtype=np.float32),
        values=np.zeros((3, 1), dtype=np.float32),
        policy_train_mask=np.array([[True], [False], [True]], dtype=np.bool_),
        legal_actions=LegalActionBatch.from_mask(
            np.array(
                [
                    [[True, True, False]],
                    [[True, True, False]],
                    [[True, True, False]],
                ],
                dtype=np.bool_,
            )
        ),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
        pass_with_nonpass_penalty=0.05,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([-0.05, 0.0, 0.0])


def test_build_learner_batch_preserves_teacher_labels() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 3
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 3), dtype=np.bool_)),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        teacher_family=np.array([[1], [2]], dtype=np.int32),
        teacher_slot=np.array([[0], [-1]], dtype=np.int32),
        teacher_move_source=np.array([[-1], [2]], dtype=np.int32),
        teacher_attack_type=np.array([[-1], [1]], dtype=np.int32),
        teacher_action=np.array([[7], [9]], dtype=np.int32),
        teacher_valid=np.array([[True], [False]], dtype=np.bool_),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert np.array_equal(batch["teacher_family"], unroll.teacher_family)
    assert np.array_equal(batch["teacher_slot"], unroll.teacher_slot)
    assert np.array_equal(batch["teacher_move_source"], unroll.teacher_move_source)
    assert np.array_equal(batch["teacher_attack_type"], unroll.teacher_attack_type)
    assert np.array_equal(batch["teacher_action"], unroll.teacher_action)
    assert np.array_equal(batch["teacher_valid"], unroll.teacher_valid)


def test_build_learner_batch_preserves_bootstrap_inputs_for_learner_values() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 3
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 3), dtype=np.bool_)),
        bootstrap_obs=np.array([[3.0]], dtype=np.float32),
        bootstrap_actor=np.array([1], dtype=np.int64),
        final_hidden_state=np.array([[[1.0, 2.0]]], dtype=np.float32),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert np.array_equal(batch["bootstrap_obs"], unroll.bootstrap_obs)
    assert np.array_equal(batch["bootstrap_actor"], unroll.bootstrap_actor)
    assert np.array_equal(batch["final_hidden_state"], unroll.final_hidden_state)


def test_build_ppo_batch_does_not_double_apply_truncation_reward() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    runtime_any._bootstrap_values = lambda unroll: np.zeros((1,), dtype=np.float32)
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 2), dtype=np.bool_)),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_ppo_batch(
        runtime,
        [unroll],
        gamma=0.99,
        gae_lambda=0.95,
        truncation_reward=-0.25,
        truncation_bootstrap_value=False,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert batch["discounts"][:, 0].tolist() == pytest.approx([0.99, 0.0])


def test_build_ppo_batch_uses_stored_behavior_bootstrap_values() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    runtime_any._bootstrap_values = lambda unroll: np.array([9.0], dtype=np.float32)
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=3),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 2), dtype=np.bool_)),
        bootstrap_value=np.array([0.25], dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_ppo_batch(
        runtime,
        [unroll],
        gamma=1.0,
        gae_lambda=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
    )

    assert batch["advantages"][:, 0].tolist() == pytest.approx([0.25])
    assert batch["returns"][:, 0].tolist() == pytest.approx([0.25])


def test_runtime_metrics_report_window_and_cumulative_env_step_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._runtime_start = 100.0
    runtime_any._runtime_last_metrics_time = 108.0
    runtime_any._runtime_cumulative_env_steps = 128
    runtime_any._last_published_snapshot_version = 5
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 3
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
        )
    )
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._pfsp_pool_size = 3
    runtime_any._pfsp_quarantined_opponents = 1
    runtime_any._pfsp_champion_pool_size = 1
    runtime_any._pfsp_recent_pool_size = 1
    runtime_any._pfsp_hard_negative_pool_size = 1
    runtime_any._pfsp_last_sampled_envs = 2
    runtime_any._pfsp_last_mirror_envs = 6
    runtime_any._pfsp_last_heuristic_public_envs = 2
    runtime_any._pfsp_last_noleague_baseline_envs = 1
    runtime_any._pfsp_last_champion_envs = 1
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 1
    runtime_any._pfsp_epoch = 3

    monkeypatch.setattr("weiss_rl.runtime.time.time", lambda: 110.0)
    metrics = QueueRuntime._runtime_metrics(
        runtime,
        [
            _make_runtime_unroll(
                actor_id=0,
                unroll_seq=0,
                behavior_policy_version=4,
                counters={
                    "engine_fault_done_rows": 2,
                    "no_progress_timeout_rows": 1,
                    "pass_actions": 3,
                    "main_move_actions": 4,
                    "max_consecutive_main_moves": 2,
                },
            ),
            replace(
                _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=5),
                obs=np.zeros((2, 3, 1), dtype=np.float32),
            ),
        ],
        occupancy_samples=[0.25, 0.75],
    )

    assert metrics["batch_env_steps"] == pytest.approx(7.0)
    assert metrics["actor_env_steps_per_sec"] == pytest.approx(3.5)
    assert metrics["actor_env_steps_per_sec_cumulative"] == pytest.approx(13.5)
    assert metrics["policy_version_lag_p50"] == pytest.approx(0.5)
    assert metrics["league_effective_update"] == pytest.approx(3.0)
    assert metrics["league_raw_effective_update"] == pytest.approx(3.0)
    assert metrics["league_update_lag"] == pytest.approx(2.0)
    assert metrics["actor_heuristic_fraction_active"] == pytest.approx(0.55)
    assert metrics["heuristic_public_mix_fraction_active"] == pytest.approx(0.55)
    assert metrics["noleague_baseline_reward_scale_active"] == pytest.approx(1.0)
    assert metrics["noleague_baseline_force_focal_seat_active"] == pytest.approx(-1.0)
    assert metrics["pfsp_quarantined_opponents"] == pytest.approx(1.0)
    assert metrics["pfsp_sampling_ready"] == pytest.approx(0.0)
    assert metrics["pfsp_candidate_model_count"] == pytest.approx(0.0)
    assert metrics["pfsp_sampling_weight_mirror"] == pytest.approx(1.0)
    assert metrics["pfsp_champion_pool_size"] == pytest.approx(1.0)
    assert metrics["pfsp_heuristic_public_envs"] == pytest.approx(2.0)
    assert metrics["pfsp_noleague_baseline_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_hard_negative_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_epoch"] == pytest.approx(3.0)
    assert metrics["queue_occupancy_p50"] == pytest.approx(0.5)
    assert metrics["collector_engine_fault_done_rows"] == pytest.approx(2.0)
    assert metrics["collector_no_progress_timeout_rows"] == pytest.approx(1.0)
    assert metrics["collector_pass_actions"] == pytest.approx(3.0)
    assert metrics["collector_main_move_actions"] == pytest.approx(4.0)
    assert metrics["collector_max_consecutive_main_moves"] == pytest.approx(2.0)
    assert runtime_any._runtime_last_metrics_time == pytest.approx(110.0)
    assert runtime_any._runtime_cumulative_env_steps == 135


def test_apply_opponent_reward_scale_targets_b1_baseline_rows() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(noleague_baseline_reward_scale=3.0)
    )
    actor = SimpleNamespace(
        opponent_policy_id_by_env=np.asarray(
            [_NOLEAGUE_BASELINE_POLICY_ID, "mirror", _NOLEAGUE_BASELINE_POLICY_ID],
            dtype=object,
        )
    )
    rewards = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)

    shaped = QueueRuntime._apply_opponent_reward_scale(runtime, cast(Any, actor), rewards)

    assert shaped.tolist() == pytest.approx([3.0, -2.0, 1.5])
    assert rewards.tolist() == pytest.approx([1.0, -2.0, 0.5])


def test_runtime_metrics_fall_back_to_current_update_for_league_reference_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._runtime_start = 100.0
    runtime_any._runtime_last_metrics_time = 108.0
    runtime_any._runtime_cumulative_env_steps = 0
    runtime_any._last_published_snapshot_version = 5
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 0
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
        )
    )
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._pfsp_epoch = 0

    monkeypatch.setattr("weiss_rl.runtime.time.time", lambda: 110.0)
    metrics = QueueRuntime._runtime_metrics(
        runtime,
        [_make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=5)],
        occupancy_samples=[0.5],
    )

    assert metrics["league_effective_update"] == pytest.approx(5.0)
    assert metrics["league_raw_effective_update"] == pytest.approx(0.0)
    assert metrics["league_update_lag"] == pytest.approx(0.0)
    assert metrics["pfsp_sampling_weight_mirror"] == pytest.approx(1.0)


def test_resolve_actor_topology_keeps_ordered_runtime_strict_layout() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=96,
        runtime_mode="train_ordered",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 12
    assert envs_per_actor == 8


def test_resolve_actor_topology_prefers_fatter_async_collectors() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=96,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 2
    assert envs_per_actor == 48


def test_resolve_actor_topology_prefers_64_envs_per_actor_when_available() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=128,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 2
    assert envs_per_actor == 64


def test_resolve_actor_topology_prefers_6x64_over_8x48_for_384_envs() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=384,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 6
    assert envs_per_actor == 64


def test_runtime_honors_non_cpu_actor_device_and_disables_process_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(
            Any, SimpleNamespace(actor_id=actor_id, env=SimpleNamespace(close=lambda: None))
        ),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cuda:0", actor_torch_threads=1),
                    training=SimpleNamespace(mixed_precision=True),
                    experiment=SimpleNamespace(role="baseline_noleague"),
                    league=None,
                    model=SimpleNamespace(encoder_kind="typed_v1"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, object()),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._device == torch.device("cuda:0")
        assert runtime_any._actor_amp_enabled is True
        assert runtime_any._use_process_collectors is False
    finally:
        runtime.close()


def test_resolve_actor_device_layout_spreads_cuda_auto_across_non_learner_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.device_count", lambda: 4)

    stack = cast(
        Any,
        SimpleNamespace(
            config=SimpleNamespace(system=SimpleNamespace(actor_device="cuda:auto", learner_device="cuda:auto"))
        ),
    )

    layout = resolve_actor_device_layout(
        stack,
        actor_count=5,
        learner_device=torch.device("cuda:0"),
        prefer_process_collectors=True,
    )

    assert layout == ("cuda:1", "cuda:2", "cuda:3", "cuda:1", "cuda:2")


def test_runtime_can_force_process_collectors_for_structured_cuda_auto_async_league(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DummyProcessModel:
        def to(self, device: torch.device) -> _DummyProcessModel:
            return self

    started_with: list[Any] = []

    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr(
        QueueRuntime,
        "_start_process_collectors",
        lambda self, model: started_with.append(model),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = _DummyProcessModel()
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(
                        actor_device="cuda:auto",
                        learner_device="cuda:auto",
                        actor_torch_threads=1,
                        collection_backend="process",
                    ),
                    training=SimpleNamespace(
                        mixed_precision=True,
                        compile_learner=False,
                        structured_warmstart=SimpleNamespace(enabled=True),
                    ),
                    experiment=SimpleNamespace(role="main"),
                    league=SimpleNamespace(enabled=True, pfsp_window_episodes=50_000),
                    model=SimpleNamespace(encoder_kind="structured_v2"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=4,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
        learner_device=torch.device("cuda:0"),
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._league_enabled is True
        assert runtime_any._collection_backend == "process"
        assert runtime_any._use_central_batched_collection is False
        assert runtime_any._use_process_collectors is True
        assert runtime_any._process_actor_device_names == ("cuda:1", "cuda:2", "cuda:3", "cuda:1")
        assert started_with == [dummy_model]
    finally:
        runtime.close()


def test_runtime_uses_central_batched_collection_for_typed_cpu_async(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(
            Any,
            SimpleNamespace(
                actor_id=actor_id,
                env=SimpleNamespace(close=lambda: None),
                model=model,
                compiled_model=None,
                opponent_policy_id_by_env=np.full((64,), "latest_policy_mirror", dtype=object),
                seat_hidden=torch.zeros((64, 4)),
                opponent_hidden=torch.zeros((64, 4)),
                current_batch=SimpleNamespace(obs=np.zeros((64, 8), dtype=np.float32)),
                layout_name="i16_legal_ids",
                focal_seat_by_env=np.zeros((64,), dtype=np.int64),
                rng=np.random.default_rng(7),
                snapshot_version=0,
                next_unroll_seq=0,
            ),
        ),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1),
                    training=SimpleNamespace(mixed_precision=False, compile_learner=False),
                    experiment=SimpleNamespace(role="baseline_noleague"),
                    league=None,
                    model=SimpleNamespace(encoder_kind="typed_v1"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_uses_central_batched_collection_for_structured_cpu_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(
            Any,
            SimpleNamespace(
                actor_id=actor_id,
                env=SimpleNamespace(close=lambda: None),
                model=model,
                compiled_model=None,
                opponent_policy_id_by_env=np.full((64,), "latest_policy_mirror", dtype=object),
                seat_hidden=torch.zeros((64, 4)),
                opponent_hidden=torch.zeros((64, 4)),
                current_batch=SimpleNamespace(obs=np.zeros((64, 8), dtype=np.float32)),
                layout_name="i16_legal_ids",
                focal_seat_by_env=np.zeros((64,), dtype=np.int64),
                rng=np.random.default_rng(7),
                snapshot_version=0,
                next_unroll_seq=0,
            ),
        ),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1),
                    training=SimpleNamespace(
                        mixed_precision=False,
                        compile_learner=False,
                        structured_warmstart=SimpleNamespace(enabled=True),
                    ),
                    experiment=SimpleNamespace(role="baseline_noleague"),
                    league=None,
                    model=SimpleNamespace(encoder_kind="structured_v2"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_uses_central_batched_collection_for_structured_cuda_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyCentralModel:
        def to(self, device: torch.device) -> _DummyCentralModel:
            return self

        def eval(self) -> _DummyCentralModel:
            return self

    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(
            Any,
            SimpleNamespace(
                actor_id=actor_id,
                env=SimpleNamespace(close=lambda: None),
                model=model,
                compiled_model=None,
                opponent_policy_id_by_env=np.full((64,), "latest_policy_mirror", dtype=object),
                seat_hidden=torch.zeros((64, 4), device=torch.device("cpu")),
                opponent_hidden=torch.zeros((64, 4), device=torch.device("cpu")),
                current_batch=SimpleNamespace(obs=np.zeros((64, 8), dtype=np.float32)),
                layout_name="i16_legal_ids",
                focal_seat_by_env=np.zeros((64,), dtype=np.int64),
                rng=np.random.default_rng(7),
                snapshot_version=0,
                next_unroll_seq=0,
            ),
        ),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = _DummyCentralModel()
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cuda:0", actor_torch_threads=1),
                    training=SimpleNamespace(
                        mixed_precision=True,
                        compile_learner=False,
                        structured_warmstart=SimpleNamespace(enabled=True),
                    ),
                    experiment=SimpleNamespace(role="baseline_noleague"),
                    league=None,
                    model=SimpleNamespace(encoder_kind="structured_v2"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._device == torch.device("cuda:0")
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_keeps_central_batched_collection_for_typed_cpu_async_league(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(
            Any,
            SimpleNamespace(
                actor_id=actor_id,
                env=SimpleNamespace(close=lambda: None),
                model=model,
                compiled_model=None,
                opponent_policy_id_by_env=np.full((64,), "latest_policy_mirror", dtype=object),
                seat_hidden=torch.zeros((64, 4)),
                opponent_hidden=torch.zeros((64, 4)),
                current_batch=SimpleNamespace(obs=np.zeros((64, 8), dtype=np.float32)),
                layout_name="i16_legal_ids",
                focal_seat_by_env=np.zeros((64,), dtype=np.int64),
                rng=np.random.default_rng(7),
                snapshot_version=0,
                next_unroll_seq=0,
            ),
        ),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1),
                    training=SimpleNamespace(mixed_precision=False, compile_learner=False),
                    experiment=SimpleNamespace(role="main"),
                    league=SimpleNamespace(enabled=True, pfsp_window_episodes=50_000),
                    model=SimpleNamespace(encoder_kind="typed_v1"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._league_enabled is True
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_can_force_process_collectors_for_structured_cpu_async_league(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started_with: list[Any] = []

    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_start_process_collectors",
        lambda self, model: started_with.append(model),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1, collection_backend="process"),
                    training=SimpleNamespace(
                        mixed_precision=False,
                        compile_learner=False,
                        structured_warmstart=SimpleNamespace(enabled=True),
                    ),
                    experiment=SimpleNamespace(role="main"),
                    league=SimpleNamespace(enabled=True, pfsp_window_episodes=50_000),
                    model=SimpleNamespace(encoder_kind="structured_v2"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._league_enabled is True
        assert runtime_any._collection_backend == "process"
        assert runtime_any._use_central_batched_collection is False
        assert runtime_any._use_process_collectors is True
        assert started_with == [dummy_model]
    finally:
        runtime.close()


def test_runtime_process_collectors_start_before_refreshing_opponent_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_order: list[str] = []

    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_start_process_collectors",
        lambda self, model: call_order.append("start"),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: call_order.append("refresh"))

    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1, collection_backend="process"),
                    training=SimpleNamespace(
                        mixed_precision=False,
                        compile_learner=False,
                        structured_warmstart=SimpleNamespace(enabled=True),
                    ),
                    experiment=SimpleNamespace(role="main"),
                    league=SimpleNamespace(enabled=True, pfsp_window_episodes=50_000),
                    model=SimpleNamespace(encoder_kind="structured_v2"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, torch.nn.Linear(8, 4)),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
    )
    try:
        assert call_order == ["start", "refresh"]
    finally:
        runtime.close()


def test_fill_pending_unrolls_spills_shared_slots_when_target_exceeds_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)

    class _ResultQueue:
        def __init__(self, payloads: list[dict[str, Any]]) -> None:
            self._payloads = deque(payloads)

        def get(self) -> dict[str, Any]:
            return self._payloads.popleft()

    freed_slots: list[int] = []
    copied_payloads: list[tuple[int, int]] = []

    runtime_any._collector_result_queue = _ResultQueue(
        [
            {"actor_id": 0, "slot_id": 0, "unroll_seq": 1, "behavior_policy_version": 1, "unroll_hash": "0:1:1"},
            {"actor_id": 0, "slot_id": 0, "unroll_seq": 2, "behavior_policy_version": 1, "unroll_hash": "0:2:1"},
        ]
    )
    runtime_any._use_shared_collector_transport = True
    runtime_any._collector_shared_slots = {0: (object(),)}
    runtime_any._collector_free_queues = [SimpleNamespace(put=lambda slot_id: freed_slots.append(int(slot_id)))]
    runtime_any._pending_unrolls = deque()
    runtime_any.config = SimpleNamespace(queue_capacity_unrolls=8)

    def fake_read(slot: object, metadata: dict[str, Any]) -> SimpleNamespace:
        del slot
        copied_payloads.append((int(metadata["actor_id"]), int(metadata["slot_id"])))
        return SimpleNamespace(kind="copied", unroll_seq=int(metadata["unroll_seq"]))

    monkeypatch.setattr("weiss_rl.runtime._read_unroll_from_shared_slot", fake_read)

    occupancy_samples: list[float] = []
    runtime._fill_pending_unrolls(target_count=2, occupancy_samples=occupancy_samples)

    assert len(runtime_any._pending_unrolls) == 2
    assert all(not isinstance(item, _SharedPendingUnroll) for item in runtime_any._pending_unrolls)
    assert copied_payloads == [(0, 0), (0, 0)]
    assert freed_slots == [0, 0]


def test_fill_pending_unrolls_keeps_shared_views_when_target_fits_capacity() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)

    class _ResultQueue:
        def __init__(self, payloads: list[dict[str, Any]]) -> None:
            self._payloads = deque(payloads)

        def get(self) -> dict[str, Any]:
            return self._payloads.popleft()

    runtime_any._collector_result_queue = _ResultQueue(
        [
            {"actor_id": 0, "slot_id": 0, "unroll_seq": 1, "behavior_policy_version": 1, "unroll_hash": "0:1:1"},
            {"actor_id": 1, "slot_id": 0, "unroll_seq": 2, "behavior_policy_version": 1, "unroll_hash": "1:2:1"},
        ]
    )
    runtime_any._use_shared_collector_transport = True
    runtime_any._collector_shared_slots = {0: (object(),), 1: (object(),)}
    runtime_any._collector_free_queues = [queue.Queue(), queue.Queue()]
    runtime_any._pending_unrolls = deque()
    runtime_any.config = SimpleNamespace(queue_capacity_unrolls=8)
    runtime_any._profile_timers = False

    occupancy_samples: list[float] = []
    runtime._fill_pending_unrolls(target_count=2, occupancy_samples=occupancy_samples)

    assert len(runtime_any._pending_unrolls) == 2
    assert all(isinstance(item, _SharedPendingUnroll) for item in runtime_any._pending_unrolls)
    assert runtime_any._collector_free_queues[0].empty()
    assert runtime_any._collector_free_queues[1].empty()


def test_fill_pending_unrolls_waits_for_diverse_lane_payloads_when_quota_enabled() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)

    class _ResultQueue:
        def __init__(self, payloads: list[RuntimeUnroll]) -> None:
            self._payloads = deque(payloads)

        def get(self, timeout: float | None = None) -> RuntimeUnroll:
            del timeout
            if not self._payloads:
                raise queue.Empty
            return self._payloads.popleft()

    runtime_any.config = SimpleNamespace(queue_capacity_unrolls=8)
    runtime_any._collector_result_queue = _ResultQueue(
        [
            _make_runtime_unroll(actor_id=4, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=5, unroll_seq=1, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=2, behavior_policy_version=0),
        ]
    )
    runtime_any._use_shared_collector_transport = False
    runtime_any._pending_unrolls = deque()
    runtime_any._diverse_opponent_actor_count = 2
    runtime_any._diverse_opponent_batch_fraction = 0.5
    runtime_any._diverse_opponent_batch_wait_ms = 50

    occupancy_samples: list[float] = []
    QueueRuntime._fill_pending_unrolls(runtime, target_count=2, occupancy_samples=occupancy_samples)

    assert [item.actor_id for item in runtime_any._pending_unrolls] == [4, 5, 0]
    assert QueueRuntime._pending_diverse_unroll_count(runtime) == 1


def test_fill_pending_unrolls_uses_parallel_executor_for_distinct_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=4,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=4,
        queue_capacity_unrolls=8,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._actors = [cast(Any, SimpleNamespace(actor_id=actor_id)) for actor_id in range(4)]
    runtime_any._collector_result_queue = None
    runtime_any._pending_unrolls = deque()
    runtime_any._next_actor_index = 0
    runtime_any._use_central_batched_collection = False

    submitted_actor_ids: list[int] = []

    class _ImmediateFuture:
        def __init__(self, value: RuntimeUnroll) -> None:
            self._value = value

        def result(self) -> RuntimeUnroll:
            return self._value

    class _FakeExecutor:
        def submit(self, fn, actor):
            submitted_actor_ids.append(int(actor.actor_id))
            return _ImmediateFuture(fn(actor))

    runtime_any._collector_executor = _FakeExecutor()
    runtime_any._collect_actor_unroll = lambda actor: _make_runtime_unroll(
        actor_id=int(actor.actor_id),
        unroll_seq=0,
        behavior_policy_version=0,
    )

    occupancy_samples: list[float] = []
    QueueRuntime._fill_pending_unrolls(runtime, target_count=4, occupancy_samples=occupancy_samples)

    assert submitted_actor_ids == [0, 1, 2, 3]
    assert [item.actor_id for item in runtime_any._pending_unrolls] == [0, 1, 2, 3]
    assert occupancy_samples == [0.0]


def test_shared_collector_slot_round_trip_preserves_packed_unroll_payload() -> None:
    slot_config = _create_shared_collector_slot_config(
        actor_id=0,
        profile="fast",
        unroll_length=2,
        envs_per_actor=2,
        observation_dim=3,
        action_dim=5,
        hidden_size=4,
        layout_name="i16_legal_ids",
    )
    slot = _open_shared_collector_slot(slot_config, create=True)
    try:
        packed = LegalActionBatch.from_packed(
            np.array([0, 1, 2, 3, 4, 1], dtype=np.uint32),
            np.array([0, 2, 3, 5, 6], dtype=np.uint32),
            meta=np.array(
                [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [2, 0, 0, 0],
                    [3, 0, 1, 0],
                    [3, 1, 1, 0],
                    [8, 0, 0, 0],
                ],
                dtype=np.uint16,
            ),
            action_space=5,
        )
        unroll = RuntimeUnroll(
            actor_id=0,
            unroll_seq=7,
            behavior_policy_version=11,
            unroll_hash="roundtrip",
            obs=np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            actions=np.array([[1, 2], [3, 4]], dtype=np.uint16),
            rewards=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            terminated=np.array([[False, True], [False, False]], dtype=np.bool_),
            truncated=np.array([[False, False], [True, False]], dtype=np.bool_),
            to_play_seat=np.array([[0, 1], [1, 0]], dtype=np.int8),
            behavior_logp=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            values=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            legal_actions=packed,
            bootstrap_obs=np.arange(6, dtype=np.float32).reshape(2, 3),
            bootstrap_actor=np.array([0, 1], dtype=np.int64),
            bootstrap_value=np.array([0.25, -0.5], dtype=np.float32),
            initial_hidden_state=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
            final_hidden_state=np.arange(16, 32, dtype=np.float32).reshape(2, 2, 4),
            episode_seed=np.array([[5, 6], [7, 8]], dtype=np.uint64),
            policy_train_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
            b1_opponent_mask=np.array([[True, False], [False, True]], dtype=np.bool_),
            teacher_family=np.array([[1, 2], [3, -1]], dtype=np.int32),
            teacher_slot=np.array([[0, -1], [2, -1]], dtype=np.int32),
            teacher_move_source=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_attack_type=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_action=np.array([[4, 9], [12, -1]], dtype=np.int32),
            teacher_valid=np.array([[True, True], [True, False]], dtype=np.bool_),
            behavior_logits=None,
        )

        _write_unroll_to_shared_slot(slot, unroll)
        restored = _read_unroll_from_shared_slot(slot, _shared_unroll_metadata(unroll))

        assert restored.actor_id == unroll.actor_id
        assert restored.unroll_seq == unroll.unroll_seq
        assert restored.behavior_policy_version == unroll.behavior_policy_version
        assert np.array_equal(restored.obs, unroll.obs)
        assert np.array_equal(restored.actions, unroll.actions)
        assert np.array_equal(restored.bootstrap_obs, unroll.bootstrap_obs)
        assert np.array_equal(restored.bootstrap_value, unroll.bootstrap_value)
        assert np.array_equal(restored.final_hidden_state, unroll.final_hidden_state)
        assert np.array_equal(restored.teacher_family, unroll.teacher_family)
        assert np.array_equal(restored.teacher_slot, unroll.teacher_slot)
        assert np.array_equal(restored.teacher_move_source, unroll.teacher_move_source)
        assert np.array_equal(restored.teacher_attack_type, unroll.teacher_attack_type)
        assert np.array_equal(restored.teacher_action, unroll.teacher_action)
        assert np.array_equal(restored.teacher_valid, unroll.teacher_valid)
        assert restored.legal_actions.ids is not None
        assert restored.legal_actions.offsets is not None
        assert restored.legal_actions.action_space == 5
        assert restored.legal_actions.ids.tolist() == unroll.legal_actions.ids.tolist()
        assert restored.legal_actions.offsets.tolist() == unroll.legal_actions.offsets.tolist()
        assert restored.legal_actions.meta is not None
        assert restored.legal_actions.meta.tolist() == unroll.legal_actions.meta.tolist()
    finally:
        slot.close(unlink=True)


def test_shared_pending_unroll_keeps_shared_views_until_release() -> None:
    slot_config = _create_shared_collector_slot_config(
        actor_id=1,
        slot_id=3,
        profile="fast",
        unroll_length=2,
        envs_per_actor=2,
        observation_dim=3,
        action_dim=5,
        hidden_size=4,
        layout_name="i16_legal_ids",
    )
    slot = _open_shared_collector_slot(slot_config, create=True)
    try:
        packed = LegalActionBatch.from_packed(
            np.array([0, 1, 2, 3, 4, 1], dtype=np.uint32),
            np.array([0, 2, 3, 5, 6], dtype=np.uint32),
            meta=np.array(
                [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [2, 0, 0, 0],
                    [3, 0, 1, 0],
                    [3, 1, 1, 0],
                    [8, 0, 0, 0],
                ],
                dtype=np.uint16,
            ),
            action_space=5,
        )
        unroll = RuntimeUnroll(
            actor_id=1,
            unroll_seq=9,
            behavior_policy_version=4,
            unroll_hash="shared-view",
            obs=np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            actions=np.array([[1, 2], [3, 4]], dtype=np.uint16),
            rewards=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            terminated=np.array([[False, True], [False, False]], dtype=np.bool_),
            truncated=np.array([[False, False], [True, False]], dtype=np.bool_),
            to_play_seat=np.array([[0, 1], [1, 0]], dtype=np.int8),
            behavior_logp=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            values=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            legal_actions=packed,
            bootstrap_obs=np.arange(6, dtype=np.float32).reshape(2, 3),
            bootstrap_actor=np.array([0, 1], dtype=np.int64),
            bootstrap_value=np.array([0.25, -0.5], dtype=np.float32),
            initial_hidden_state=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
            final_hidden_state=np.arange(16, 32, dtype=np.float32).reshape(2, 2, 4),
            episode_seed=np.array([[5, 6], [7, 8]], dtype=np.uint64),
            policy_train_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
            b1_opponent_mask=np.array([[True, False], [False, True]], dtype=np.bool_),
            teacher_family=np.array([[1, 2], [3, -1]], dtype=np.int32),
            teacher_slot=np.array([[0, -1], [2, -1]], dtype=np.int32),
            teacher_move_source=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_attack_type=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_action=np.array([[4, 9], [12, -1]], dtype=np.int32),
            teacher_valid=np.array([[True, True], [True, False]], dtype=np.bool_),
            behavior_logits=None,
        )

        _write_unroll_to_shared_slot(slot, unroll)
        pending = _SharedPendingUnroll.from_metadata(slot, _shared_unroll_metadata(unroll, slot_id=3))

        assert pending.slot_id == 3
        assert pending.obs is slot.obs
        assert np.shares_memory(pending.legal_actions.ids, slot.legal_ids)
        assert np.shares_memory(pending.legal_actions.meta, slot.legal_action_meta)
        assert pending.teacher_move_source is slot.teacher_move_source
        assert pending.teacher_action is slot.teacher_action

        runtime = object.__new__(QueueRuntime)
        runtime_any = cast(Any, runtime)
        runtime_any._use_shared_collector_transport = True
        runtime_any._collector_free_queues = [queue.Queue(), queue.Queue()]

        QueueRuntime._release_shared_pending_unrolls(runtime, [pending])

        assert runtime_any._collector_free_queues[1].get_nowait() == 3
    finally:
        slot.close(unlink=True)


def test_teacher_labels_from_actions_group_main_play_and_attack_semantics() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    action_catalog = _teacher_test_catalog()
    main_move_to_slot_2 = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if ((decoded := action_catalog.decode(action_id)).family == "main_move" and decoded.to_slot == 2)
    )
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_action_catalog = action_catalog
    runtime_any._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime_any._teacher_attack_type_index = {
        name: index for index, name in enumerate(action_catalog.attack_type_names)
    }

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_actions(
            runtime,
            row_indices=np.array([0, 1, 2, 3], dtype=np.int64),
            chosen_actions=np.array([0, 14, main_move_to_slot_2, 40], dtype=np.int64),
            num_rows=4,
        )
    )

    assert teacher_valid.tolist() == [True, True, True, True]
    assert teacher_family.tolist() == [
        runtime_any._teacher_family_index["main_play_character"],
        runtime_any._teacher_family_index["attack"],
        runtime_any._teacher_family_index["main_move"],
        runtime_any._teacher_family_index["pass"],
    ]
    assert teacher_slot.tolist() == [0, 1, 2, -1]
    assert teacher_move_source.tolist() == [-1, -1, 0, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1]
    assert teacher_action.tolist() == [0, 14, main_move_to_slot_2, 40]


def test_teacher_guidance_active_for_collection_respects_warmstart_only() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_aux_mode = "warmstart_only"
    runtime_any._teacher_guidance_warmstart_updates = 1
    runtime_any._current_learner_update = 0

    assert QueueRuntime._teacher_guidance_active_for_collection(runtime) is True

    runtime_any._current_learner_update = 1

    assert QueueRuntime._teacher_guidance_active_for_collection(runtime) is False


def test_teacher_labels_from_actions_skip_after_warmstart_only_phase() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    action_catalog = _teacher_test_catalog()
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_aux_mode = "warmstart_only"
    runtime_any._teacher_guidance_warmstart_updates = 1
    runtime_any._current_learner_update = 1
    runtime_any._teacher_action_catalog = action_catalog
    runtime_any._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime_any._teacher_attack_type_index = {
        name: index for index, name in enumerate(action_catalog.attack_type_names)
    }

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_actions(
            runtime,
            row_indices=np.array([0], dtype=np.int64),
            chosen_actions=np.array([0], dtype=np.int64),
            num_rows=1,
        )
    )

    assert teacher_valid.tolist() == [False]
    assert teacher_family.tolist() == [-1]
    assert teacher_slot.tolist() == [-1]
    assert teacher_move_source.tolist() == [-1]
    assert teacher_attack_type.tolist() == [-1]
    assert teacher_action.tolist() == [-1]


def test_concat_optional_time_major_field_fills_unlabeled_rows_with_sentinels() -> None:
    labeled = SimpleNamespace(
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        teacher_family=np.asarray([[7, 8], [9, 10]], dtype=np.int32),
        teacher_valid=np.asarray([[True, False], [False, True]], dtype=np.bool_),
    )
    unlabeled = SimpleNamespace(
        obs=np.zeros((2, 3, 1), dtype=np.float32),
        teacher_family=None,
        teacher_valid=None,
    )

    teacher_family = _concat_optional_time_major_field(
        [labeled, unlabeled],
        "teacher_family",
        missing_fill_value=-1,
    )
    teacher_valid = _concat_optional_time_major_field(
        [labeled, unlabeled],
        "teacher_valid",
        missing_fill_value=False,
    )

    assert teacher_family is not None
    assert teacher_valid is not None
    assert teacher_family.shape == (2, 5)
    assert teacher_valid.shape == (2, 5)
    npt.assert_array_equal(teacher_family[:, :2], labeled.teacher_family)
    npt.assert_array_equal(teacher_family[:, 2:], np.full((2, 3), -1, dtype=np.int32))
    npt.assert_array_equal(teacher_valid[:, :2], labeled.teacher_valid)
    npt.assert_array_equal(teacher_valid[:, 2:], np.zeros((2, 3), dtype=np.bool_))


def test_teacher_labels_from_ids_cover_public_decision_kinds_beyond_tactical_subset() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    action_catalog = _teacher_test_catalog()
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_policy = object()
    runtime_any._teacher_action_catalog = action_catalog
    runtime_any._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime_any._teacher_attack_type_index = {
        name: index for index, name in enumerate(action_catalog.attack_type_names)
    }
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.asarray(
        [
            int(kwargs["legal_ids"][int(kwargs["legal_offsets"][row_index])])
            for row_index in np.asarray(kwargs["row_indices"], dtype=np.int64).tolist()
        ],
        dtype=np.int64,
    )

    legal_ids = np.asarray([0, 40, 14, 40, 39, 40, 40, 40], dtype=np.uint32)
    legal_offsets = np.asarray([0, 2, 4, 6, 8], dtype=np.uint32)
    counters = {"teacher_tactical_row_count": 0}

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_ids(
            runtime,
            focal_rows=np.asarray([True, True, True, True], dtype=np.bool_),
            decision_kind=np.asarray([1, 5, 8, -1], dtype=np.int32),
            obs_step=np.zeros((4, 4), dtype=np.float32),
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=None,
            counters=counters,
        )
    )

    assert teacher_valid.tolist() == [True, True, True, False]
    assert teacher_family.tolist() == [
        runtime_any._teacher_family_index["main_play_character"],
        runtime_any._teacher_family_index["attack"],
        runtime_any._teacher_family_index["climax_play"],
        -1,
    ]
    assert teacher_slot.tolist() == [0, 1, -1, -1]
    assert teacher_move_source.tolist() == [-1, -1, -1, -1]
    assert teacher_attack_type.tolist() == [-1, 1, -1, -1]
    assert teacher_action.tolist() == [0, 14, 39, -1]
    assert counters["teacher_tactical_row_count"] == 3


def test_teacher_labels_from_ids_cover_mulligan_decision_kind_zero() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    action_catalog = _teacher_mulligan_test_catalog()
    runtime_any._teacher_guidance_enabled = True
    runtime_any._teacher_policy = object()
    runtime_any._teacher_action_catalog = action_catalog
    runtime_any._teacher_family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    runtime_any._teacher_attack_type_index = {}
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.asarray(
        [
            int(kwargs["legal_ids"][int(kwargs["legal_offsets"][row_index])])
            for row_index in np.asarray(kwargs["row_indices"], dtype=np.int64).tolist()
        ],
        dtype=np.int64,
    )

    legal_ids = np.asarray([0, 1, 2, 6], dtype=np.uint32)
    legal_offsets = np.asarray([0, 1, 3, 4], dtype=np.uint32)
    counters = {"teacher_tactical_row_count": 0}

    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, teacher_action, teacher_valid = (
        QueueRuntime._teacher_labels_from_ids(
            runtime,
            focal_rows=np.asarray([True, True, True], dtype=np.bool_),
            decision_kind=np.asarray([0, 0, -1], dtype=np.int32),
            obs_step=np.zeros((3, 4), dtype=np.float32),
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=None,
            counters=counters,
        )
    )

    assert teacher_valid.tolist() == [True, True, False]
    assert teacher_family.tolist() == [
        runtime_any._teacher_family_index["mulligan_confirm"],
        runtime_any._teacher_family_index["mulligan_select"],
        -1,
    ]
    assert teacher_slot.tolist() == [-1, -1, -1]
    assert teacher_move_source.tolist() == [-1, -1, -1]
    assert teacher_attack_type.tolist() == [-1, -1, -1]
    assert teacher_action.tolist() == [0, 1, -1]
    assert counters["teacher_tactical_row_count"] == 2


def test_public_teacher_rows_cover_mulligan_and_public_decision_kinds() -> None:
    runtime = object.__new__(QueueRuntime)

    rows = QueueRuntime._public_teacher_rows(
        runtime,
        focal_rows=np.asarray([True, True, True, False, True], dtype=np.bool_),
        decision_kind=np.asarray([0, 1, 8, 5, -1], dtype=np.int32),
    )

    assert rows.tolist() == [0, 1, 2]
