from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from weiss_rl.artifacts.reproducibility import derive_actor_seed
from weiss_rl.runtime import (
    QueueRuntime,
    QueueRuntimeConfig,
    _resolve_actor_topology,
    resolve_actor_device_layout,
)
from weiss_rl.runtime.components import process as process_components
from weiss_rl.runtime.components.topology import actor_seed as topology_actor_seed


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


def test_topology_actor_seed_matches_reproducibility_contract() -> None:
    assert topology_actor_seed(20260514, 0) == derive_actor_seed(20260514, actor_id=0)
    assert topology_actor_seed(20260514, 5) == derive_actor_seed(20260514, actor_id=5)


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


def test_runtime_auto_prefers_process_collectors_for_structured_cpu_model_async(
    monkeypatch: pytest.MonkeyPatch,
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
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1),
                    training=SimpleNamespace(
                        mixed_precision=False,
                        compile_learner=False,
                        actor_policy_backend="model",
                        structured_warmstart=SimpleNamespace(enabled=False),
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
        assert runtime_any._collection_backend == "auto"
        assert runtime_any._use_central_batched_collection is False
        assert runtime_any._use_process_collectors is True
        assert started_with == [dummy_model]
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
                        structured_warmstart=SimpleNamespace(enabled=False),
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


def test_collector_process_main_forwards_attack_guard_to_child_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_configs: list[QueueRuntimeConfig] = []

    class _DummyModel(torch.nn.Module):
        def to(self, device: torch.device) -> _DummyModel:
            return self

        def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
            return None

        def eval(self) -> _DummyModel:
            return self

    class _FakeRuntime:
        def __init__(
            self,
            *,
            config: QueueRuntimeConfig,
            **_kwargs: Any,
        ) -> None:
            self.config = config
            self._actors = [
                SimpleNamespace(
                    env=SimpleNamespace(close=lambda: None),
                    fixed_opponent_policy_id_by_env=None,
                )
            ]
            self._forced_fixed_opponent_policy_ids = ()
            self._opponent_heuristic_policies = {}
            self._opponent_models = {}
            captured_configs.append(config)

        def close(self) -> None:
            return None

    monkeypatch.setattr(process_components, "build_policy_value_model", lambda **_kwargs: _DummyModel())
    monkeypatch.setattr(process_components, "process_debug_log", lambda **_kwargs: None)
    monkeypatch.setattr(process_components, "_collector_loop", lambda **_kwargs: None)

    process_components.collector_process_main(
        runtime_cls=_FakeRuntime,
        stack=cast(Any, SimpleNamespace(config=SimpleNamespace(system=None, model=SimpleNamespace()))),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=6,
            envs_per_actor=48,
            unroll_length=64,
            batch_unrolls_per_update=64,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
            mulligan_force_confirm_after_select=True,
            actor_sampling_temperature=0.25,
            force_pass_over_main_move_only=True,
            force_attack_over_pass_when_attack_legal=True,
        ),
        model_state_dict={},
        observation_dim=8,
        action_dim=64,
        observation_spec=None,
        spec_bundle=None,
        run_dir=None,
        actor_id=0,
        actor_device_name=None,
        learner_device_name=None,
        control_queue=SimpleNamespace(),
        free_queue=None,
        result_queue=SimpleNamespace(),
        shared_slot_configs=None,
    )

    assert captured_configs
    assert captured_configs[0].mulligan_force_confirm_after_select is True
    assert captured_configs[0].actor_sampling_temperature == pytest.approx(0.25)
    assert captured_configs[0].force_pass_over_main_move_only is True
    assert captured_configs[0].force_attack_over_pass_when_attack_legal is True


def test_collector_process_main_preserves_parent_global_diverse_lane_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_build: list[tuple[int, int, int]] = []

    class _DummyModel(torch.nn.Module):
        def to(self, device: torch.device) -> _DummyModel:
            return self

        def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
            return None

        def eval(self) -> _DummyModel:
            return self

    class _FakeRuntime:
        def __init__(
            self,
            *,
            config: QueueRuntimeConfig,
            **_kwargs: Any,
        ) -> None:
            self.config = config
            self._actors = [
                SimpleNamespace(
                    env=SimpleNamespace(close=lambda: None),
                    fixed_opponent_policy_id_by_env=None,
                )
            ]
            self._forced_fixed_opponent_policy_ids = ()
            self._opponent_heuristic_policies = {}
            self._opponent_models = {}
            self._diverse_opponent_actor_count = min(int(config.actor_count), 999)
            self._diverse_model_actor_count = min(int(config.actor_count), 999)

        def _build_actor_state(self, *, model: torch.nn.Module, actor_id: int) -> SimpleNamespace:
            del model
            captured_build.append(
                (
                    int(actor_id),
                    int(self._diverse_opponent_actor_count),
                    int(self._diverse_model_actor_count),
                )
            )
            return SimpleNamespace(
                env=SimpleNamespace(close=lambda: None),
                fixed_opponent_policy_id_by_env=None,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(process_components, "build_policy_value_model", lambda **_kwargs: _DummyModel())
    monkeypatch.setattr(process_components, "process_debug_log", lambda **_kwargs: None)
    monkeypatch.setattr(process_components, "_collector_loop", lambda **_kwargs: None)

    process_components.collector_process_main(
        runtime_cls=_FakeRuntime,
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=None,
                    model=SimpleNamespace(),
                    training=SimpleNamespace(
                        diverse_opponent_actor_count=999,
                        diverse_model_actor_count=999,
                    ),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=6,
            envs_per_actor=48,
            unroll_length=64,
            batch_unrolls_per_update=64,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model_state_dict={},
        observation_dim=8,
        action_dim=64,
        observation_spec=None,
        spec_bundle=None,
        run_dir=None,
        actor_id=2,
        actor_device_name=None,
        learner_device_name=None,
        control_queue=SimpleNamespace(),
        free_queue=None,
        result_queue=SimpleNamespace(),
        shared_slot_configs=None,
    )

    assert captured_build == [(2, 6, 6)]
