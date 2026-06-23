from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from weiss_rl.runtime import QueueRuntimeConfig
from weiss_rl.runtime.components import process as process_components


class _DummyProcessModel(torch.nn.Module):
    def to(self, device: torch.device) -> _DummyProcessModel:
        del device
        return self

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        del state_dict, strict
        return None

    def eval(self) -> _DummyProcessModel:
        return self


def test_collector_process_main_forwards_attack_guard_to_child_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_configs: list[QueueRuntimeConfig] = []

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

    monkeypatch.setattr(process_components, "build_policy_value_model", lambda **_kwargs: _DummyProcessModel())
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

    monkeypatch.setattr(process_components, "build_policy_value_model", lambda **_kwargs: _DummyProcessModel())
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
