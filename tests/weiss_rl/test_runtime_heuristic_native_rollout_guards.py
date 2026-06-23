from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_POLICY_ID,
)
from weiss_rl.runtime import (
    QueueRuntime,
)


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
    runtime_any.config = SimpleNamespace()

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


def test_collect_all_heuristic_ids_native_rollout_rejects_rl_action_surface_guards() -> None:
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
    runtime_any.config = SimpleNamespace(force_attack_over_pass_when_attack_legal=True)

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
    runtime_any.config = SimpleNamespace()

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
