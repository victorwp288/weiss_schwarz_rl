from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.runtime.components.heuristic_fast_path import (
    actor_fixed_opponents_all_heuristic_public,
    can_collect_all_heuristic_ids_fast,
    can_collect_all_heuristic_ids_native_rollout,
    simulator_native_fixed_opponent_available,
)


def _actor(*, opponent_ids: tuple[str, ...], fixed_ids: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        layout_name="i16_legal_ids",
        env=SimpleNamespace(
            pool=SimpleNamespace(
                choose_heuristic_public_actions_into=lambda *args, **kwargs: None,
                rollout_heuristic_public_into_i16_legal_ids=lambda *args, **kwargs: None,
                reset_done_into_i16_legal_ids=lambda *args, **kwargs: None,
            )
        ),
        opponent_policy_id_by_env=np.asarray(opponent_ids, dtype=object),
        fixed_opponent_policy_id_by_env=np.asarray(fixed_ids, dtype=object),
    )


def test_simulator_native_fixed_opponent_available_requires_backend_and_pool_hook() -> None:
    actor = _actor(opponent_ids=(HEURISTIC_PUBLIC_POLICY_ID,))

    assert simulator_native_fixed_opponent_available(actor, fixed_opponent_backend="simulator_native") is True
    assert simulator_native_fixed_opponent_available(actor, fixed_opponent_backend="python_batched") is False
    assert simulator_native_fixed_opponent_available(None, fixed_opponent_backend="simulator_native") is False


def test_actor_fixed_opponents_all_heuristic_public_ignores_empty_and_inactive_slots() -> None:
    actor = _actor(
        opponent_ids=(HEURISTIC_PUBLIC_POLICY_ID,),
        fixed_ids=("", "inactive_snapshot", HEURISTIC_PUBLIC_POLICY_ID),
    )

    assert (
        actor_fixed_opponents_all_heuristic_public(
            actor,
            fixed_opponent_policy_is_active=lambda policy_id: policy_id != "inactive_snapshot",
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        )
        is True
    )


def test_can_collect_all_heuristic_ids_fast_preserves_all_gate_conditions() -> None:
    actor = _actor(
        opponent_ids=(HEURISTIC_PUBLIC_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID),
        fixed_ids=("", HEURISTIC_PUBLIC_POLICY_ID),
    )

    assert (
        can_collect_all_heuristic_ids_fast(
            actor,
            actor_policy_backend="heuristic_public",
            active_actor_heuristic_fraction=1.0,
            fixed_opponent_backend="simulator_native",
            teacher_policy=object(),
            league_config=object(),
            active_heuristic_public_mix_fraction=1.0,
            fixed_opponent_policy_is_active=lambda policy_id: bool(policy_id),
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        )
        is True
    )

    actor.opponent_policy_id_by_env = np.asarray((HEURISTIC_PUBLIC_POLICY_ID, "snapshot"), dtype=object)

    assert (
        can_collect_all_heuristic_ids_fast(
            actor,
            actor_policy_backend="heuristic_public",
            active_actor_heuristic_fraction=1.0,
            fixed_opponent_backend="simulator_native",
            teacher_policy=object(),
            league_config=object(),
            active_heuristic_public_mix_fraction=1.0,
            fixed_opponent_policy_is_active=lambda policy_id: bool(policy_id),
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        )
        is False
    )


def test_can_collect_all_heuristic_ids_native_rollout_requires_stateless_value_free_actor() -> None:
    actor = _actor(opponent_ids=(HEURISTIC_PUBLIC_POLICY_ID,))

    assert (
        can_collect_all_heuristic_ids_native_rollout(
            actor,
            heuristic_native_rollout_enabled=True,
            actor_policy_backend="heuristic_public",
            active_actor_heuristic_fraction=1.0,
            fixed_opponent_backend="simulator_native",
            teacher_policy=object(),
            league_config=object(),
            active_heuristic_public_mix_fraction=1.0,
            fixed_opponent_policy_is_active=lambda policy_id: bool(policy_id),
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
            actor_behavior_values_required=False,
            should_track_heuristic_actor_hidden_state=False,
        )
        is True
    )

    assert (
        can_collect_all_heuristic_ids_native_rollout(
            actor,
            heuristic_native_rollout_enabled=True,
            actor_policy_backend="heuristic_public",
            active_actor_heuristic_fraction=1.0,
            fixed_opponent_backend="simulator_native",
            teacher_policy=object(),
            league_config=object(),
            active_heuristic_public_mix_fraction=1.0,
            fixed_opponent_policy_is_active=lambda policy_id: bool(policy_id),
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
            actor_behavior_values_required=True,
            should_track_heuristic_actor_hidden_state=False,
        )
        is False
    )
