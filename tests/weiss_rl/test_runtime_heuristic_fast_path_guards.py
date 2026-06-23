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
