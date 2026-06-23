from __future__ import annotations

from typing import Any, cast

from weiss_rl.runtime import QueueRuntime


def test_disable_mirror_policy_fusion_context_restores_previous_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._disable_mirror_policy_fusion = False

    with QueueRuntime.disable_mirror_policy_fusion(runtime):
        assert runtime_any._disable_mirror_policy_fusion is True

    assert runtime_any._disable_mirror_policy_fusion is False

    runtime_any._disable_mirror_policy_fusion = True  # type: ignore[unreachable]
    with QueueRuntime.disable_mirror_policy_fusion(runtime):
        assert runtime_any._disable_mirror_policy_fusion is True
    assert runtime_any._disable_mirror_policy_fusion is True
