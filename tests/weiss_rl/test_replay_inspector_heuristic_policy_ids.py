from __future__ import annotations

from pathlib import Path

from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
)

from .replay_inspector_test_support import _inspect_with_heuristic_public_policy


def test_inspect_replay_bundle_supports_all_heuristic_public_policy_ids(tmp_path: Path) -> None:
    heuristic_policy_ids = (
        HEURISTIC_PUBLIC_POLICY_ID,
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
        HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    )

    for policy_id in heuristic_policy_ids:
        report = _inspect_with_heuristic_public_policy(
            tmp_path,
            policy_b=policy_id,
            policy_a_logits={51: 1.0, 472: 0.5, 473: 0.2, 474: 0.0},
            top_actions=2,
        )

        assert report["policy_b"]["kind"] == "heuristic_public"
        assert report["policy_b"]["spec"] == policy_id
