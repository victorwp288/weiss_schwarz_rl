from __future__ import annotations

from pathlib import Path

from weiss_rl.config import load_stack_config
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)
from weiss_rl.eval.simulator_runner import resolve_eval_policies

from ._config_paths import canonical_stack_config_path
from .heuristic_public_test_support import _heuristic_spec_bundle


def test_resolve_eval_policies_supports_b2_without_snapshot_weights(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[RANDOM_LEGAL_POLICY_ID, HEURISTIC_PUBLIC_POLICY_ID],
        run_dir=tmp_path,
        observation_dim=100,
        action_dim=527,
        spec_bundle=_heuristic_spec_bundle(),
    )

    assert resolved[RANDOM_LEGAL_POLICY_ID].kind == "random_legal"
    assert resolved[HEURISTIC_PUBLIC_POLICY_ID].kind == "heuristic_public"
    assert resolved[HEURISTIC_PUBLIC_POLICY_ID].heuristic_policy is not None


def test_resolve_eval_policies_supports_heuristic_public_variants(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[HEURISTIC_PUBLIC_AGGRO_POLICY_ID, HEURISTIC_PUBLIC_CONTROL_POLICY_ID],
        run_dir=tmp_path,
        observation_dim=100,
        action_dim=527,
        spec_bundle=_heuristic_spec_bundle(),
    )

    assert resolved[HEURISTIC_PUBLIC_AGGRO_POLICY_ID].kind == "heuristic_public"
    assert resolved[HEURISTIC_PUBLIC_AGGRO_POLICY_ID].heuristic_policy is not None
    assert resolved[HEURISTIC_PUBLIC_CONTROL_POLICY_ID].kind == "heuristic_public"
    assert resolved[HEURISTIC_PUBLIC_CONTROL_POLICY_ID].heuristic_policy is not None
