from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.eval.policies.set import NO_LEAGUE_POLICY_ID
from weiss_rl.eval.simulator.simulator_runner import resolve_eval_policies

from ._config_paths import canonical_stack_config_path
from .eval_policy_resolution_test_support import (
    write_consumer_run_manifest,
    write_legacy_b1_manifest_marker,
    write_nested_b1_manifest_marker,
    write_snapshot_registry_run,
)
from .heuristic_public_test_support import _heuristic_spec_bundle


def test_resolve_eval_policies_refuses_nested_manifest_latest_only_b1_snapshot(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    b1_run_dir = tmp_path / "baselines" / "b1_run"
    write_snapshot_registry_run(stack=stack, run_dir=b1_run_dir, snapshots=[("policy_000005", 5)])
    write_nested_b1_manifest_marker(stack=stack, run_dir=b1_run_dir)

    with pytest.raises(FileNotFoundError, match="mandatory B1 NoLeague baseline"):
        resolve_eval_policies(
            stack=stack,
            policy_ids=[NO_LEAGUE_POLICY_ID],
            run_dir=write_consumer_run_manifest(tmp_path, "consumer_run"),
            observation_dim=512,
            action_dim=9,
            spec_bundle=_heuristic_spec_bundle(),
            b1_baseline_run_dir=b1_run_dir,
        )


def test_resolve_eval_policies_refuses_legacy_manifest_latest_only_b1_snapshot(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    b1_run_dir = tmp_path / "baselines" / "b1_run_legacy"
    write_snapshot_registry_run(stack=stack, run_dir=b1_run_dir, snapshots=[("policy_000005", 5)])
    write_legacy_b1_manifest_marker(stack=stack, run_dir=b1_run_dir)

    with pytest.raises(FileNotFoundError, match="mandatory B1 NoLeague baseline"):
        resolve_eval_policies(
            stack=stack,
            policy_ids=[NO_LEAGUE_POLICY_ID],
            run_dir=write_consumer_run_manifest(tmp_path, "consumer_run_legacy"),
            observation_dim=512,
            action_dim=9,
            spec_bundle=_heuristic_spec_bundle(),
            b1_baseline_run_dir=b1_run_dir,
        )
