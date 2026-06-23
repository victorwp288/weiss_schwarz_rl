from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.eval.policies import resolution_context as policy_resolution_context_module
from weiss_rl.eval.policies.set import NO_LEAGUE_POLICY_ID, RANDOM_LEGAL_POLICY_ID
from weiss_rl.eval.simulator_runner import resolve_eval_policies

from ._config_paths import canonical_stack_config_path
from .eval_policy_resolution_test_support import (
    copy_policy_weights_to_cache,
    copy_snapshot_registry_to_cache,
    write_consumer_run_manifest,
    write_snapshot_registry_run,
)
from .heuristic_public_test_support import _heuristic_spec_bundle


def test_resolve_eval_policies_loads_b1_from_registry_source_run(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    source_run_dir = tmp_path / "external_runs" / "source_run"
    write_snapshot_registry_run(
        stack=stack,
        run_dir=source_run_dir,
        snapshots=[("b1_noleague_baseline", 5), ("policy_000100", 100)],
    )
    copied_registry_path = copy_snapshot_registry_to_cache(
        source_run_dir / "training" / "snapshots" / "registry.json",
        tmp_path,
    )

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[NO_LEAGUE_POLICY_ID, "policy_000100"],
        run_dir=write_consumer_run_manifest(tmp_path, "consumer_run"),
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved[NO_LEAGUE_POLICY_ID].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved[NO_LEAGUE_POLICY_ID].snapshot_path == "training/snapshots/b1_noleague_baseline/weights.pt"
    assert resolved[NO_LEAGUE_POLICY_ID].model is not None


def test_resolve_eval_policies_preserves_b1_display_id_preference_when_both_aliases_exist(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    b1_run_dir = tmp_path / "baselines" / "b1_run_with_aliases"
    write_snapshot_registry_run(
        stack=stack,
        run_dir=b1_run_dir,
        snapshots=[(NO_LEAGUE_POLICY_ID, 1), ("b1_noleague_baseline", 5)],
    )

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[NO_LEAGUE_POLICY_ID],
        run_dir=write_consumer_run_manifest(tmp_path, "consumer_run_b1_alias_preference"),
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        b1_baseline_run_dir=b1_run_dir,
    )

    assert resolved[NO_LEAGUE_POLICY_ID].snapshot_path == "training/snapshots/B1 NoLeague baseline/weights.pt"


def test_resolve_eval_policies_requires_b1_snapshot_for_mixed_copied_registry_requests(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    source_run_dir = tmp_path / "external_runs" / "source_run"
    weights_by_policy = write_snapshot_registry_run(
        stack=stack,
        run_dir=source_run_dir,
        snapshots=[("b1_noleague_baseline", 5), ("policy_000100", 100)],
    )
    copied_registry_path = copy_snapshot_registry_to_cache(
        source_run_dir / "training" / "snapshots" / "registry.json",
        tmp_path,
    )
    copy_policy_weights_to_cache(
        source_weights=weights_by_policy["policy_000100"],
        tmp_path=tmp_path,
        policy_id="policy_000100",
    )

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[NO_LEAGUE_POLICY_ID, "policy_000100"],
        run_dir=write_consumer_run_manifest(tmp_path, "consumer_run_mixed_b1"),
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved[NO_LEAGUE_POLICY_ID].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved[NO_LEAGUE_POLICY_ID].snapshot_path == "training/snapshots/b1_noleague_baseline/weights.pt"
    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()


def test_resolve_eval_policies_skips_registry_resolution_for_explicit_b1_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    source_run_dir = tmp_path / "external_runs" / "source_run"
    write_snapshot_registry_run(stack=stack, run_dir=source_run_dir, snapshots=[("policy_000100", 100)])
    copied_registry_path = copy_snapshot_registry_to_cache(
        source_run_dir / "training" / "snapshots" / "registry.json",
        tmp_path,
    )
    b1_run_dir = tmp_path / "baselines" / "b1_run"
    write_snapshot_registry_run(stack=stack, run_dir=b1_run_dir, snapshots=[("b1_noleague_baseline", 5)])

    def _unexpected_registry_resolution(**kwargs):
        raise AssertionError("registry source resolution should not be attempted")

    monkeypatch.setattr(
        policy_resolution_context_module.SnapshotRegistrySource,
        "resolve_run_dir",
        _unexpected_registry_resolution,
    )

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[RANDOM_LEGAL_POLICY_ID, NO_LEAGUE_POLICY_ID],
        run_dir=write_consumer_run_manifest(tmp_path, "consumer_run"),
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
        b1_baseline_run_dir=b1_run_dir,
    )

    assert resolved[RANDOM_LEGAL_POLICY_ID].kind == "random_legal"
    assert resolved[NO_LEAGUE_POLICY_ID].kind == "baseline_noleague"
    assert resolved[NO_LEAGUE_POLICY_ID].source_run_dir == b1_run_dir.resolve().as_posix()
