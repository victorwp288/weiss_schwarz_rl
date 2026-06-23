from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest
from weiss_rl.config import StackConfig
from weiss_rl.eval.b1_policy_resolution import resolve_b1_policy as b1_policy_resolution_resolve_b1_policy
from weiss_rl.eval.policies import resolution_context as policy_resolution_context_module
from weiss_rl.eval.policies.resolution import (
    _is_recursive_registry_search_root as policy_resolution_is_recursive_registry_search_root,
)
from weiss_rl.eval.policies.resolution import _resolve_b1_policy as policy_resolution_resolve_b1_policy
from weiss_rl.eval.policies.resolution import (
    _resolve_snapshot_registry_policy as policy_resolution_resolve_snapshot_registry_policy,
)
from weiss_rl.eval.policies.resolution import (
    _resolve_static_eval_policy as policy_resolution_resolve_static_eval_policy,
)
from weiss_rl.eval.policies.resolution import (
    _should_include_common_search_root as policy_resolution_should_include_common_search_root,
)
from weiss_rl.eval.policies.resolution import resolve_eval_policies as policy_resolution_resolve_eval_policies
from weiss_rl.eval.policies.resolution_context import EvalPolicyResolutionContext
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner
from weiss_rl.eval.snapshot_policy_resolution import (
    resolve_snapshot_registry_policy as snapshot_policy_resolution_resolve_snapshot_registry_policy,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    is_recursive_registry_search_root as snapshot_registry_is_recursive_registry_search_root,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    should_include_common_search_root as snapshot_registry_should_include_common_search_root,
)
from weiss_rl.eval.static_policy_resolution import (
    resolve_static_eval_policy as static_policy_resolve_static_eval_policy,
)
from weiss_rl.league.registry import SnapshotRegistry


def test_simulator_runner_exposes_only_runner_and_policy_resolver_boundary() -> None:
    import weiss_rl.eval.simulator_runner as simulator_runner

    retired_helper_exports = {
        "_candidate_b1_run_dirs",
        "_common_search_root",
        "_config_marks_noleague_baseline",
        "_find_b1_snapshot",
        "_is_recursive_registry_search_root",
        "_load_snapshot_eval_model",
        "_observation_spec_from_bundle",
        "_resolve_b1_policy",
        "_resolve_snapshot_registry_policy",
        "_resolve_snapshot_registry_run_dir",
        "_resolve_static_eval_policy",
        "_sha256_file",
        "_should_include_common_search_root",
        "_snapshot_by_policy_id_or_imported_seed_suffix",
        "_unique_paths",
    }

    assert simulator_runner.__all__ == [
        "ResolvedEvalPolicy",
        "SimulatorEvalRunner",
        "resolve_eval_policies",
    ]
    assert simulator_runner.SimulatorEvalRunner is SimulatorEvalRunner
    assert simulator_runner.resolve_eval_policies is policy_resolution_resolve_eval_policies
    assert not any(hasattr(simulator_runner, name) for name in retired_helper_exports)
    assert policy_resolution_is_recursive_registry_search_root is snapshot_registry_is_recursive_registry_search_root
    assert policy_resolution_should_include_common_search_root is snapshot_registry_should_include_common_search_root
    assert policy_resolution_resolve_b1_policy is b1_policy_resolution_resolve_b1_policy
    assert (
        policy_resolution_resolve_snapshot_registry_policy
        is snapshot_policy_resolution_resolve_snapshot_registry_policy
    )
    assert policy_resolution_resolve_static_eval_policy is static_policy_resolve_static_eval_policy
    assert policy_resolution_resolve_eval_policies.__module__ == "weiss_rl.eval.policies.resolution"
    assert policy_resolution_resolve_b1_policy.__module__ == "weiss_rl.eval.b1_policy_resolution"
    assert policy_resolution_resolve_snapshot_registry_policy.__module__ == "weiss_rl.eval.snapshot_policy_resolution"
    assert policy_resolution_resolve_static_eval_policy.__module__ == "weiss_rl.eval.static_policy_resolution"
    assert (
        policy_resolution_is_recursive_registry_search_root.__module__ == "weiss_rl.eval.snapshot_registry_resolution"
    )


def test_eval_root_does_not_export_policy_module_aliases() -> None:
    import weiss_rl.eval as eval_package

    assert not hasattr(eval_package, "policy_alignment")
    assert not hasattr(eval_package, "policy_resolution")
    assert not hasattr(eval_package, "policy_resolution_context")
    assert not hasattr(eval_package, "policy_set")
    assert not hasattr(eval_package, "policy_types")


def test_eval_policy_resolution_context_owns_registry_policy_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_run_dir = tmp_path / "runs" / "source_run"
    registry_path = registry_run_dir / "training" / "snapshots" / "registry.json"
    weights_path = registry_run_dir / "training" / "snapshots" / "policy_000100" / "weights.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000100",
        update=100,
        weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        path="training/snapshots/policy_000100/weights.pt",
    )
    registry.save(registry_path)
    consumer_run_dir = tmp_path / "runs" / "consumer"
    consumer_run_dir.mkdir(parents=True)
    context = EvalPolicyResolutionContext.load(
        run_dir=consumer_run_dir,
        policy_ids=["policy_000100"],
        snapshot_registry_path=registry_path,
    )
    observed: dict[str, object] = {}

    def _fake_resolve_snapshot_registry_policy(**kwargs: object):
        observed.update(kwargs)
        return kwargs["policy_id"]

    monkeypatch.setattr(
        policy_resolution_context_module,
        "resolve_snapshot_registry_policy",
        _fake_resolve_snapshot_registry_policy,
    )

    resolved = context.resolve_registry_policy(
        policy_id="policy_000100",
        stack=cast(StackConfig, object()),
        observation_dim=512,
        action_dim=9,
        spec_bundle={"observation": {}},
    )

    assert resolved == "policy_000100"
    assert observed["registry_source"] is context.registry_source
    assert context.registry_path == registry_path
    assert context.registry is context.registry_source.registry
    assert context.snapshots_by_policy_id is context.registry_source.snapshots_by_policy_id
    assert observed["snapshot_run_dir"] == registry_run_dir.resolve()
    assert observed["observation_dim"] == 512
    assert observed["action_dim"] == 9
    assert context.registry_run_dir_if_resolved == registry_run_dir.resolve()
