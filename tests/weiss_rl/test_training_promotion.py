from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.training import promotion as promotion_module
from weiss_rl.training.promotion import (
    build_heuristic_public_policy,
    find_noleague_baseline_snapshot,
    promotion_anchor_policy_id_candidates,
    resolve_promotion_anchor_policy_ids,
    resolve_symbolic_promotion_anchor_policy_id,
    slug_policy_id,
    snapshot_meta_by_policy_id,
    trace_promotion_anchor_resolution,
)


def _registry_with_snapshots() -> SnapshotRegistry:
    registry = SnapshotRegistry(recent_size=10, champion_size=10)
    for update in (80, 120, 160):
        registry.add_snapshot(
            policy_id=f"policy_{update:06d}",
            update=update,
            weights_sha256=str(update % 10) * 64,
            path=f"training/snapshots/policy_{update:06d}/weights.pt",
        )
    registry.add_champion("policy_000080")
    registry.add_champion("policy_000120")
    return registry


def _write_snapshot_registry(run_dir, *, policy_ids: tuple[str, ...]) -> SnapshotRegistry:
    snapshots_dir = run_dir / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=10, champion_size=10)
    for index, policy_id in enumerate(policy_ids, start=1):
        registry.add_snapshot(
            policy_id=policy_id,
            update=index * 10,
            weights_sha256=str(index) * 64,
            path=f"training/snapshots/{policy_id}/weights.pt",
        )
    registry.save(snapshots_dir / "registry.json")
    return registry


def _write_manifest(run_dir, *, config_canonical) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": config_canonical}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_promotion_anchor_policy_id_candidates_preserve_legacy_aliases() -> None:
    assert slug_policy_id("Latest Champion Snapshot") == "latest_champion_snapshot"
    assert promotion_anchor_policy_id_candidates("B0 RandomLegal") == ("b0_randomlegal",)
    assert promotion_anchor_policy_id_candidates("B1 NoLeague baseline") == (
        "b1_noleague_baseline",
        "B1 NoLeague baseline",
    )
    assert promotion_anchor_policy_id_candidates("B2 HeuristicPublic") == ("B2 HeuristicPublic",)
    assert promotion_anchor_policy_id_candidates("Custom Anchor") == ("custom_anchor", "Custom Anchor")


def test_resolve_symbolic_promotion_anchor_policy_id_uses_registry_windows() -> None:
    registry = _registry_with_snapshots()

    assert resolve_symbolic_promotion_anchor_policy_id("Latest champion snapshot", registry=registry) == "policy_000120"
    assert (
        resolve_symbolic_promotion_anchor_policy_id("Previous champion snapshot", registry=registry) == "policy_000080"
    )
    assert resolve_symbolic_promotion_anchor_policy_id("Latest recent snapshot", registry=registry) == "policy_000160"
    assert resolve_symbolic_promotion_anchor_policy_id("Previous recent snapshot", registry=registry) == "policy_000120"
    assert resolve_symbolic_promotion_anchor_policy_id("Not symbolic", registry=registry) is None


def test_resolve_promotion_anchor_policy_ids_returns_missing_required_only() -> None:
    registry = _registry_with_snapshots()
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="5" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline", "Missing Required"),
                    optional_if_available=("B2 HeuristicPublic", "Latest champion snapshot", "Missing Optional"),
                )
            )
        )
    )

    resolved, missing = resolve_promotion_anchor_policy_ids(stack=stack, registry=registry)

    assert resolved == {
        "B0 RandomLegal": "b0_randomlegal",
        "B1 NoLeague baseline": "b1_noleague_baseline",
        "B2 HeuristicPublic": "B2 HeuristicPublic",
        "Latest champion snapshot": "policy_000120",
    }
    assert missing == ("Missing Required",)


def test_trace_promotion_anchor_resolution_explains_sources_and_missing_required() -> None:
    registry = _registry_with_snapshots()
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="5" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline", "Missing Required"),
                    optional_if_available=("B2 HeuristicPublic", "Latest champion snapshot", "Missing Optional"),
                )
            )
        )
    )

    trace = trace_promotion_anchor_resolution(stack=stack, registry=registry)
    payload = [row.as_payload() for row in trace]

    assert [(row.anchor_name, row.policy_id, row.source) for row in trace] == [
        ("B0 RandomLegal", "b0_randomlegal", "builtin_random_legal"),
        ("B1 NoLeague baseline", "b1_noleague_baseline", "registry_candidate"),
        ("Missing Required", None, "missing_required"),
        ("B2 HeuristicPublic", "B2 HeuristicPublic", "builtin_heuristic_public"),
        ("Latest champion snapshot", "policy_000120", "symbolic_registry"),
        ("Missing Optional", None, "missing_optional"),
    ]
    assert payload[2]["missing_required"] is True
    assert payload[5]["required"] is False


def test_find_noleague_baseline_snapshot_prefers_canonical_and_legacy_ids(tmp_path) -> None:
    canonical_run = tmp_path / "canonical"
    _write_snapshot_registry(canonical_run, policy_ids=("policy_000010", "b1_noleague_baseline"))

    canonical_snapshot = find_noleague_baseline_snapshot(canonical_run)

    assert canonical_snapshot is not None
    assert canonical_snapshot.policy_id == "b1_noleague_baseline"

    legacy_run = tmp_path / "legacy"
    _write_snapshot_registry(legacy_run, policy_ids=("policy_000010", "B1 NoLeague baseline"))

    legacy_snapshot = find_noleague_baseline_snapshot(legacy_run)

    assert legacy_snapshot is not None
    assert legacy_snapshot.policy_id == "B1 NoLeague baseline"


def test_find_noleague_baseline_snapshot_requires_explicit_alias(tmp_path) -> None:
    run_dir = tmp_path / "missing_alias"
    _write_snapshot_registry(run_dir, policy_ids=("policy_000010", "policy_000020", "policy_000030"))
    _write_manifest(run_dir, config_canonical={"config": {"experiment": {"role": "baseline_noleague"}}})

    assert find_noleague_baseline_snapshot(run_dir) is None


def test_find_noleague_baseline_snapshot_returns_none_without_registry_or_baseline_marker(tmp_path) -> None:
    assert find_noleague_baseline_snapshot(tmp_path / "missing") is None

    run_dir = tmp_path / "not_baseline"
    _write_snapshot_registry(run_dir, policy_ids=("policy_000010",))
    _write_manifest(run_dir, config_canonical={"config": {"experiment": {"role": "main"}}})

    assert find_noleague_baseline_snapshot(run_dir) is None


def test_build_heuristic_public_policy_passes_scoring_profile_when_supported(monkeypatch) -> None:
    calls: list[tuple[dict[str, object], str]] = []

    class _ProfileAwarePolicy:
        @classmethod
        def from_spec_bundle(cls, spec_bundle: dict[str, object], *, scoring_profile: str):
            calls.append((spec_bundle, scoring_profile))
            return ("policy", scoring_profile)

    monkeypatch.setattr(promotion_module, "HeuristicPublicPolicy", _ProfileAwarePolicy)

    policy = build_heuristic_public_policy({"spec": "bundle"}, scoring_profile="control")

    assert cast(Any, policy) == ("policy", "control")
    assert calls == [({"spec": "bundle"}, "control")]


def test_build_heuristic_public_policy_preserves_legacy_factory_fallback(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _LegacyPolicy:
        @classmethod
        def from_spec_bundle(cls, spec_bundle: dict[str, object]):
            calls.append(spec_bundle)
            return "legacy-policy"

    monkeypatch.setattr(promotion_module, "HeuristicPublicPolicy", _LegacyPolicy)

    policy = build_heuristic_public_policy({"spec": "bundle"}, scoring_profile="aggressive")

    assert cast(Any, policy) == "legacy-policy"
    assert calls == [{"spec": "bundle"}]


def test_snapshot_meta_by_policy_id_indexes_registry_snapshots() -> None:
    registry = _registry_with_snapshots()

    indexed = snapshot_meta_by_policy_id(registry)

    assert list(indexed) == ["policy_000080", "policy_000120", "policy_000160"]
    assert indexed["policy_000120"].update == 120
