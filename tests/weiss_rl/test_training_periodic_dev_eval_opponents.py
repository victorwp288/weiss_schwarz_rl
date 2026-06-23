from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import weiss_rl.training.train_entrypoint as train_script
from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.training.train_entrypoint import (
    _periodic_dev_eval_opponents,
)

from tests.weiss_rl._config_paths import repo_root


def test_periodic_dev_eval_opponents_include_optional_b2_when_available(monkeypatch, tmp_path: Path) -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    contract = SimpleNamespace(spec_bundle={"observation": {"kind": "stub"}, "action": {"kind": "stub"}})
    fake_model = object()
    fake_heuristic = object()
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=2, champion_size=1)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )
    registry.save(snapshots_dir / "registry.json")

    monkeypatch.setattr(train_script, "_load_snapshot_eval_model", lambda **kwargs: fake_model)

    class _FakeHeuristicPolicy:
        @staticmethod
        def from_spec_bundle(spec_bundle):
            return fake_heuristic

    monkeypatch.setattr(train_script, "HeuristicPublicPolicy", _FakeHeuristicPolicy)

    opponents = _periodic_dev_eval_opponents(
        stack=cast(Any, stack),
        contract=cast(Any, contract),
        run_dir=tmp_path,
        observation_dim=1,
        action_dim=2,
    )

    assert [item[0] for item in opponents] == [
        "b0_randomlegal",
        "b1_noleague_baseline",
        "B2 HeuristicPublic",
    ]
    assert opponents[1][2] is fake_model
    assert opponents[2][3] is fake_heuristic


def test_periodic_dev_eval_opponents_include_extra_snapshot_anchor_from_promotion_anchor_set(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline"),
                    optional_if_available=("policy_000123",),
                )
            )
        )
    )
    contract = SimpleNamespace(spec_bundle={"observation": {"kind": "stub"}, "action": {"kind": "stub"}})
    fake_model = object()
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=4, champion_size=1)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )
    registry.add_snapshot(
        policy_id="policy_000123",
        update=123,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000123"),
    )
    registry.save(snapshots_dir / "registry.json")

    monkeypatch.setattr(train_script, "_load_snapshot_eval_model", lambda **kwargs: fake_model)

    opponents = _periodic_dev_eval_opponents(
        stack=cast(Any, stack),
        contract=cast(Any, contract),
        run_dir=tmp_path,
        observation_dim=1,
        action_dim=2,
    )

    assert [item[1] for item in opponents] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "policy_000123",
    ]
    assert opponents[1][2] is fake_model
    assert opponents[2][0] == "policy_000123"
    assert opponents[2][2] is fake_model


def test_periodic_dev_eval_opponents_resolve_symbolic_snapshot_anchor_aliases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline"),
                    optional_if_available=("Previous champion snapshot", "Previous recent snapshot"),
                )
            )
        )
    )
    contract = SimpleNamespace(spec_bundle={"observation": {"kind": "stub"}, "action": {"kind": "stub"}})
    fake_model = object()
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=4, champion_size=2)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )
    registry.add_snapshot(
        policy_id="policy_000010",
        update=10,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000010"),
    )
    registry.add_snapshot(
        policy_id="policy_000020",
        update=20,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000020"),
    )
    registry.add_snapshot(
        policy_id="policy_000030",
        update=30,
        weights_sha256="d" * 64,
        path=snapshot_weights_relpath("policy_000030"),
    )
    registry.add_champion("policy_000010")
    registry.add_champion("policy_000020")
    registry.save(snapshots_dir / "registry.json")

    monkeypatch.setattr(train_script, "_load_snapshot_eval_model", lambda **kwargs: fake_model)

    opponents = _periodic_dev_eval_opponents(
        stack=cast(Any, stack),
        contract=cast(Any, contract),
        run_dir=tmp_path,
        observation_dim=1,
        action_dim=2,
    )

    assert [item[1] for item in opponents] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "Previous champion snapshot",
        "Previous recent snapshot",
    ]
    assert opponents[2][0] == "policy_000010"
    assert opponents[2][2] is fake_model
    assert opponents[3][0] == "policy_000020"
    assert opponents[3][2] is fake_model
