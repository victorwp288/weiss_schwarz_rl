from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np
import pytest
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import canonical_config_dict, load_stack_config
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.harness import ScheduledGame
from weiss_rl.eval.simulator_runner import ResolvedEvalPolicy, SimulatorEvalRunner
from weiss_rl.league import PromotionGatePosterior, PromotionGateRate, PromotionGateResult
from weiss_rl.league.registry import (
    ChampionDemotion,
    SnapshotReferenceNormalization,
    SnapshotRegistry,
    champion_demotion_newer_than,
    champion_demotion_stale_by_age,
    normalize_snapshot_references,
    snapshot_weights_relpath,
)
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.model import PolicyValueModel
from weiss_rl.tests._config_paths import canonical_stack_config_path
from weiss_rl.training.snapshots import demote_registry_champions_newer_than, seed_snapshot_policy_id

REPO_ROOT = Path(__file__).resolve().parents[3]


class _TrainingPathsLike(Protocol):
    snapshots_dir: Path


@lru_cache(maxsize=1)
def _load_train_script_module() -> ModuleType:
    python_root = str(REPO_ROOT / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)

    train_script_path = REPO_ROOT / "python" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_for_tests", train_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load train.py from {train_script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _retention_stack(*, recent_size: int, champion_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                snapshot_pool_recent_size=recent_size,
                snapshot_pool_champion_size=champion_size,
            )
        )
    )


def _make_policy_value_model(stack: Any) -> PolicyValueModel:
    assert stack.config.model is not None
    return PolicyValueModel(
        observation_dim=512,
        config=stack.config.model,
        action_dim=9,
        observation_spec=_heuristic_public_contract_bundle()["observation"],  # type: ignore[arg-type]
    )


def _canonical_config_with_role(stack: Any, *, experiment_role: str) -> dict[str, Any]:
    config_canonical = canonical_config_dict(stack)
    config_sections = cast(dict[str, Any], config_canonical.setdefault("config", {}))
    experiment = dict(cast(dict[str, Any], config_sections.get("experiment", {})))
    experiment["role"] = experiment_role
    config_sections["experiment"] = experiment
    return config_canonical


def test_seed_snapshot_policy_id_preserves_hash_input_and_sanitizes_policy_id() -> None:
    source_run_dir = Path("relative") / "seed_run"

    policy_id = seed_snapshot_policy_id(
        source_run_dir=source_run_dir,
        source_policy_id=" folder/policy\\000010 ",
    )

    assert policy_id == "seed_c3bd127559_folder_policy_000010"


def test_train_seed_snapshot_policy_id_wrapper_matches_training_helper() -> None:
    train_script = _load_train_script_module()
    source_run_dir = Path("relative") / "seed_run"

    assert train_script._seed_snapshot_policy_id(
        source_run_dir=source_run_dir,
        source_policy_id="policy/000020",
    ) == seed_snapshot_policy_id(source_run_dir=source_run_dir, source_policy_id="policy/000020")


def _legacy_config_with_training_mode(stack: Any, *, training_mode: str) -> dict[str, Any]:
    config_sections = dict(cast(dict[str, Any], canonical_config_dict(stack).get("config", {})))
    config_sections.pop("experiment", None)
    config_sections["training_family_a"] = {"mode": training_mode}
    return config_sections


def _write_b1_baseline_run_fixture(
    tmp_path: Path,
    *,
    update: int = 5,
    policy_id: str = "b1_noleague_baseline",
    config_hash256: str = "ab" * 32,
    spec_hash256: str = "cd" * 32,
    experiment_role: str = "baseline_noleague",
    legacy_training_mode: str | None = None,
    stack_path: Path | None = None,
) -> Path:
    train_script = _load_train_script_module()
    stack = load_stack_config(stack_path or canonical_stack_config_path())
    run_dir = tmp_path / "b1_run"
    training_paths = train_script._training_paths(run_dir)
    checkpoint_path = training_paths.checkpoints_dir / f"checkpoint_{update}.pt"
    torch.save({"format": "checkpoint_stub"}, checkpoint_path)

    weights_path, weights_sha256 = train_script._write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=policy_id,
        update=update,
        config_hash256=config_hash256,
        device=torch.device("cpu"),
        model_state_dict=_make_policy_value_model(stack).state_dict(),
        structured_policy_contract=None
        if stack.config.model is None
        else stack.config.model.structured_policy_contract,
    )
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    registry.add_snapshot(
        policy_id=policy_id,
        update=update,
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    registry.pin_snapshot(policy_id)
    registry.save(training_paths.snapshots_dir / "registry.json")
    (run_dir / "config_hash256.txt").write_text(f"{config_hash256}\n", encoding="utf-8")
    (run_dir / "spec_hash256.txt").write_text(f"{spec_hash256}\n", encoding="utf-8")
    config_canonical = (
        _legacy_config_with_training_mode(stack, training_mode=legacy_training_mode)
        if legacy_training_mode is not None
        else _canonical_config_with_role(stack, experiment_role=experiment_role)
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"config_canonical": config_canonical},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _mark_fixture_as_locked_selected_candidate(run_dir: Path, *, update: int) -> None:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry.load(registry_path)
    snapshot = next(entry for entry in registry.snapshots if entry.policy_id == "selected_candidate")
    metadata_path = run_dir / "training" / "snapshots" / "selected_candidate" / "policy_meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "format": "selected_candidate_alias_metadata_v1",
                "policy_id": "selected_candidate",
                "alias_for_policy_id": f"policy_{update // 5:06d}",
                "update": update,
                "weights_path": snapshot_weights_relpath("selected_candidate"),
                "weights_sha256": snapshot.weights_sha256,
                "source_weights_path": f"training/snapshots/policy_{update // 5:06d}/weights.pt",
                "selection_summary": {
                    "selected": {
                        "snapshot_policy_id": f"policy_{update // 5:06d}",
                        "update_count": update,
                        "selection_score_source": "targeted_confirm",
                    },
                    "required_anchors": [
                        "B2 HeuristicPublic",
                        "B3 HeuristicPublicAggro",
                        "B4 HeuristicPublicControl",
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "paper_readiness_summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "alarms": [],
                "checks": {
                    "baseline_win_rate_vs_b0": {
                        "passed": True,
                        "focal_policy_id": "selected_candidate",
                        "baseline_policy_id": "B0 RandomLegal",
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_seed_snapshot_run_fixture(
    tmp_path: Path,
    *,
    updates: tuple[int, ...] = (10, 20),
    champion_updates: tuple[int, ...] = (20,),
    pinned_policy_ids: tuple[str, ...] = (),
    config_hash256: str = "ab" * 32,
    spec_hash256: str = "cd" * 32,
    experiment_role: str = "main",
) -> Path:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "seed_run"
    training_paths = train_script._training_paths(run_dir)
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    for update in updates:
        policy_id = f"policy_{update:06d}"
        checkpoint_path = training_paths.checkpoints_dir / f"checkpoint_{update}.pt"
        torch.save({"format": "checkpoint_stub"}, checkpoint_path)
        weights_path, weights_sha256 = train_script._write_snapshot_artifact(
            snapshots_dir=training_paths.snapshots_dir,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            policy_id=policy_id,
            update=update,
            config_hash256=config_hash256,
            device=torch.device("cpu"),
            model_state_dict=_make_policy_value_model(stack).state_dict(),
        )
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
        )
    for update in champion_updates:
        registry.add_champion(f"policy_{update:06d}")
    for policy_id in pinned_policy_ids:
        registry.pin_snapshot(policy_id)
    registry.save(training_paths.snapshots_dir / "registry.json")
    (run_dir / "config_hash256.txt").write_text(f"{config_hash256}\n", encoding="utf-8")
    (run_dir / "spec_hash256.txt").write_text(f"{spec_hash256}\n", encoding="utf-8")
    config_canonical = _canonical_config_with_role(stack, experiment_role=experiment_role)
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": config_canonical}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _heuristic_public_contract_bundle() -> dict[str, object]:
    return {
        "observation": {
            "obs_len": 512,
            "self_first": True,
            "header_fields": [
                {"name": "active_player", "index": 0},
                {"name": "phase", "index": 1},
                {"name": "decision_kind", "index": 2},
                {"name": "decision_player", "index": 3},
                {"name": "terminal", "index": 4},
                {"name": "last_action_kind", "index": 5},
                {"name": "last_action_arg0", "index": 6},
                {"name": "last_action_arg1", "index": 7},
                {"name": "attack_slot", "index": 8},
                {"name": "defender_slot", "index": 9},
                {"name": "attack_type", "index": 10},
                {"name": "attack_damage", "index": 11},
                {"name": "attack_counter_power", "index": 12},
                {"name": "focus_slot", "index": 13},
                {"name": "choice_page_start", "index": 14},
                {"name": "choice_total", "index": 15},
            ],
            "player_blocks": [
                {
                    "player_index": 0,
                    "base": 16,
                    "len": 42,
                    "slices": [
                        {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                        {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                        {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                        {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                        {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                    ],
                },
                {
                    "player_index": 1,
                    "base": 58,
                    "len": 42,
                    "slices": [
                        {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                        {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                        {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                        {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                        {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                    ],
                },
            ],
        },
        "action": {
            "action_space_size": 9,
            "pass_action_id": 8,
            "attack_type_encoding": [["frontal", 0], ["side", 1], ["direct", 2]],
            "constants": [["MAX_HAND", 4], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
            "families": [
                {"name": "mulligan_confirm", "base": 0, "count": 1},
                {"name": "attack", "base": 1, "count": 3},
                {"name": "main_play_character", "base": 4, "count": 5},
                {"name": "pass", "base": 8, "count": 1},
            ],
        },
    }


def test_snapshot_registry_survives_restart_and_returns_latest_n(tmp_path: Path) -> None:
    registry_path = tmp_path / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry(recent_size=3, champion_size=1)

    registry.add_snapshot(
        policy_id="policy_000003",
        update=3,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000003"),
    )
    registry.add_snapshot(
        policy_id="policy_000001",
        update=1,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("policy_000001"),
    )
    registry.add_snapshot(
        policy_id="policy_000002",
        update=2,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000002"),
    )
    registry.save(registry_path)

    reloaded = SnapshotRegistry.load(registry_path)

    assert [snapshot.policy_id for snapshot in reloaded.snapshots] == [
        "policy_000001",
        "policy_000002",
        "policy_000003",
    ]
    assert reloaded.latest_ids(2) == ["policy_000002", "policy_000003"]


def test_snapshot_registry_prune_keeps_recent_champion_and_pinned_union() -> None:
    registry = SnapshotRegistry(recent_size=2, champion_size=1)
    for update in range(1, 6):
        policy_id = f"policy_{update:06d}"
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=(str(update) * 64)[:64],
            path=snapshot_weights_relpath(policy_id),
        )

    registry.pin_snapshot("policy_000002")
    registry.add_champion("policy_000003")
    registry.add_champion("policy_000004")

    pruned = registry.prune()

    assert [snapshot.policy_id for snapshot in registry.snapshots] == [
        "policy_000002",
        "policy_000004",
        "policy_000005",
    ]
    assert registry.champion_snapshots == ["policy_000004"]
    assert registry.pinned_snapshots == ["policy_000002"]
    assert [snapshot.policy_id for snapshot in pruned] == ["policy_000001", "policy_000003"]


def test_snapshot_registry_add_champion_dedupes_and_trims_window() -> None:
    registry = SnapshotRegistry(recent_size=0, champion_size=2)
    for update in range(1, 4):
        policy_id = f"policy_{update:06d}"
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=(str(update) * 64)[:64],
            path=snapshot_weights_relpath(policy_id),
        )

    registry.add_champion("policy_000001")
    registry.add_champion("policy_000002")
    registry.add_champion("policy_000001")
    registry.add_champion("policy_000003")

    assert registry.champion_snapshots == ["policy_000001", "policy_000003"]


def test_snapshot_reference_normalization_reports_dropped_and_trimmed_refs() -> None:
    result = normalize_snapshot_references(
        ["ghost", "policy_000001", "policy_000002", "policy_000001", "policy_000003"],
        existing_snapshot_ids={"policy_000001", "policy_000002", "policy_000003"},
        limit=2,
    )

    assert result == SnapshotReferenceNormalization(
        refs=["policy_000001", "policy_000003"],
        dropped_refs=["ghost"],
        trimmed_refs=["policy_000002"],
    )


def test_champion_demotion_helpers_preserve_order_and_report_remaining_refs() -> None:
    refs = ["policy_000080", "policy_000120", "policy_000160"]
    updates_by_policy = {
        "policy_000080": 80,
        "policy_000120": 120,
        "policy_000160": 160,
    }

    newer = champion_demotion_newer_than(
        refs,
        updates_by_policy=updates_by_policy,
        update=120,
    )
    stale = champion_demotion_stale_by_age(
        refs,
        updates_by_policy=updates_by_policy,
        current_update=180,
        max_age_updates=80,
    )

    assert newer == ChampionDemotion(
        removed_refs=["policy_000160"],
        remaining_refs=["policy_000080", "policy_000120"],
    )
    assert stale == ChampionDemotion(
        removed_refs=["policy_000080"],
        remaining_refs=["policy_000120", "policy_000160"],
    )


def test_snapshot_registry_add_champion_rejects_unknown_snapshot() -> None:
    registry = SnapshotRegistry()

    with pytest.raises(ValueError, match="existing snapshot"):
        registry.add_champion("policy_999999")


def test_snapshot_registry_load_normalizes_orphaned_refs(tmp_path: Path) -> None:
    registry_path = tmp_path / "training" / "snapshots" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 3,
                "champion_size": 2,
                "snapshots": [
                    {
                        "policy_id": "policy_000001",
                        "update": 1,
                        "weights_sha256": "a" * 64,
                        "path": snapshot_weights_relpath("policy_000001"),
                        "created_utc": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "policy_id": "policy_000002",
                        "update": 2,
                        "weights_sha256": "b" * 64,
                        "path": snapshot_weights_relpath("policy_000002"),
                        "created_utc": "2026-01-01T00:00:01+00:00",
                    },
                ],
                "champion_snapshots": ["ghost", "policy_000001", "policy_000001", "policy_000002"],
                "pinned_snapshots": ["ghost", "policy_000001", "policy_000001"],
            }
        ),
        encoding="utf-8",
    )

    registry = SnapshotRegistry.load(registry_path)

    assert registry.champion_snapshots == ["policy_000001", "policy_000002"]
    assert registry.pinned_snapshots == ["policy_000001"]


def test_snapshot_registry_rejects_checkpoint_paths() -> None:
    registry = SnapshotRegistry()

    try:
        registry.add_snapshot(
            policy_id="policy_000001",
            update=1,
            weights_sha256="a" * 64,
            path="training/checkpoints/checkpoint_1.pt",
        )
    except ValueError as exc:
        assert "training/snapshots" in str(exc)
    else:
        raise AssertionError("expected add_snapshot() to reject checkpoint paths")


def test_train_snapshot_persistence_writes_artifact_bundle_and_registry_entry(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = _retention_stack(recent_size=24, champion_size=4)
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_7.pt"
    torch.save({"format": "checkpoint_stub"}, checkpoint_path)

    model = torch.nn.Linear(3, 2)
    train_script._persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        model_state_dict=model.state_dict(),
        config_hash256="ab" * 32,
        device=torch.device("cpu"),
        update=7,
        policy_version=7,
    )

    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert len(registry.snapshots) == 1

    snapshot = registry.snapshots[0]
    expected_weights_relpath = snapshot_weights_relpath("policy_000007")
    weights_path = run_dir / expected_weights_relpath
    metadata_path = training_paths.snapshots_dir / "policy_000007" / "policy_meta.json"

    assert snapshot.policy_id == "policy_000007"
    assert snapshot.update == 7
    assert snapshot.path == expected_weights_relpath
    assert snapshot.path != checkpoint_path.relative_to(run_dir).as_posix()
    assert weights_path.is_file()
    assert snapshot.weights_sha256 == train_script._sha256_file(weights_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "format": "minimal_train_snapshot_metadata_v1",
        "policy_id": "policy_000007",
        "source_checkpoint_path": "training/checkpoints/checkpoint_7.pt",
        "update": 7,
        "weights_path": expected_weights_relpath,
        "weights_sha256": snapshot.weights_sha256,
    }

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert payload["format"] == "minimal_train_snapshot_weights_v1"
    assert payload["policy_id"] == "policy_000007"
    assert payload["update"] == 7
    assert payload["config_hash256"] == "ab" * 32
    assert payload["device"] == "cpu"
    assert set(payload["model_state_dict"]) == set(model.state_dict())


def test_train_snapshot_retention_prunes_old_snapshot_artifacts(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = _retention_stack(recent_size=1, champion_size=0)
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    model = torch.nn.Linear(3, 2)

    for policy_version in (1, 2, 3):
        checkpoint_path = training_paths.checkpoints_dir / f"checkpoint_{policy_version}.pt"
        torch.save({"format": "checkpoint_stub"}, checkpoint_path)
        train_script._persist_snapshot_registry_entry(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            model_state_dict=model.state_dict(),
            config_hash256="ab" * 32,
            device=torch.device("cpu"),
            update=policy_version,
            policy_version=policy_version,
        )

    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")

    assert [snapshot.policy_id for snapshot in registry.snapshots] == ["policy_000003"]
    assert not (training_paths.snapshots_dir / "policy_000001").exists()
    assert not (training_paths.snapshots_dir / "policy_000002").exists()
    assert (training_paths.snapshots_dir / "policy_000003").is_dir()


def test_ensure_noleague_baseline_anchor_imports_frozen_snapshot_once(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, update=5)

    run_dir = tmp_path / "consumer_run"
    training_paths = train_script._training_paths(run_dir)
    bootstrap_learner = SimpleNamespace(
        model=_make_policy_value_model(stack),
        update_count=0,
        optimizer=None,
        get_policy_version=lambda: 0,
    )

    policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=bootstrap_learner,
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=baseline_run_dir,
    )

    assert policy_id == "b1_noleague_baseline"
    registry_path = training_paths.snapshots_dir / "registry.json"
    registry = SnapshotRegistry.load(registry_path)
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [policy_id]
    assert registry.champion_snapshots == []
    assert registry.pinned_snapshots == [policy_id]

    snapshot = registry.snapshots[0]
    weights_path = run_dir / snapshot_weights_relpath(policy_id)
    metadata_path = training_paths.snapshots_dir / policy_id / "policy_meta.json"

    assert snapshot.update == 5
    assert weights_path.is_file()
    assert snapshot.weights_sha256 == train_script._sha256_file(weights_path)
    source_weights_sha256 = train_script._sha256_file(baseline_run_dir / snapshot_weights_relpath(policy_id))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "format": "imported_train_snapshot_metadata_v1",
        "imported_from_policy_id": policy_id,
        "imported_from_run_dir": baseline_run_dir.resolve().as_posix(),
        "imported_from_snapshot_path": snapshot_weights_relpath(policy_id),
        "imported_from_weights_sha256": source_weights_sha256,
        "policy_id": policy_id,
        "update": 5,
        "weights_path": snapshot_weights_relpath(policy_id),
        "weights_sha256": snapshot.weights_sha256,
    }

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert payload["policy_id"] == policy_id
    assert payload["update"] == 5
    assert payload["imported_from_run_dir"] == baseline_run_dir.resolve().as_posix()
    assert payload["imported_from_policy_id"] == policy_id
    assert payload["imported_from_snapshot_path"] == snapshot_weights_relpath(policy_id)
    assert payload["imported_from_weights_sha256"] == source_weights_sha256

    second_policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=bootstrap_learner,
        device=torch.device("cpu"),
        config_hash256="ff" * 32,
        baseline_run_dir=baseline_run_dir,
    )

    assert second_policy_id == policy_id
    reloaded = SnapshotRegistry.load(registry_path)
    assert [snapshot.policy_id for snapshot in reloaded.snapshots] == [policy_id]
    assert reloaded.champion_snapshots == []
    assert reloaded.pinned_snapshots == [policy_id]
    assert reloaded.snapshots[0].weights_sha256 == snapshot.weights_sha256


def test_ensure_noleague_baseline_anchor_imports_locked_selected_candidate(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    selected_run_dir = _write_b1_baseline_run_fixture(
        tmp_path,
        update=15,
        policy_id="selected_candidate",
        experiment_role="guided_league_bootstrap",
    )
    _mark_fixture_as_locked_selected_candidate(selected_run_dir, update=15)

    run_dir = tmp_path / "consumer_selected_run"
    training_paths = train_script._training_paths(run_dir)
    policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=SimpleNamespace(
            model=_make_policy_value_model(stack),
            update_count=0,
            optimizer=None,
            get_policy_version=lambda: 0,
        ),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=selected_run_dir,
    )

    assert policy_id == "b1_noleague_baseline"
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    snapshot = next(entry for entry in registry.snapshots if entry.policy_id == policy_id)
    assert snapshot.update == 15
    payload = torch.load(run_dir / snapshot_weights_relpath(policy_id), map_location="cpu", weights_only=True)
    assert payload["imported_from_policy_id"] == "selected_candidate"
    assert payload["imported_from_run_dir"] == selected_run_dir.resolve().as_posix()
    assert payload["imported_from_weights_sha256"] == train_script._sha256_file(
        selected_run_dir / snapshot_weights_relpath("selected_candidate")
    )


def test_ensure_noleague_baseline_anchor_rejects_unlocked_selected_candidate(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    selected_run_dir = _write_b1_baseline_run_fixture(
        tmp_path,
        update=15,
        policy_id="selected_candidate",
        experiment_role="guided_league_bootstrap",
    )

    with pytest.raises(FileNotFoundError, match="canonical B1 no-league baseline snapshot"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_unlocked_selected_run"),
            run_dir=tmp_path / "consumer_unlocked_selected_run",
            learner=SimpleNamespace(
                model=_make_policy_value_model(stack),
                update_count=0,
                optimizer=None,
                get_policy_version=lambda: 0,
            ),
            device=torch.device("cpu"),
            config_hash256="ab" * 32,
            baseline_run_dir=selected_run_dir,
        )


def test_ensure_noleague_baseline_anchor_imports_explicit_b1_run_when_required_by_main_gate(
    tmp_path: Path,
) -> None:
    train_script = _load_train_script_module()
    stack_path = (
        REPO_ROOT
        / "configs"
        / "thesis"
        / ("main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml")
    )
    stack = load_stack_config(stack_path)
    league_config = stack.config.league
    assert league_config is not None
    assert "B1 NoLeague baseline" in league_config.promotion_anchor_set_v1.required
    selected_run_dir = _write_b1_baseline_run_fixture(
        tmp_path,
        update=15,
        policy_id="selected_candidate",
        experiment_role="guided_league_bootstrap",
        stack_path=stack_path,
    )
    _mark_fixture_as_locked_selected_candidate(selected_run_dir, update=15)

    run_dir = tmp_path / "consumer_optional_b1_run"
    training_paths = train_script._training_paths(run_dir)
    policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=SimpleNamespace(
            model=_make_policy_value_model(stack),
            update_count=0,
            optimizer=None,
            get_policy_version=lambda: 0,
        ),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=selected_run_dir,
    )

    assert policy_id == "b1_noleague_baseline"
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert registry.pinned_snapshots == [policy_id]
    payload = torch.load(run_dir / snapshot_weights_relpath(policy_id), map_location="cpu", weights_only=True)
    assert payload["imported_from_policy_id"] == "selected_candidate"
    assert payload["imported_from_weights_sha256"] == train_script._sha256_file(
        selected_run_dir / snapshot_weights_relpath("selected_candidate")
    )


def test_ensure_noleague_baseline_anchor_refreshes_current_run_alias_to_latest_update(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "b1_run"
    training_paths = train_script._training_paths(run_dir)
    learner = SimpleNamespace(
        model=_make_policy_value_model(stack),
        update_count=1,
        optimizer=None,
        get_policy_version=lambda: 1,
    )

    first_policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=learner,
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        permit_current_run_alias=True,
        update=1,
    )

    first_registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    first_snapshot = next(snapshot for snapshot in first_registry.snapshots if snapshot.policy_id == first_policy_id)
    first_hash = first_snapshot.weights_sha256

    learner.update_count = 3
    second_policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=learner,
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        permit_current_run_alias=True,
        update=3,
    )

    assert second_policy_id == first_policy_id == "b1_noleague_baseline"
    second_registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    second_snapshot = next(snapshot for snapshot in second_registry.snapshots if snapshot.policy_id == second_policy_id)
    assert second_snapshot.update == 3
    assert second_snapshot.weights_sha256 != first_hash


def test_ensure_noleague_baseline_anchor_rejects_non_b1_imported_run(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, experiment_role="main")

    with pytest.raises(RuntimeError, match="must come from a dedicated baseline_noleague run"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run"),
            run_dir=tmp_path / "consumer_run",
            learner=SimpleNamespace(
                model=_make_policy_value_model(stack),
                update_count=0,
                optimizer=None,
                get_policy_version=lambda: 0,
            ),
            device=torch.device("cpu"),
            config_hash256="11" * 32,
            spec_hash256="cd" * 32,
            baseline_run_dir=baseline_run_dir,
        )


def test_ensure_noleague_baseline_anchor_rejects_legacy_non_b1_imported_run(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, legacy_training_mode="main")

    with pytest.raises(RuntimeError, match=r"training_family_a\.mode='main'"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_legacy"),
            run_dir=tmp_path / "consumer_run_legacy",
            learner=SimpleNamespace(
                model=_make_policy_value_model(stack),
                update_count=0,
                optimizer=None,
                get_policy_version=lambda: 0,
            ),
            device=torch.device("cpu"),
            config_hash256="11" * 32,
            spec_hash256="cd" * 32,
            baseline_run_dir=baseline_run_dir,
        )


def test_ensure_noleague_baseline_anchor_rejects_imported_environment_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path)
    manifest_path = baseline_run_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sections = manifest_payload["config_canonical"]["config"]
    config_sections["environment"] = {
        **dict(config_sections["environment"]),
        "best_of": 99,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="config does not match the current run for section='environment'"):
        train_script._ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_env_mismatch"),
            run_dir=tmp_path / "consumer_run_env_mismatch",
            learner=SimpleNamespace(
                model=_make_policy_value_model(stack),
                update_count=0,
                optimizer=None,
                get_policy_version=lambda: 0,
            ),
            device=torch.device("cpu"),
            config_hash256="11" * 32,
            spec_hash256="cd" * 32,
            baseline_run_dir=baseline_run_dir,
        )


def test_import_seed_snapshot_pool_imports_external_snapshots_and_champions(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path)
    consumer_run_dir = tmp_path / "consumer_run"
    training_paths = train_script._training_paths(consumer_run_dir)
    bootstrap_learner = SimpleNamespace(
        model=_make_policy_value_model(stack),
        update_count=0,
        optimizer=None,
        get_policy_version=lambda: 0,
    )

    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        expected_model_state_dict=bootstrap_learner.model.state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256="cd" * 32,
    )

    assert len(imported_policy_ids) == 2
    expected_policy_ids = [
        train_script._seed_snapshot_policy_id(source_run_dir=seed_run_dir.resolve(), source_policy_id="policy_000010"),
        train_script._seed_snapshot_policy_id(source_run_dir=seed_run_dir.resolve(), source_policy_id="policy_000020"),
    ]
    assert imported_policy_ids == expected_policy_ids

    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert [snapshot.policy_id for snapshot in registry.snapshots] == expected_policy_ids
    assert [snapshot.update for snapshot in registry.snapshots] == [0, 0]
    assert registry.champion_snapshots == [expected_policy_ids[-1]]

    weights_path = consumer_run_dir / snapshot_weights_relpath(expected_policy_ids[-1])
    metadata_path = training_paths.snapshots_dir / expected_policy_ids[-1] / "policy_meta.json"
    assert weights_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "seeded_train_snapshot_metadata_v1"
    assert metadata["policy_id"] == expected_policy_ids[-1]
    assert metadata["update"] == 0
    assert metadata["imported_from_update"] == 20
    assert metadata["imported_from_run_dir"] == seed_run_dir.resolve().as_posix()
    assert metadata["imported_from_policy_id"] == "policy_000020"


def test_import_seed_snapshot_pool_can_mark_all_seed_snapshots_as_training_champions(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.league is not None
    pool = replace(stack.config.league.pool, seed_snapshot_champion_import="all")
    league = replace(stack.config.league, pool=pool)
    stack = replace(stack, config=replace(stack.config, league=league))
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path, champion_updates=())
    consumer_run_dir = tmp_path / "consumer_run_seedchampions"
    training_paths = train_script._training_paths(consumer_run_dir)
    bootstrap_learner = SimpleNamespace(
        model=_make_policy_value_model(stack),
        update_count=0,
        optimizer=None,
        get_policy_version=lambda: 0,
    )

    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        expected_model_state_dict=bootstrap_learner.model.state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256="cd" * 32,
    )

    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert imported_policy_ids
    assert registry.champion_snapshots == imported_policy_ids


def test_import_seed_snapshot_pool_can_mark_pinned_seed_snapshots_as_training_champions(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.league is not None
    pool = replace(stack.config.league.pool, seed_snapshot_champion_import="pinned")
    league = replace(stack.config.league, pool=pool)
    stack = replace(stack, config=replace(stack.config, league=league))
    seed_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        updates=(10, 20, 30),
        champion_updates=(),
        pinned_policy_ids=("policy_000020",),
    )
    consumer_run_dir = tmp_path / "consumer_run_pinned_seedchampions"
    training_paths = train_script._training_paths(consumer_run_dir)
    bootstrap_learner = SimpleNamespace(
        model=_make_policy_value_model(stack),
        update_count=0,
        optimizer=None,
        get_policy_version=lambda: 0,
    )

    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        expected_model_state_dict=bootstrap_learner.model.state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256="cd" * 32,
    )

    expected_champion = train_script._seed_snapshot_policy_id(
        source_run_dir=seed_run_dir.resolve(),
        source_policy_id="policy_000020",
    )
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert len(imported_policy_ids) == 3
    assert registry.champion_snapshots == [expected_champion]


def test_import_seed_snapshot_pool_can_import_only_pinned_seed_snapshots(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.league is not None
    pool = replace(
        stack.config.league.pool,
        seed_snapshot_champion_import="pinned",
        seed_snapshot_import_filter="pinned",
    )
    league = replace(stack.config.league, pool=pool)
    stack = replace(stack, config=replace(stack.config, league=league))
    seed_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        updates=(10, 20, 30),
        champion_updates=(),
        pinned_policy_ids=("policy_000020",),
    )
    consumer_run_dir = tmp_path / "consumer_run_pinned_seed_filter"
    training_paths = train_script._training_paths(consumer_run_dir)

    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256="cd" * 32,
    )

    expected_policy_id = train_script._seed_snapshot_policy_id(
        source_run_dir=seed_run_dir.resolve(),
        source_policy_id="policy_000020",
    )
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert imported_policy_ids == [expected_policy_id]
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [expected_policy_id]
    assert registry.champion_snapshots == [expected_policy_id]


def test_import_seed_snapshot_pool_can_use_explicit_registry_json(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.league is not None
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path, updates=(10, 20, 30), champion_updates=(20,))
    source_registry = SnapshotRegistry.load(seed_run_dir / "training" / "snapshots" / "registry.json")
    source_registry.champion_snapshots = ["policy_000030"]
    explicit_registry = seed_run_dir / "training" / "snapshots" / "registry_explicit_champions.json"
    source_registry.save(explicit_registry)
    pool = replace(
        stack.config.league.pool,
        seed_snapshot_import_filter="source_champions",
        seed_snapshot_registry_json=explicit_registry.as_posix(),
    )
    league = replace(stack.config.league, pool=pool)
    stack = replace(stack, config=replace(stack.config, league=league))
    consumer_run_dir = tmp_path / "consumer_run_explicit_registry"
    training_paths = train_script._training_paths(consumer_run_dir)

    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256="cd" * 32,
    )

    expected_policy_id = train_script._seed_snapshot_policy_id(
        source_run_dir=seed_run_dir.resolve(),
        source_policy_id="policy_000030",
    )
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert imported_policy_ids == [expected_policy_id]
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [expected_policy_id]
    assert registry.champion_snapshots == [expected_policy_id]


def test_import_seed_snapshot_pool_accepts_guided_bootstrap_source_role(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        champion_updates=(),
        experiment_role="guided_league_bootstrap",
    )
    consumer_run_dir = tmp_path / "consumer_run_guided_bootstrap_seed"
    training_paths = train_script._training_paths(consumer_run_dir)

    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256="cd" * 32,
    )

    assert len(imported_policy_ids) == 2


def test_import_seed_snapshot_pool_rejects_environment_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path)
    manifest_path = seed_run_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sections = manifest_payload["config_canonical"]["config"]
    config_sections["environment"] = {
        **dict(config_sections["environment"]),
        "best_of": 99,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="config does not match the current run for section='environment'"):
        train_script._import_seed_snapshot_pool(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_seed_env_mismatch"),
            run_dir=tmp_path / "consumer_run_seed_env_mismatch",
            seed_snapshot_run_dir=seed_run_dir,
            expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256="cd" * 32,
        )


def test_import_seed_snapshot_pool_rejects_strict_b1_baseline_role(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path, experiment_role="baseline_noleague")

    with pytest.raises(RuntimeError, match="Use --b1-baseline-run-dir for the strict B1 baseline"):
        train_script._import_seed_snapshot_pool(
            stack=stack,
            training_paths=train_script._training_paths(tmp_path / "consumer_run_seed_role_mismatch"),
            run_dir=tmp_path / "consumer_run_seed_role_mismatch",
            seed_snapshot_run_dir=seed_run_dir,
            expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256="cd" * 32,
        )


def test_run_minimal_training_bootstraps_noleague_baseline_before_env_start(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())

    bootstrap_calls: list[dict[str, Any]] = []

    def fake_ensure_noleague_baseline_anchor(**kwargs):
        bootstrap_calls.append(kwargs)
        return "b1_noleague_baseline"

    def stop_before_runtime(*args, **kwargs):
        raise RuntimeError("stop after bootstrap")

    monkeypatch.setattr(train_script, "_ensure_noleague_baseline_anchor", fake_ensure_noleague_baseline_anchor)
    monkeypatch.setattr(train_script, "QueueRuntime", stop_before_runtime)

    run_dir = tmp_path / "run"
    try:
        train_script._run_minimal_training(
            stack=stack,
            contract=SimpleNamespace(spec_bundle=_heuristic_public_contract_bundle()),
            artifacts=SimpleNamespace(run_dir=run_dir),
            num_envs=1,
            unroll_length=1,
            max_updates=1,
            profile="fast",
            device=torch.device("cpu"),
            seed=7,
            checkpoint_interval_updates=1,
            run_id256="12" * 32,
            config_hash256="34" * 32,
            spec_hash256="56" * 32,
            runtime_mode="train_ordered",
            b1_baseline_run_dir=None,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after bootstrap"
    else:
        raise AssertionError("expected QueueRuntime to stop the test after baseline bootstrap")

    assert len(bootstrap_calls) == 1
    bootstrap_call = bootstrap_calls[0]
    assert bootstrap_call["run_dir"] == run_dir
    training_paths_arg = cast(_TrainingPathsLike, bootstrap_call["training_paths"])
    assert training_paths_arg.snapshots_dir == run_dir / "training" / "snapshots"
    assert bootstrap_call["device"] == torch.device("cpu")
    assert bootstrap_call["config_hash256"] == train_script.compute_config_hash256(stack)
    assert bootstrap_call["baseline_run_dir"] is None


def test_run_snapshot_promotion_gate_marks_passed_candidate_as_champion(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, update=5)

    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)

    train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=SimpleNamespace(
            model=_make_policy_value_model(stack),
            update_count=0,
            optimizer=None,
            get_policy_version=lambda: 0,
        ),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=baseline_run_dir,
    )
    registry_path = training_paths.snapshots_dir / "registry.json"

    learner_model = _make_policy_value_model(stack)
    candidate_checkpoint_path = training_paths.checkpoints_dir / "checkpoint_7.pt"
    torch.save({"format": "checkpoint_stub"}, candidate_checkpoint_path)
    candidate_policy_id = train_script._persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=candidate_checkpoint_path,
        model_state_dict=learner_model.state_dict(),
        config_hash256="cd" * 32,
        device=torch.device("cpu"),
        update=7,
        policy_version=7,
    )

    def fake_run_promotion_gate(**kwargs):
        assert kwargs["focal_policy_id"] == candidate_policy_id
        assert kwargs["anchor_policy_ids"] == {
            "B0 RandomLegal": "b0_randomlegal",
            "B1 NoLeague baseline": "b1_noleague_baseline",
            "B2 HeuristicPublic": "B2 HeuristicPublic",
        }
        return PromotionGateResult(
            focal_policy_id=candidate_policy_id,
            ordered_opponents=("B0 RandomLegal", "B1 NoLeague baseline", "B2 HeuristicPublic"),
            record_path="promotion_gate.json",
            seed_file_path="configs/seeds/promotion_eval_seeds.txt",
            seed_file_sha256="ef" * 32,
            paired_seed_count=1,
            weighting="uniform_across_anchors",
            seat_swap=True,
            folding="S0",
            anchors=(),
            overall_posterior=PromotionGatePosterior(
                mean=0.75,
                ci_low=0.7,
                ci_high=0.8,
                ci_half_width=0.05,
                prob_gt_half=1.0,
                prob_lt_half=0.0,
                prob_gt_target=1.0,
                prob_lt_guardrail=0.0,
                paired_seed_count=1,
                sample_count=64,
            ),
            truncation=PromotionGateRate(numerator=0, denominator=2, rate=0.0),
            passed=True,
            reasons=(),
        )

    monkeypatch.setattr(train_script, "run_promotion_gate", fake_run_promotion_gate)

    promoted = train_script._run_snapshot_promotion_gate(
        stack=stack,
        contract=SimpleNamespace(spec_bundle=_heuristic_public_contract_bundle()),
        artifacts=SimpleNamespace(run_dir=run_dir),
        training_paths=training_paths,
        learner=SimpleNamespace(model=learner_model),
        candidate_policy_id=candidate_policy_id,
        update_count=int(cast(Any, stack.config.league).warmup.first_updates),
        league_reference_update=int(cast(Any, stack.config.league).warmup.first_updates),
        policy_version=int(cast(Any, stack.config.league).warmup.first_updates),
        run_id256="12" * 32,
        config_hash256="34" * 32,
        spec_hash256="56" * 32,
    )

    assert promoted is True
    registry = SnapshotRegistry.load(registry_path)
    assert registry.champion_snapshots == [candidate_policy_id]


def test_run_snapshot_promotion_gate_skips_during_warmup(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, update=5)

    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)

    train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=SimpleNamespace(
            model=_make_policy_value_model(stack),
            update_count=0,
            optimizer=None,
            get_policy_version=lambda: 0,
        ),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=baseline_run_dir,
    )

    learner_model = _make_policy_value_model(stack)
    candidate_checkpoint_path = training_paths.checkpoints_dir / "checkpoint_7.pt"
    torch.save({"format": "checkpoint_stub"}, candidate_checkpoint_path)
    candidate_policy_id = train_script._persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=candidate_checkpoint_path,
        model_state_dict=learner_model.state_dict(),
        config_hash256="cd" * 32,
        device=torch.device("cpu"),
        update=7,
        policy_version=7,
    )

    def fail_run_promotion_gate(**kwargs):
        raise AssertionError("promotion gate should be skipped during warmup")

    monkeypatch.setattr(train_script, "run_promotion_gate", fail_run_promotion_gate)

    promoted = train_script._run_snapshot_promotion_gate(
        stack=stack,
        contract=SimpleNamespace(spec_bundle=_heuristic_public_contract_bundle()),
        artifacts=SimpleNamespace(run_dir=run_dir),
        training_paths=training_paths,
        learner=SimpleNamespace(model=learner_model),
        candidate_policy_id=candidate_policy_id,
        update_count=int(cast(Any, stack.config.league).warmup.first_updates) - 1,
        league_reference_update=int(cast(Any, stack.config.league).warmup.first_updates) - 1,
        policy_version=7,
        run_id256="12" * 32,
        config_hash256="34" * 32,
        spec_hash256="56" * 32,
    )

    assert promoted is None
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert registry.champion_snapshots == []


def test_run_snapshot_promotion_gate_uses_effective_update_for_warmup(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, update=5)

    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)

    train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=SimpleNamespace(
            model=_make_policy_value_model(stack),
            update_count=0,
            optimizer=None,
            get_policy_version=lambda: 0,
        ),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=baseline_run_dir,
    )

    learner_model = _make_policy_value_model(stack)
    candidate_checkpoint_path = training_paths.checkpoints_dir / "checkpoint_220.pt"
    torch.save({"format": "checkpoint_stub"}, candidate_checkpoint_path)
    candidate_policy_id = train_script._persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=candidate_checkpoint_path,
        model_state_dict=learner_model.state_dict(),
        config_hash256="cd" * 32,
        device=torch.device("cpu"),
        update=220,
        policy_version=220,
    )

    def fail_run_promotion_gate(**kwargs):
        raise AssertionError("promotion gate should be skipped while effective update is still in warmup")

    monkeypatch.setattr(train_script, "run_promotion_gate", fail_run_promotion_gate)

    promoted = train_script._run_snapshot_promotion_gate(
        stack=stack,
        contract=SimpleNamespace(spec_bundle=_heuristic_public_contract_bundle()),
        artifacts=SimpleNamespace(run_dir=run_dir),
        training_paths=training_paths,
        learner=SimpleNamespace(model=learner_model),
        candidate_policy_id=candidate_policy_id,
        update_count=int(cast(Any, stack.config.league).warmup.first_updates) + 20,
        league_reference_update=int(cast(Any, stack.config.league).warmup.first_updates) - 20,
        policy_version=220,
        run_id256="12" * 32,
        config_hash256="34" * 32,
        spec_hash256="56" * 32,
    )

    assert promoted is None


def test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    artifacts = train_script._run_artifacts_from_existing_run_dir(run_dir)
    alias_stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=stack.config.curriculum,
            evaluation=SimpleNamespace(periodic_dev_eval_interval_updates=25),
        )
    )

    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 3
    learner.policy_version = 2
    learner.total_samples_processed = 96
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_3.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )

    tracker = train_script._publish_checkpoint_aliases(
        stack=alias_stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.25},
    )
    assert training_paths.latest_checkpoint_path.is_file()
    assert not training_paths.best_checkpoint_path.is_file()
    assert tracker["latest"]["metric_kind"] is None
    assert tracker["best"] is None

    learner.update_count = 4
    learner.policy_version = 3
    learner.total_samples_processed = 128
    second_checkpoint_path = training_paths.checkpoints_dir / "checkpoint_4.pt"
    train_script._write_checkpoint(
        checkpoint_path=second_checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    tracker = train_script._publish_checkpoint_aliases(
        stack=alias_stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=second_checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.5},
        dev_eval_summary={"aggregate_score": 0.61, "uncertainty": {"mean": 0.61}},
    )
    assert tracker["best"]["metric_kind"] == "dev_eval_mean"
    assert tracker["best"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_4.pt")

    restored_learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    resume_state = train_script._restore_learner_from_checkpoint(
        checkpoint_path=training_paths.best_checkpoint_path,
        learner=restored_learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    assert resume_state.update_count == 4
    assert resume_state.policy_version == 3
    assert resume_state.total_samples_processed == 128
    assert restored_learner.update_count == 4
    assert restored_learner.policy_version == 3

    restored_learner.update_count = 99
    restored_learner.policy_version = 77
    restored_learner.total_samples_processed = 12345
    preserved_start_time = restored_learner.start_time
    preserved_resume_state = train_script._restore_learner_from_checkpoint(
        checkpoint_path=training_paths.best_checkpoint_path,
        learner=restored_learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        restore_counters=False,
    )
    assert preserved_resume_state.update_count == 99
    assert preserved_resume_state.policy_version == 77
    assert preserved_resume_state.total_samples_processed == 12345
    assert restored_learner.update_count == 99
    assert restored_learner.policy_version == 77
    assert restored_learner.start_time == preserved_start_time


def test_write_checkpoint_payload_shape_is_stable(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    training_paths = train_script._training_paths(tmp_path / "run")
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 9
    learner.policy_version = 4
    learner.total_samples_processed = 288
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_9.pt"

    payload = train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )

    expected_keys = {
        "algorithm",
        "config_hash256",
        "device",
        "format",
        "grad_scaler_state_dict",
        "init_schedule_offset_updates",
        "model_state_dict",
        "optimizer_state_dict",
        "policy_anchor_model_state_dict",
        "policy_version",
        "public_heuristic_actor_logit_bias_scale",
        "public_heuristic_logit_bias_scale",
        "recurrent_core",
        "spec_hash256",
        "total_samples_processed",
        "update_count",
    }
    assert set(payload) == expected_keys
    assert payload["format"] == "minimal_train_checkpoint_v1"
    assert payload["update_count"] == 9
    assert payload["policy_version"] == 4
    assert payload["total_samples_processed"] == 288
    assert payload["device"] == "cpu"
    assert payload["spec_hash256"] == "ab" * 32
    assert payload["algorithm"] == "impala_vtrace_gru"
    assert isinstance(payload["model_state_dict"], dict)
    assert isinstance(payload["optimizer_state_dict"], dict)
    assert checkpoint_path.is_file()


def _write_restore_checkpoint_fixture(
    tmp_path: Path,
) -> tuple[ModuleType, Any, Path, ImpalaLearner]:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    training_paths = train_script._training_paths(tmp_path / "run")

    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_bad.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    restore_learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    return train_script, stack, checkpoint_path, restore_learner


@pytest.mark.parametrize(
    ("case_name", "match"),
    [
        ("non_dict_payload", "checkpoint payload must be a dict"),
        ("unsupported_format", "unsupported checkpoint format"),
        ("config_hash_mismatch", "checkpoint config hash mismatch"),
        ("spec_hash_mismatch", "checkpoint spec hash mismatch"),
        ("algorithm_mismatch", "checkpoint algorithm mismatch"),
        ("missing_model_state_dict", "checkpoint is missing a model_state_dict"),
    ],
)
def test_restore_checkpoint_rejects_invalid_payload_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    match: str,
) -> None:
    train_script, stack, checkpoint_path, restore_learner = _write_restore_checkpoint_fixture(tmp_path)
    monkeypatch.delenv("WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH", raising=False)

    if case_name == "non_dict_payload":
        torch.save(["not", "a", "dict"], checkpoint_path)
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert isinstance(payload, dict)
        if case_name == "unsupported_format":
            payload["format"] = "future_checkpoint_v999"
        elif case_name == "config_hash_mismatch":
            payload["config_hash256"] = "00" * 32
        elif case_name == "spec_hash_mismatch":
            payload["spec_hash256"] = "cd" * 32
        elif case_name == "algorithm_mismatch":
            payload["algorithm"] = "different_algorithm"
        elif case_name == "missing_model_state_dict":
            payload.pop("model_state_dict", None)
        else:
            raise AssertionError(f"unhandled case: {case_name}")
        torch.save(payload, checkpoint_path)

    with pytest.raises(RuntimeError, match=match):
        train_script._restore_learner_from_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=restore_learner,
            stack=stack,
            device=torch.device("cpu"),
            expected_spec_hash256="ab" * 32,
            algorithm="impala_vtrace_gru",
        )


def test_restore_checkpoint_allows_config_hash_mismatch_only_with_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_script, stack, checkpoint_path, restore_learner = _write_restore_checkpoint_fixture(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    payload["config_hash256"] = "00" * 32
    torch.save(payload, checkpoint_path)

    monkeypatch.setenv("WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH", "1")
    resume_state = train_script._restore_learner_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restore_learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )

    assert resume_state.checkpoint_path == checkpoint_path.resolve()


def test_rollback_to_best_checkpoint_preserves_latest_alias_and_logs_event(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "presets" / "typed_local.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = SimpleNamespace(run_dir=run_dir)
    training_paths = train_script._training_paths(run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()

    learner.update_count = 160
    learner.policy_version = 8
    learner.total_samples_processed = 5120
    best_checkpoint = training_paths.checkpoints_dir / "checkpoint_160.pt"
    train_script._write_checkpoint(
        checkpoint_path=best_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=best_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.5},
        dev_eval_summary={"aggregate_score": 0.65625, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )

    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000160",
        update=160,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000160/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000220",
        update=220,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000220/weights.pt",
    )
    registry.add_champion("policy_000160")
    registry.add_champion("policy_000220")
    registry.save(training_paths.snapshots_dir / "registry.json")

    learner.update_count = 220
    learner.policy_version = 11
    learner.total_samples_processed = 7040
    current_checkpoint = training_paths.checkpoints_dir / "checkpoint_220.pt"
    train_script._write_checkpoint(
        checkpoint_path=current_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=current_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.48, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )
    latest_before_rollback = train_script._load_checkpoint_tracker(training_paths)["latest"]

    runtime = SimpleNamespace(reset_count=0, refresh_count=0, publish_calls=[])
    runtime.reset_outcome_tracker = lambda: setattr(runtime, "reset_count", runtime.reset_count + 1)
    runtime.refresh_opponent_pool = lambda: setattr(runtime, "refresh_count", runtime.refresh_count + 1)

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        runtime.publish_calls.append(kwargs)
        return {"snapshot_publish_latency_ms": 1.25, "snapshot_apply_latency_ms": 2.5}

    runtime.maybe_publish_snapshot = _publish_snapshot

    event = train_script._maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        model=cast(PolicyValueModel, learner.model),
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.48, "stall_monitor": {"worst_truncation_rate": 0.0}},
        last_rollback_update=None,
    )

    assert event is not None
    assert event["action"] == "rollback_to_best"
    assert event["reasons"] == ["score_drop"]
    assert event["best_update_count"] == 160
    assert event["demoted_champions"] == ["policy_000220"]
    assert runtime.reset_count == 1
    assert runtime.refresh_count == 1
    assert len(runtime.publish_calls) == 1
    tracker = train_script._load_checkpoint_tracker(training_paths)
    assert tracker["latest"] == latest_before_rollback
    assert tracker["latest"]["metric_kind"] == "dev_eval_mean"
    assert tracker["latest"]["metric_value"] == pytest.approx(0.48)
    assert tracker["latest"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_220.pt")
    assert tracker["latest"]["update_count"] == 220
    assert tracker["latest"]["policy_version"] == 11
    assert training_paths.latest_checkpoint_path.read_bytes() != training_paths.best_checkpoint_path.read_bytes()
    assert training_paths.latest_checkpoint_path.read_bytes() == current_checkpoint.read_bytes()
    reloaded = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert reloaded.champion_snapshots == ["policy_000160"]
    log_event = json.loads((training_paths.logs_dir / "checkpoint_guard.jsonl").read_text(encoding="utf-8"))
    assert log_event["action"] == "rollback_to_best"
    assert log_event["rolled_back_checkpoint_path"].endswith("training/checkpoints/best.pt")
    assert log_event["demoted_champions"] == ["policy_000220"]


def test_finalize_from_best_checkpoint_keeps_latest_alias_chronological(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "presets" / "typed_local.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = SimpleNamespace(run_dir=run_dir)
    training_paths = train_script._training_paths(run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()

    learner.update_count = 160
    learner.policy_version = 8
    learner.total_samples_processed = 5120
    best_checkpoint = training_paths.checkpoints_dir / "checkpoint_160.pt"
    train_script._write_checkpoint(
        checkpoint_path=best_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=best_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.5},
        dev_eval_summary={"aggregate_score": 0.65625, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )

    learner.update_count = 220
    learner.policy_version = 11
    learner.total_samples_processed = 7040
    current_checkpoint = training_paths.checkpoints_dir / "checkpoint_220.pt"
    train_script._write_checkpoint(
        checkpoint_path=current_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    tracker = train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=current_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.28125, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )
    assert tracker["best"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_160.pt")

    runtime = SimpleNamespace(
        reset_count=0,
        refresh_count=0,
    )
    runtime.reset_outcome_tracker = lambda: setattr(runtime, "reset_count", runtime.reset_count + 1)
    runtime.refresh_opponent_pool = lambda: setattr(runtime, "refresh_count", runtime.refresh_count + 1)

    event = train_script._maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.28125, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )

    assert event is not None
    assert runtime.reset_count == 1
    assert runtime.refresh_count == 1
    tracker = train_script._load_checkpoint_tracker(training_paths)
    assert tracker["latest"]["metric_kind"] == "dev_eval_mean"
    assert tracker["latest"]["metric_value"] == pytest.approx(0.28125)
    assert tracker["latest"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_220.pt")
    assert tracker["latest"]["update_count"] == 220
    assert tracker["latest"]["policy_version"] == 11
    assert training_paths.latest_checkpoint_path.read_bytes() == current_checkpoint.read_bytes()
    assert training_paths.latest_checkpoint_path.read_bytes() != training_paths.best_checkpoint_path.read_bytes()


def test_demote_registry_champions_newer_than_removes_newer_refs_only(tmp_path: Path) -> None:
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000080",
        update=80,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000080/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000120",
        update=120,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000120/weights.pt",
    )
    registry.add_champion("policy_000080")
    registry.add_champion("policy_000120")

    removed = registry.demote_champions_newer_than(80)

    assert removed == ["policy_000120"]
    assert registry.champion_snapshots == ["policy_000080"]


def test_demote_registry_champions_newer_than_updates_registry_file(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True)
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000080",
        update=80,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000080/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000120",
        update=120,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000120/weights.pt",
    )
    registry.add_champion("policy_000080")
    registry.add_champion("policy_000120")
    registry.save(snapshots_dir / "registry.json")

    removed = demote_registry_champions_newer_than(
        SimpleNamespace(snapshots_dir=snapshots_dir),
        update_count=80,
    )

    assert removed == ["policy_000120"]
    reloaded = SnapshotRegistry.load(snapshots_dir / "registry.json")
    assert reloaded.champion_snapshots == ["policy_000080"]


def test_demote_stale_champions_removes_old_refs_only() -> None:
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000080",
        update=80,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000080/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000180",
        update=180,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000180/weights.pt",
    )
    registry.add_champion("policy_000080")
    registry.add_champion("policy_000180")

    removed = registry.demote_stale_champions(current_update=220, max_age_updates=60)

    assert removed == ["policy_000080"]
    assert registry.champion_snapshots == ["policy_000180"]


def test_resolve_resume_checkpoint_path_defaults_to_latest_alias(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    from weiss_rl.training import checkpoint_resolution
    from weiss_rl.training.checkpoints import resolve_resume_checkpoint_path

    run_dir = tmp_path / "resume_run"
    latest_path = run_dir / "training" / "checkpoints" / "latest.pt"
    best_path = run_dir / "training" / "checkpoints" / "best.pt"
    observed_best_path = run_dir / "training" / "checkpoints" / "observed_best.pt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_bytes(b"checkpoint")
    best_path.write_bytes(b"best")
    observed_best_path.write_bytes(b"observed")
    explicit_checkpoint_path = tmp_path / "manual.pt"
    explicit_checkpoint_path.write_bytes(b"manual")

    resolved = train_script._resolve_resume_checkpoint_path(
        resume_from="",
        resume_run_dir=run_dir,
    )

    assert resolve_resume_checkpoint_path is checkpoint_resolution.resolve_resume_checkpoint_path
    assert resolved == latest_path.resolve()
    assert resolve_resume_checkpoint_path(resume_from="", resume_run_dir=run_dir) == latest_path.resolve()
    assert resolve_resume_checkpoint_path(resume_from=" BEST ", resume_run_dir=run_dir) == best_path.resolve()
    assert (
        train_script._resolve_resume_checkpoint_path(
            resume_from="observed_best",
            resume_run_dir=run_dir,
        )
        == observed_best_path.resolve()
    )
    assert (
        resolve_resume_checkpoint_path(resume_from="observed_best", resume_run_dir=run_dir)
        == observed_best_path.resolve()
    )
    assert (
        resolve_resume_checkpoint_path(resume_from=str(explicit_checkpoint_path), resume_run_dir=None)
        == explicit_checkpoint_path.resolve()
    )
    with pytest.raises(ValueError, match="requires --resume-run-dir"):
        resolve_resume_checkpoint_path(resume_from="latest", resume_run_dir=None)


def test_periodic_dev_eval_runner_resets_env_with_scheduled_episode_seed(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()

    terminal_batch = train_script.DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        to_play=np.array([-1], dtype=np.int32),
        actor=np.array([-1], dtype=np.int32),
        decision_id=np.array([0], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.int32)),
    )

    class FakeEnv:
        def __init__(self, batch: object) -> None:
            self._batch = batch
            self.reset_seed: int | None = None
            self.closed = False

        def reset(self, seed: int | None = None):
            self.reset_seed = seed
            return self._batch

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 1), device=device)

    env = FakeEnv(terminal_batch)
    monkeypatch.setattr(train_script, "_build_ids_eval_env", lambda *args, **kwargs: env)

    runner = train_script._PeriodicDevEvalRunner(
        stack=SimpleNamespace(),
        model=FakeModel(),
        opponent_policy_id="baseline",
        observation_dim=1,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        focal_policy_id="focal",
        require_sorted_legal_ids=False,
    )
    scheduled_game = train_script.ScheduledGame(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=579856027068064,
        focal_policy_id="focal",
        opponent_policy_id="baseline",
        seat0_policy_id="focal",
        seat1_policy_id="baseline",
        focal_seat=0,
    )

    result = runner.run_game(scheduled_game)

    assert env.reset_seed == scheduled_game.episode_seed
    assert env.closed is True
    assert result.episode_seed == scheduled_game.episode_seed


def test_periodic_dev_eval_runner_uses_learner_scoring_mode(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()

    live_batch = train_script.DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        to_play=np.array([0], dtype=np.int32),
        actor=np.array([0], dtype=np.int32),
        decision_id=np.array([0], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.int32)),
    )
    terminal_batch = train_script.DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        to_play=np.array([-1], dtype=np.int32),
        actor=np.array([-1], dtype=np.int32),
        decision_id=np.array([1], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([1], dtype=np.uint32),
        tick_count=np.array([1], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.int32)),
    )

    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False
            self.reset_seed: int | None = None

        def reset(self, seed: int | None = None):
            self.reset_seed = seed
            return live_batch

        def step(self, actions: np.ndarray):
            return terminal_batch

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        def __init__(self) -> None:
            self.scoring_modes: list[str] = []

        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 1), device=device)

        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            seat_hidden_state: torch.Tensor | None = None,
            *,
            scoring_mode: str = "auto",
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.scoring_modes.append(str(scoring_mode))
            logits = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
            values = torch.zeros((1,), dtype=torch.float32, device=obs.device)
            next_hidden = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
            return logits, values, next_hidden

    env = FakeEnv()
    model = FakeModel()
    monkeypatch.setattr(train_script, "_build_ids_eval_env", lambda *args, **kwargs: env)

    runner = train_script._PeriodicDevEvalRunner(
        stack=SimpleNamespace(),
        model=model,
        opponent_policy_id="baseline",
        observation_dim=1,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        focal_policy_id="focal",
        require_sorted_legal_ids=False,
    )
    scheduled_game = train_script.ScheduledGame(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=579856027068064,
        focal_policy_id="focal",
        opponent_policy_id="baseline",
        seat0_policy_id="focal",
        seat1_policy_id="baseline",
        focal_seat=0,
    )

    result = runner.run_game(scheduled_game)

    assert model.scoring_modes == ["learner"]
    assert result.total_actions == 1
    assert result.pass_actions == 1
    assert result.pass_with_nonpass_available == 0
    assert env.closed is True


def test_promotion_gate_runner_resets_env_with_scheduled_episode_seed(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()

    terminal_batch = train_script.DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        to_play=np.array([-1], dtype=np.int32),
        actor=np.array([-1], dtype=np.int32),
        decision_id=np.array([0], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.int32)),
    )

    class FakeEnv:
        def __init__(self, batch: object) -> None:
            self._batch = batch
            self.reset_seed: int | None = None
            self.closed = False

        def reset(self, seed: int | None = None):
            self.reset_seed = seed
            return self._batch

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 1), device=device)

    env = FakeEnv(terminal_batch)
    monkeypatch.setattr(train_script, "_build_ids_eval_env", lambda *args, **kwargs: env)

    runner = train_script._PromotionGateRunner(
        stack=SimpleNamespace(),
        focal_policy_id="candidate",
        focal_model=FakeModel(),
        anchor_models={},
        heuristic_policies={},
        observation_dim=1,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        require_sorted_legal_ids=False,
    )
    scheduled_game = train_script.ScheduledGame(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=579856027068064,
        focal_policy_id="candidate",
        opponent_policy_id="baseline",
        seat0_policy_id="candidate",
        seat1_policy_id="baseline",
        focal_seat=0,
    )

    result = runner.run_game(scheduled_game)

    assert env.reset_seed == scheduled_game.episode_seed
    assert env.closed is True
    assert result.episode_seed == scheduled_game.episode_seed


def test_promotion_gate_runner_uses_learner_scoring_mode(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()

    live_batch = train_script.DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        to_play=np.array([0], dtype=np.int32),
        actor=np.array([0], dtype=np.int32),
        decision_id=np.array([0], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.int32)),
    )
    terminal_batch = train_script.DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        to_play=np.array([-1], dtype=np.int32),
        actor=np.array([-1], dtype=np.int32),
        decision_id=np.array([1], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([1], dtype=np.uint32),
        tick_count=np.array([1], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.int32)),
    )

    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False

        def reset(self, seed: int | None = None):
            return live_batch

        def step(self, actions: np.ndarray):
            return terminal_batch

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        def __init__(self) -> None:
            self.scoring_modes: list[str] = []

        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 1), device=device)

        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            seat_hidden_state: torch.Tensor | None = None,
            *,
            scoring_mode: str = "auto",
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.scoring_modes.append(str(scoring_mode))
            logits = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
            values = torch.zeros((1,), dtype=torch.float32, device=obs.device)
            next_hidden = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
            return logits, values, next_hidden

    env = FakeEnv()
    focal_model = FakeModel()
    monkeypatch.setattr(train_script, "_build_ids_eval_env", lambda *args, **kwargs: env)

    runner = train_script._PromotionGateRunner(
        stack=SimpleNamespace(),
        focal_policy_id="candidate",
        focal_model=focal_model,
        anchor_models={},
        heuristic_policies={},
        observation_dim=1,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        require_sorted_legal_ids=False,
    )
    scheduled_game = train_script.ScheduledGame(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=579856027068064,
        focal_policy_id="candidate",
        opponent_policy_id="baseline",
        seat0_policy_id="candidate",
        seat1_policy_id="baseline",
        focal_seat=0,
    )

    runner.run_game(scheduled_game)

    assert focal_model.scoring_modes == ["learner"]
    assert env.closed is True


def test_simulator_eval_runner_uses_learner_scoring_mode(tmp_path: Path, monkeypatch) -> None:
    live_batch = DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        to_play=np.array([0], dtype=np.int32),
        actor=np.array([0], dtype=np.int32),
        decision_id=np.array([0], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.int32)),
    )
    terminal_batch = DecisionBoundaryBatch(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        to_play=np.array([-1], dtype=np.int32),
        actor=np.array([-1], dtype=np.int32),
        decision_id=np.array([1], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([1], dtype=np.uint32),
        tick_count=np.array([1], dtype=np.uint32),
        episode_seed=np.array([579856027068064], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.int32)),
    )

    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False
            self.reset_seed: int | None = None

        def reset(self, seed: int | None = None):
            self.reset_seed = seed
            return live_batch

        def step(self, actions: np.ndarray):
            return terminal_batch

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        def __init__(self) -> None:
            self.scoring_modes: list[str] = []

        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 1), device=device)

        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            seat_hidden_state: torch.Tensor | None = None,
            *,
            scoring_mode: str = "auto",
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.scoring_modes.append(str(scoring_mode))
            logits = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
            values = torch.zeros((1,), dtype=torch.float32, device=obs.device)
            next_hidden = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
            return logits, values, next_hidden

    env = FakeEnv()
    model = FakeModel()
    layout = ArtifactLayout.from_run_dir(tmp_path)
    layout.ensure_directories()
    runner = SimulatorEvalRunner(
        stack=cast(Any, SimpleNamespace(config=SimpleNamespace(curriculum=None))),
        policies={
            "candidate": ResolvedEvalPolicy(
                policy_id="candidate",
                kind="snapshot_registry",
                model=cast(Any, model),
            )
        },
        artifact_layout=layout,
        run_id256="ab" * 32,
        spec_hash256="cd" * 32,
        action_dim=1,
        pass_action_id=0,
        require_sorted_legal_ids=False,
        replay_capture_rate=0.0,
        regression_capture_count=0,
    )
    monkeypatch.setattr(runner, "_build_ids_eval_env", lambda *, seed, scheduled_game=None: env)

    scheduled_game = ScheduledGame(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=579856027068064,
        focal_policy_id="candidate",
        opponent_policy_id="baseline",
        seat0_policy_id="candidate",
        seat1_policy_id="candidate",
        focal_seat=0,
    )

    result = runner.run_game(scheduled_game)

    assert result.episode_seed == scheduled_game.episode_seed
    assert env.reset_seed == scheduled_game.episode_seed
    assert model.scoring_modes == ["learner"]
    assert env.closed is True
