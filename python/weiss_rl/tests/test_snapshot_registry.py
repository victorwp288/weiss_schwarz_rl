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
from weiss_rl.config import canonical_config_dict, compute_config_hash256, load_stack_config
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.harness import ScheduledGame
from weiss_rl.eval.simulator_runner import ResolvedEvalPolicy, SimulatorEvalRunner
from weiss_rl.league import PromotionGatePosterior, PromotionGateRate, PromotionGateResult
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.model import PolicyValueModel
from weiss_rl.tests._config_paths import canonical_stack_config_path

REPO_ROOT = Path(__file__).resolve().parents[3]
NOLEAGUE_BASELINE_STACK_CONFIG = REPO_ROOT / "configs" / "baselines" / "noleague_impala.yaml"


def _fake_terminal_eval_batch(*, seed: int, terminated: bool = False, reward: float = 0.0) -> DecisionBoundaryBatch:
    return DecisionBoundaryBatch(
        obs=np.zeros((1, 4), dtype=np.float32),
        reward=np.asarray([reward], dtype=np.float32),
        terminated=np.asarray([terminated], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        to_play=np.asarray([0], dtype=np.int64),
        actor=np.asarray([0], dtype=np.int64),
        decision_id=np.asarray([1 if not terminated else 2], dtype=np.int64),
        engine_status=np.asarray([0], dtype=np.int64),
        decision_count=np.asarray([1 if not terminated else 2], dtype=np.int64),
        tick_count=np.asarray([1 if not terminated else 2], dtype=np.int64),
        episode_seed=np.asarray([seed], dtype=np.uint64),
        episode_key=np.asarray([seed], dtype=np.uint64),
        action_space=1,
        ids_offsets=(np.asarray([0], dtype=np.uint32), np.asarray([0, 1], dtype=np.int64)),
    )


class _FakeOneStepEvalEnv:
    max_decisions = 64
    max_ticks = 128
    max_no_progress_decisions = None

    def __init__(self) -> None:
        self.reset_seeds: list[int] = []
        self.close_count = 0

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        resolved_seed = 0 if seed is None else int(seed)
        self.reset_seeds.append(resolved_seed)
        return _fake_terminal_eval_batch(seed=resolved_seed, terminated=False)

    def step(self, _actions: object) -> DecisionBoundaryBatch:
        seed = self.reset_seeds[-1]
        return _fake_terminal_eval_batch(seed=seed, terminated=True)

    def close(self) -> None:
        self.close_count += 1


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


def _stack_with_periodic_dev_eval_interval(stack: Any, *, interval_updates: int) -> Any:
    assert stack.config.evaluation is not None
    evaluation = replace(stack.config.evaluation, periodic_dev_eval_interval_updates=int(interval_updates))
    config = replace(stack.config, evaluation=evaluation)
    return replace(stack, config=config)


def _canonical_config_with_role(stack: Any, *, experiment_role: str) -> dict[str, Any]:
    config_canonical = canonical_config_dict(stack)
    config_sections = cast(dict[str, Any], config_canonical.setdefault("config", {}))
    experiment = dict(cast(dict[str, Any], config_sections.get("experiment", {})))
    experiment["role"] = experiment_role
    config_sections["experiment"] = experiment
    return config_canonical


def _legacy_config_with_training_mode(stack: Any, *, training_mode: str) -> dict[str, Any]:
    config_sections = dict(cast(dict[str, Any], canonical_config_dict(stack).get("config", {})))
    config_sections.pop("experiment", None)
    config_sections["training_family_a"] = {"mode": training_mode}
    return config_sections


def _write_b1_baseline_run_fixture(
    tmp_path: Path,
    *,
    stack: Any | None = None,
    update: int = 5,
    policy_id: str = "b1_noleague_baseline",
    config_hash256: str = "ab" * 32,
    spec_hash256: str = "cd" * 32,
    experiment_role: str = "baseline_noleague",
    legacy_training_mode: str | None = None,
) -> Path:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path()) if stack is None else stack
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


def _write_seed_snapshot_run_fixture(
    tmp_path: Path,
    *,
    updates: tuple[int, ...] = (10, 20),
    champion_updates: tuple[int, ...] = (20,),
    config_hash256: str = "ab" * 32,
    spec_hash256: str = "cd" * 32,
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
    registry.save(training_paths.snapshots_dir / "registry.json")
    (run_dir / "config_hash256.txt").write_text(f"{config_hash256}\n", encoding="utf-8")
    (run_dir / "spec_hash256.txt").write_text(f"{spec_hash256}\n", encoding="utf-8")
    config_canonical = _canonical_config_with_role(stack, experiment_role="main")
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


def test_snapshot_registry_add_champion_rejects_unknown_snapshot() -> None:
    registry = SnapshotRegistry()

    with pytest.raises(ValueError, match="existing snapshot"):
        registry.add_champion("policy_999999")


def test_snapshot_registry_tracks_rejections_and_clears_on_champion(tmp_path: Path) -> None:
    registry_path = tmp_path / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry(recent_size=4, champion_size=2)
    registry.add_snapshot(
        policy_id="policy_000001",
        update=1,
        weights_sha256="1" * 64,
        path=snapshot_weights_relpath("policy_000001"),
    )
    registry.reject_snapshot("policy_000001")
    registry.save(registry_path)

    reloaded = SnapshotRegistry.load(registry_path)
    assert reloaded.rejected_snapshots == ["policy_000001"]

    reloaded.add_champion("policy_000001")
    assert reloaded.champion_snapshots == ["policy_000001"]
    assert reloaded.rejected_snapshots == []


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

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "format": "imported_train_snapshot_metadata_v1",
        "imported_from_policy_id": policy_id,
        "imported_from_run_dir": baseline_run_dir.resolve().as_posix(),
        "imported_from_snapshot_path": snapshot_weights_relpath(policy_id),
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


def test_ensure_noleague_baseline_anchor_imports_for_reference_bc_without_promotion(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(
        REPO_ROOT / "configs" / "archive" / "presets" / "pass3_b1_s1_retrain_from_u450_rawprotect.yaml"
    )
    assert stack.config.league is not None
    assert not stack.config.league.promotion.enabled
    assert stack.config.training is not None
    assert stack.config.training.reference_policy_top_action_bc_coef == pytest.approx(0.0)
    assert stack.config.training.reference_policy_top_action_family_bc_coef > 0.0
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack=stack, update=5)

    run_dir = tmp_path / "consumer_run"
    training_paths = train_script._training_paths(run_dir)
    bootstrap_learner = SimpleNamespace(
        model=_make_policy_value_model(stack),
        update_count=450,
        optimizer=None,
        get_policy_version=lambda: 9,
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
    weights_path = training_paths.snapshots_dir / policy_id / "weights.pt"
    assert weights_path.is_file()
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [policy_id]
    assert registry.pinned_snapshots == [policy_id]


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


def test_import_seed_snapshot_pool_imports_external_snapshots_as_seed_history_not_champions(tmp_path: Path) -> None:
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
    assert {snapshot.source_kind for snapshot in registry.snapshots} == {"seed_import"}
    assert registry.champion_snapshots == []
    assert registry.latest_seed_history_ids(4) == expected_policy_ids
    assert registry.latest_active_champion_ids(4) == []

    weights_path = consumer_run_dir / snapshot_weights_relpath(expected_policy_ids[-1])
    metadata_path = training_paths.snapshots_dir / expected_policy_ids[-1] / "policy_meta.json"
    assert weights_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "seeded_train_snapshot_metadata_v1"
    assert metadata["policy_id"] == expected_policy_ids[-1]
    assert metadata["imported_from_run_dir"] == seed_run_dir.resolve().as_posix()
    assert metadata["imported_from_policy_id"] == "policy_000020"
    assert metadata["source_was_champion"] is True


def test_import_seed_snapshot_pool_respects_max_update_for_resume_continuation(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        updates=(10, 20, 30),
        champion_updates=(20, 30),
    )
    consumer_run_dir = tmp_path / "consumer_run"

    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=train_script._training_paths(consumer_run_dir),
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        max_update=20,
        expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256="cd" * 32,
    )

    assert imported_policy_ids == [
        train_script._seed_snapshot_policy_id(source_run_dir=seed_run_dir.resolve(), source_policy_id="policy_000010"),
        train_script._seed_snapshot_policy_id(source_run_dir=seed_run_dir.resolve(), source_policy_id="policy_000020"),
    ]
    registry = SnapshotRegistry.load(consumer_run_dir / "training" / "snapshots" / "registry.json")
    assert [snapshot.update for snapshot in registry.snapshots] == [10, 20]
    assert registry.champion_snapshots == []
    assert registry.latest_seed_history_ids(4) == imported_policy_ids


def test_import_resume_league_snapshot_pool_preserves_local_recents_and_champions(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    source_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        updates=(10, 20, 30),
        champion_updates=(20,),
    )
    source_checkpoint_path = source_run_dir / "training" / "checkpoints" / "checkpoint_20.pt"
    consumer_run_dir = tmp_path / "consumer_run"

    imported_policy_ids = train_script._import_resume_league_snapshot_pool(
        stack=stack,
        training_paths=train_script._training_paths(consumer_run_dir),
        run_dir=consumer_run_dir,
        resume_checkpoint_path=source_checkpoint_path,
        max_update=20,
        expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
    )

    assert imported_policy_ids == ["policy_000010", "policy_000020"]
    registry = SnapshotRegistry.load(consumer_run_dir / "training" / "snapshots" / "registry.json")
    assert [snapshot.policy_id for snapshot in registry.snapshots] == ["policy_000010", "policy_000020"]
    assert {snapshot.source_kind for snapshot in registry.snapshots} == {"league_import"}
    assert registry.champion_snapshots == ["policy_000020"]

    metadata_path = consumer_run_dir / "training" / "snapshots" / "policy_000020" / "policy_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "resume_league_snapshot_metadata_v1"
    assert metadata["imported_from_run_dir"] == source_run_dir.resolve().as_posix()
    assert metadata["imported_from_policy_id"] == "policy_000020"


def test_snapshot_registry_typed_selectors_keep_seed_history_out_of_active_champions() -> None:
    registry = SnapshotRegistry(recent_size=8, champion_size=4)
    registry.add_snapshot(
        policy_id="seed_source_policy_000450",
        update=450,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("seed_source_policy_000450"),
        source_kind="seed_import",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=451,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
        source_kind="baseline_anchor",
    )
    registry.add_snapshot(
        policy_id="policy_000480",
        update=480,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000480"),
        source_kind="league_import",
    )
    registry.add_snapshot(
        policy_id="policy_000500",
        update=500,
        weights_sha256="d" * 64,
        path=snapshot_weights_relpath("policy_000500"),
    )
    registry.add_champion("seed_source_policy_000450")
    registry.add_champion("b1_noleague_baseline")
    registry.add_champion("policy_000480")
    registry.add_champion("policy_000500")
    registry.reject_snapshot("policy_000500")

    assert registry.champion_snapshots == ["policy_000480"]
    assert registry.latest_seed_history_ids(2) == ["seed_source_policy_000450"]
    assert registry.latest_local_candidate_ids(4) == ["policy_000480"]
    assert registry.latest_active_champion_ids(4) == ["policy_000480"]


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


def test_infer_seed_snapshot_run_dir_from_direct_resume_checkpoint(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path)
    checkpoint_path = seed_run_dir / "training" / "checkpoints" / "checkpoint_20.pt"

    inferred = train_script._infer_seed_snapshot_run_dir_from_resume_checkpoint(
        stack=stack,
        resume_checkpoint_path=checkpoint_path,
        resume_run_dir=None,
    )

    assert inferred == seed_run_dir.resolve()


def test_infer_seed_snapshot_run_dir_skips_in_place_resume(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path)
    checkpoint_path = seed_run_dir / "training" / "checkpoints" / "checkpoint_20.pt"

    inferred = train_script._infer_seed_snapshot_run_dir_from_resume_checkpoint(
        stack=stack,
        resume_checkpoint_path=checkpoint_path,
        resume_run_dir=seed_run_dir,
    )

    assert inferred is None


def test_seed_snapshot_import_max_update_only_limits_auto_inferred_resume_pool(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    resume_state = train_script.ResumeCheckpoint(
        checkpoint_path=tmp_path / "checkpoint_20.pt",
        update_count=20,
        policy_version=2,
        total_samples_processed=1024,
    )

    auto_max_update = train_script._seed_snapshot_import_max_update(
        resume_state=resume_state,
        seed_snapshot_run_dir=tmp_path / "same_run",
        seed_snapshot_run_dir_auto_inferred=True,
    )
    explicit_max_update = train_script._seed_snapshot_import_max_update(
        resume_state=resume_state,
        seed_snapshot_run_dir=tmp_path / "external_pool",
        seed_snapshot_run_dir_auto_inferred=False,
    )

    assert auto_max_update == 20
    assert explicit_max_update is None


def test_run_minimal_training_barriers_after_seed_snapshot_import(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    calls: list[str] = []

    monkeypatch.setattr(train_script, "_ensure_noleague_baseline_anchor", lambda **_kwargs: "b1_noleague_baseline")
    monkeypatch.setattr(train_script, "_attach_reference_policy_model_if_configured", lambda **_kwargs: None)

    def fake_import_seed_snapshot_pool(**_kwargs):
        assert _kwargs["max_update"] is None
        calls.append("import")
        return ["seed_policy_000020"]

    def fake_barrier(_context):
        calls.append("barrier")

    def stop_before_runtime(*_args, **_kwargs):
        raise RuntimeError("stop after seed barrier")

    monkeypatch.setattr(train_script, "_import_seed_snapshot_pool", fake_import_seed_snapshot_pool)
    monkeypatch.setattr(train_script, "distributed_barrier", fake_barrier)
    monkeypatch.setattr(train_script, "QueueRuntime", stop_before_runtime)

    with pytest.raises(RuntimeError, match="stop after seed barrier"):
        train_script._run_minimal_training(
            stack=stack,
            contract=SimpleNamespace(spec_bundle=_heuristic_public_contract_bundle()),
            artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
            num_envs=2,
            unroll_length=1,
            max_updates=1,
            max_wall_clock_minutes=None,
            profile="fast",
            device=torch.device("cpu"),
            seed=7,
            checkpoint_interval_updates=1,
            run_id256="11" * 32,
            config_hash256="22" * 32,
            spec_hash256="cd" * 32,
            runtime_mode="train_ordered",
            b1_baseline_run_dir=None,
            seed_snapshot_run_dir=tmp_path / "seed_run",
            distributed_context=train_script.DistributedContext(enabled=True, rank=0, world_size=2),
        )

    assert calls[-2:] == ["import", "barrier"]


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
    monkeypatch.setattr(train_script, "_experiment_role", lambda _stack: "baseline_noleague")
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
            max_wall_clock_minutes=None,
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
    assert bootstrap_call["permit_current_run_alias"] is True
    assert bootstrap_call["update"] == 0


def test_imported_b1_contract_allows_snapshot_guidance_scale_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    source_run_dir = tmp_path / "b1_source"
    source_run_dir.mkdir()
    source_config = {
        "config": {
            "experiment": {"role": "baseline_noleague"},
            "model": {
                "encoder_kind": "structured_v2",
                "hidden_dim": 8,
                "public_heuristic_logit_bias_scale": 3.0,
                "public_heuristic_actor_logit_bias_scale": 1.0,
                "public_heuristic_logit_bias_start_updates": 0,
                "public_heuristic_logit_bias_end_updates": -1,
                "public_heuristic_logit_bias_final_scale": 3.0,
            },
            "environment": {"max_decisions": 64},
        }
    }
    expected_config = {
        "config": {
            "experiment": {"role": "main"},
            "model": {
                "encoder_kind": "structured_v2",
                "hidden_dim": 8,
                "public_heuristic_logit_bias_scale": 0.0,
                "public_heuristic_actor_logit_bias_scale": 1.0,
                "public_heuristic_logit_bias_start_updates": 20,
                "public_heuristic_logit_bias_end_updates": 120,
                "public_heuristic_logit_bias_final_scale": 0.5,
            },
            "environment": {"max_decisions": 64},
        }
    }
    (source_run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": source_config}),
        encoding="utf-8",
    )

    train_script._validate_imported_snapshot_contract(
        source_run_dir=source_run_dir,
        payload={"model_state_dict": {"w": torch.tensor([1.0])}},
        expected_model_state_dict={"w": torch.tensor([0.0])},
        expected_config_canonical=expected_config,
        expected_spec_hash256=None,
    )


def test_imported_seed_snapshot_contract_allows_snapshot_guidance_scale_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    source_run_dir = tmp_path / "seed_source"
    source_run_dir.mkdir()
    source_config = {
        "config": {
            "experiment": {"role": "main"},
            "model": {
                "encoder_kind": "structured_v2",
                "hidden_dim": 8,
                "public_heuristic_logit_bias_scale": 3.0,
                "public_heuristic_actor_logit_bias_scale": 1.0,
                "public_heuristic_logit_bias_start_updates": 0,
                "public_heuristic_logit_bias_end_updates": -1,
                "public_heuristic_logit_bias_final_scale": 3.0,
            },
            "environment": {"max_decisions": 64},
        }
    }
    expected_config = {
        "config": {
            "experiment": {"role": "main"},
            "model": {
                "encoder_kind": "structured_v2",
                "hidden_dim": 8,
                "public_heuristic_logit_bias_scale": 0.0,
                "public_heuristic_actor_logit_bias_scale": 1.0,
                "public_heuristic_logit_bias_start_updates": 20,
                "public_heuristic_logit_bias_end_updates": 120,
                "public_heuristic_logit_bias_final_scale": 0.5,
            },
            "environment": {"max_decisions": 64},
        }
    }
    (source_run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": source_config}),
        encoding="utf-8",
    )

    train_script._validate_seed_snapshot_import_contract(
        source_run_dir=source_run_dir,
        payload={"model_state_dict": {"w": torch.tensor([1.0])}},
        expected_model_state_dict={"w": torch.tensor([0.0])},
        expected_config_canonical=expected_config,
        expected_spec_hash256=None,
    )


def test_guidance_schedule_applies_configured_actor_bias_after_resume() -> None:
    train_script = _load_train_script_module()

    class _FakeLearner:
        def set_teacher_aux_coefs(self, **_kwargs: float) -> None:
            return None

        def set_reference_policy_bc_coefs(self, **_kwargs: float) -> None:
            return None

    class _FakeModel:
        def __init__(self) -> None:
            self.learner_scale = 3.0
            self.actor_scale = 1.0

        def set_public_heuristic_logit_bias_scale(self, value: float, *, actor_value: float | None = None) -> None:
            self.learner_scale = float(value)
            if actor_value is not None:
                self.actor_scale = float(actor_value)

        def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str = "learner") -> float:
            return self.actor_scale if scoring_mode == "actor" else self.learner_scale

    stack = SimpleNamespace(
        config=SimpleNamespace(
            training=None,
            model=SimpleNamespace(
                public_heuristic_logit_bias_scale=3.0,
                public_heuristic_logit_bias_final_scale=3.0,
                public_heuristic_logit_bias_start_updates=0,
                public_heuristic_logit_bias_end_updates=-1,
                public_heuristic_actor_logit_bias_scale=2.5,
            ),
        )
    )
    model = _FakeModel()

    metrics = train_script._apply_guidance_schedule_for_next_update(
        learner=_FakeLearner(),
        model=model,
        stack=stack,
        update_count=220,
    )

    assert model.learner_scale == pytest.approx(3.0)
    assert model.actor_scale == pytest.approx(2.5)
    assert metrics["public_heuristic_logit_bias_scale_active"] == pytest.approx(3.0)
    assert metrics["public_heuristic_actor_logit_bias_scale_active"] == pytest.approx(2.5)


def test_defer_noleague_baseline_alias_refresh_only_on_periodic_eval_updates() -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(NOLEAGUE_BASELINE_STACK_CONFIG)

    assert (
        train_script._should_defer_noleague_baseline_alias_refresh(
            stack=stack,
            experiment_role="baseline_noleague",
            update_count=10,
        )
        is True
    )
    assert (
        train_script._should_defer_noleague_baseline_alias_refresh(
            stack=stack,
            experiment_role="baseline_noleague",
            update_count=11,
        )
        is False
    )
    assert (
        train_script._should_defer_noleague_baseline_alias_refresh(
            stack=stack,
            experiment_role="main",
            update_count=10,
        )
        is False
    )


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
        update_count=int(stack.config.league.warmup.first_updates),
        league_reference_update=int(stack.config.league.warmup.first_updates),
        league_eval_warmup_gate_open=True,
        policy_version=int(stack.config.league.warmup.first_updates),
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
        update_count=int(stack.config.league.warmup.first_updates) - 1,
        league_reference_update=int(stack.config.league.warmup.first_updates) - 1,
        league_eval_warmup_gate_open=True,
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
        update_count=int(stack.config.league.warmup.first_updates) + 20,
        league_reference_update=int(stack.config.league.warmup.first_updates) - 20,
        league_eval_warmup_gate_open=True,
        policy_version=220,
        run_id256="12" * 32,
        config_hash256="34" * 32,
        spec_hash256="56" * 32,
    )

    assert promoted is None


def test_checkpoint_aliases_track_latest_and_best_and_restore_resume_state(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = _stack_with_periodic_dev_eval_interval(load_stack_config(canonical_stack_config_path()), interval_updates=0)
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    artifacts = train_script._run_artifacts_from_existing_run_dir(run_dir)

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
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.25},
    )
    assert training_paths.latest_checkpoint_path.is_file()
    assert training_paths.best_checkpoint_path.is_file()
    assert tracker["latest"]["metric_kind"] == "training_loss"
    assert tracker["best"]["metric_kind"] == "training_loss"

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
        stack=stack,
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
        learning_rate=0.001,
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
    restored_optimizer = restored_learner._optimizer_for_step()
    assert [group["lr"] for group in restored_optimizer.param_groups] == pytest.approx([0.001])

    reset_optimizer_learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        learning_rate=0.003,
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    reset_resume_state = train_script._restore_learner_from_checkpoint(
        checkpoint_path=training_paths.best_checkpoint_path,
        learner=reset_optimizer_learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        restore_optimizer_state=False,
    )
    assert reset_resume_state.update_count == 4
    assert getattr(reset_optimizer_learner, "optimizer", None) is None
    reset_optimizer = reset_optimizer_learner._optimizer_for_step()
    assert [group["lr"] for group in reset_optimizer.param_groups] == pytest.approx([0.003])

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


def test_checkpoint_aliases_wait_for_dev_eval_metric_when_periodic_eval_enabled(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.periodic_dev_eval_interval_updates > 0
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    artifacts = train_script._run_artifacts_from_existing_run_dir(run_dir)

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
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 0.25},
        dev_eval_summary=None,
    )

    assert training_paths.latest_checkpoint_path.is_file()
    assert not training_paths.best_checkpoint_path.exists()
    assert tracker["latest"]["metric_kind"] is None
    assert tracker["latest"]["metric_value"] is None
    assert tracker.get("best") is None


def test_publish_best_from_dev_eval_skips_null_candidate_without_existing_best(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    artifacts = train_script._run_artifacts_from_existing_run_dir(run_dir)

    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 4
    learner.policy_version = 3
    learner.total_samples_processed = 128
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_4.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )

    tracker = train_script._publish_best_checkpoint_from_dev_eval(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        update_count=4,
        policy_version=3,
        dev_eval_summary=None,
    )

    assert not training_paths.best_checkpoint_path.exists()
    assert tracker.get("best") is None


def test_seed_checkpoint_tracker_from_resume_best_carries_dev_eval_best(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
    source_run_dir = tmp_path / "source"
    source_paths = train_script._training_paths(source_run_dir)
    source_artifacts = train_script._run_artifacts_from_existing_run_dir(source_run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=source_paths.checkpoints_dir,
        logs_dir=source_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 800
    learner.policy_version = 23
    learner.total_samples_processed = 1024
    checkpoint_path = source_paths.checkpoints_dir / "checkpoint_800.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=source_paths,
        artifacts=source_artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.0},
        dev_eval_summary={
            "aggregate_score": 0.8333333333333334,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.0}}
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    target_run_dir = tmp_path / "target"
    target_paths = train_script._training_paths(target_run_dir)
    target_artifacts = train_script._run_artifacts_from_existing_run_dir(target_run_dir)

    seeded = train_script._seed_checkpoint_tracker_from_resume_best(
        stack=stack,
        training_paths=target_paths,
        artifacts=target_artifacts,
        resume_checkpoint_path=checkpoint_path,
    )

    assert seeded is not None
    assert target_paths.best_checkpoint_path.is_file()
    tracker = train_script._load_checkpoint_tracker(target_paths)
    assert tracker["best"]["metric_kind"] == "dev_eval_mean"
    assert tracker["best"]["metric_value"] == pytest.approx(0.8333333333333334)
    assert tracker["best"]["update_count"] == 800
    assert tracker["best"]["seeded_from_run_dir"] == source_run_dir.as_posix()


def test_load_resume_checkpoint_dev_eval_summary_prefers_confirmatory(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
    source_run_dir = tmp_path / "source"
    source_paths = train_script._training_paths(source_run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=source_paths.checkpoints_dir,
        logs_dir=source_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 200
    learner.policy_version = 10
    checkpoint_path = source_paths.checkpoints_dir / "checkpoint_200.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    periodic_summary = {
        "aggregate_score": 0.61,
        "evaluation_surface": {"authoritative": True, "kind": "canonical_scalar"},
    }
    confirmatory_summary = {
        "aggregate_score": 0.65,
        "evaluation_surface": {"authoritative": True, "kind": "canonical_scalar"},
    }
    periodic_path = source_run_dir / "eval" / "dev_eval" / "update_200" / "summary.json"
    confirmatory_path = source_run_dir / "eval" / "dev_eval_confirmatory" / "update_200" / "summary.json"
    periodic_path.parent.mkdir(parents=True, exist_ok=True)
    confirmatory_path.parent.mkdir(parents=True, exist_ok=True)
    periodic_path.write_text(json.dumps(periodic_summary), encoding="utf-8")
    confirmatory_path.write_text(json.dumps(confirmatory_summary), encoding="utf-8")

    loaded = train_script._load_resume_checkpoint_dev_eval_summary(
        stack=stack,
        resume_checkpoint_path=checkpoint_path,
        update_count=200,
    )

    assert loaded is not None
    assert loaded["aggregate_score"] == pytest.approx(0.65)


def test_load_resume_checkpoint_dev_eval_summary_skips_config_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    source_stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
    target_stack = _stack_with_periodic_dev_eval_interval(source_stack, interval_updates=0)
    source_run_dir = tmp_path / "source"
    source_paths = train_script._training_paths(source_run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(source_stack),
        checkpoint_dir=source_paths.checkpoints_dir,
        logs_dir=source_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 200
    checkpoint_path = source_paths.checkpoints_dir / "checkpoint_200.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=source_stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    summary_path = source_run_dir / "eval" / "dev_eval" / "update_200" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"aggregate_score": 0.65}), encoding="utf-8")

    loaded = train_script._load_resume_checkpoint_dev_eval_summary(
        stack=target_stack,
        resume_checkpoint_path=checkpoint_path,
        update_count=200,
    )

    assert loaded is None

    allowed = train_script._load_resume_checkpoint_dev_eval_summary(
        stack=target_stack,
        resume_checkpoint_path=checkpoint_path,
        update_count=200,
        allow_config_hash_mismatch=True,
    )

    assert allowed is not None
    assert allowed["aggregate_score"] == pytest.approx(0.65)


def test_seed_checkpoint_tracker_from_resume_best_skips_config_mismatch(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    source_stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
    target_stack = _stack_with_periodic_dev_eval_interval(source_stack, interval_updates=0)
    assert compute_config_hash256(source_stack) != compute_config_hash256(target_stack)
    source_run_dir = tmp_path / "source"
    source_paths = train_script._training_paths(source_run_dir)
    source_artifacts = train_script._run_artifacts_from_existing_run_dir(source_run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(source_stack),
        checkpoint_dir=source_paths.checkpoints_dir,
        logs_dir=source_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()
    learner.update_count = 800
    learner.policy_version = 23
    checkpoint_path = source_paths.checkpoints_dir / "checkpoint_800.pt"
    train_script._write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=source_stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=source_stack,
        training_paths=source_paths,
        artifacts=source_artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.0},
        dev_eval_summary={
            "aggregate_score": 0.75,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.0}}
            },
        },
    )
    target_run_dir = tmp_path / "target"
    target_paths = train_script._training_paths(target_run_dir)
    target_artifacts = train_script._run_artifacts_from_existing_run_dir(target_run_dir)

    seeded = train_script._seed_checkpoint_tracker_from_resume_best(
        stack=target_stack,
        training_paths=target_paths,
        artifacts=target_artifacts,
        resume_checkpoint_path=checkpoint_path,
    )

    assert seeded is None
    assert not target_paths.best_checkpoint_path.exists()


def test_finalize_from_best_checkpoint_rewrites_latest_alias(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
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
    assert tracker["latest"]["metric_value"] == pytest.approx(0.65625)
    assert tracker["latest"]["source_checkpoint_path"].endswith("training/checkpoints/best.pt")
    latest_payload = torch.load(training_paths.latest_checkpoint_path, map_location="cpu", weights_only=False)
    best_payload = torch.load(training_paths.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert latest_payload["update_count"] == 160
    assert latest_payload["policy_version"] == 8
    assert latest_payload["total_samples_processed"] == 5120
    assert latest_payload["model_state_dict"].keys() == best_payload["model_state_dict"].keys()
    for key, value in latest_payload["model_state_dict"].items():
        assert torch.equal(value, best_payload["model_state_dict"][key])
    resumed = train_script._restore_learner_from_checkpoint(
        checkpoint_path=training_paths.latest_checkpoint_path,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        expected_spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    assert resumed.update_count == 160
    assert resumed.policy_version == 8


def test_checkpoint_guard_rollback_restores_weights_without_rewinding_counters(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
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
        dev_eval_summary={"aggregate_score": 0.75, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )

    learner.update_count = 220
    learner.policy_version = 11
    runtime = SimpleNamespace(reset_count=0, refresh_count=0, publish_count=0)
    runtime.reset_outcome_tracker = lambda: setattr(runtime, "reset_count", runtime.reset_count + 1)
    runtime.refresh_opponent_pool = lambda: setattr(runtime, "refresh_count", runtime.refresh_count + 1)
    runtime.maybe_publish_snapshot = lambda **_kwargs: (
        setattr(runtime, "publish_count", runtime.publish_count + 1)
        or {"snapshot_publish_latency_ms": 1.0, "snapshot_apply_latency_ms": 2.0}
    )

    event = train_script._maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        model=learner.model,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.5, "stall_monitor": {"worst_truncation_rate": 0.0}},
        last_rollback_update=None,
    )

    assert event is not None
    assert event["restored_weight_update_count"] == 160
    assert event["update_count"] == 220
    assert learner.update_count == 220
    assert learner.policy_version == 11
    assert runtime.publish_count == 1
    assert runtime.reset_count == 1
    assert runtime.refresh_count == 1


def test_checkpoint_guard_does_not_rollback_improving_score_on_confidence_only_noise(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
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
    learner.update_count = 100
    learner.policy_version = 5
    best_checkpoint = training_paths.checkpoints_dir / "checkpoint_100.pt"
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
        dev_eval_summary={"aggregate_score": 0.55, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )
    learner.update_count = 120
    learner.policy_version = 6
    runtime = SimpleNamespace()

    event = train_script._maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        model=learner.model,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        latest_metrics={"loss": 0.1},
        dev_eval_summary={
            "aggregate_score": 0.59,
            "stall_monitor": {"worst_truncation_rate": 0.0},
            "anchors": {
                "B1 NoLeague baseline": {
                    "uncertainty": {
                        "mean": 0.4375,
                        "prob_gt_half": 0.0,
                        "prob_lt_half": 1.0,
                        "ci_half_width": 0.155,
                    }
                }
            },
        },
        last_rollback_update=None,
    )

    assert event is None
    assert learner.update_count == 120
    assert learner.policy_version == 6


def test_checkpoint_guard_does_not_rollback_equal_score_on_confidence_only_noise(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "local.yaml")
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
    learner.update_count = 100
    learner.policy_version = 5
    best_checkpoint = training_paths.checkpoints_dir / "checkpoint_100.pt"
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
        dev_eval_summary={"aggregate_score": 0.55, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )
    learner.update_count = 120
    learner.policy_version = 6

    event = train_script._maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=SimpleNamespace(),
        learner=learner,
        model=learner.model,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        latest_metrics={"loss": 0.1},
        dev_eval_summary={
            "aggregate_score": 0.55,
            "stall_monitor": {"worst_truncation_rate": 0.0},
            "anchors": {
                "B1 NoLeague baseline": {
                    "uncertainty": {
                        "mean": 0.4375,
                        "prob_gt_half": 0.0,
                        "prob_lt_half": 1.0,
                        "ci_half_width": 0.155,
                    }
                }
            },
        },
        last_rollback_update=None,
    )

    assert event is None
    assert learner.update_count == 120
    assert learner.policy_version == 6


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


def test_reject_registry_snapshots_newer_than_marks_newer_refs(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    training_paths = train_script._training_paths(tmp_path)
    registry_path = training_paths.snapshots_dir / "registry.json"
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
    registry.add_snapshot(
        policy_id="policy_000140",
        update=140,
        weights_sha256="2" * 64,
        path="training/snapshots/policy_000140/weights.pt",
    )
    registry.add_champion("policy_000120")
    registry.save(registry_path)

    rejected = train_script._reject_registry_snapshots_newer_than(training_paths, update_count=100)

    reloaded = SnapshotRegistry.load(registry_path)
    assert rejected == ["policy_000120", "policy_000140"]
    assert reloaded.rejected_snapshots == ["policy_000120", "policy_000140"]
    assert "policy_000080" not in reloaded.rejected_snapshots


def test_reject_registry_snapshots_newer_than_preserves_external_seed_pool(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    training_paths = train_script._training_paths(tmp_path)
    registry_path = training_paths.snapshots_dir / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000120",
        update=120,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000120/weights.pt",
    )
    registry.add_snapshot(
        policy_id="seed_abcd_policy_000450",
        update=450,
        weights_sha256="2" * 64,
        path="training/snapshots/seed_abcd_policy_000450/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=450,
        weights_sha256="3" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_imported_baseline",
        update=450,
        weights_sha256="4" * 64,
        path="training/snapshots/policy_imported_baseline/weights.pt",
    )
    imported_meta_dir = training_paths.snapshots_dir / "policy_imported_baseline"
    imported_meta_dir.mkdir(parents=True, exist_ok=True)
    (imported_meta_dir / "policy_meta.json").write_text(
        json.dumps(
            {
                "format": "imported_train_snapshot_metadata_v1",
                "policy_id": "policy_imported_baseline",
                "imported_from_run_dir": "runs/source",
                "imported_from_policy_id": "policy_000450",
            }
        ),
        encoding="utf-8",
    )
    registry.save(registry_path)

    rejected = train_script._reject_registry_snapshots_newer_than(training_paths, update_count=100)

    reloaded = SnapshotRegistry.load(registry_path)
    assert rejected == ["policy_000120"]
    assert reloaded.rejected_snapshots == ["policy_000120"]


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
    run_dir = tmp_path / "resume_run"
    latest_path = run_dir / "training" / "checkpoints" / "latest.pt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_bytes(b"checkpoint")

    resolved = train_script._resolve_resume_checkpoint_path(
        resume_from="",
        resume_run_dir=run_dir,
    )

    assert resolved == latest_path.resolve()


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
    assert env.closed is False
    runner.close()
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

    runner.run_game(scheduled_game)

    assert model.scoring_modes == ["learner"]
    assert env.closed is False
    runner.close()
    assert env.closed is True


def test_periodic_dev_eval_runner_reuses_env_across_games(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    envs: list[_FakeOneStepEvalEnv] = []

    class _FakeModel:
        def initial_seat_hidden(self, _batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((1, 1), device=device)

    def fake_build_env(*_args: object, **_kwargs: object) -> _FakeOneStepEvalEnv:
        env = _FakeOneStepEvalEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(train_script, "_build_ids_eval_env", fake_build_env)
    runner = train_script._PeriodicDevEvalRunner(
        stack=stack,
        model=_FakeModel(),
        opponent_policy_id="b0_randomlegal",
        observation_dim=4,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        focal_policy_id="policy",
        require_sorted_legal_ids=True,
        eval_device="cpu",
    )

    for index, seed in enumerate((303, 404)):
        runner.run_game(
            ScheduledGame(
                pair_index=index,
                swap_index=0,
                episode_index=index,
                episode_seed=seed,
                focal_policy_id="policy",
                opponent_policy_id="b0_randomlegal",
                seat0_policy_id="b0_randomlegal",
                seat1_policy_id="policy",
                focal_seat=1,
            )
        )

    assert len(envs) == 1
    assert envs[0].reset_seeds == [303, 404]
    runner.close()
    assert envs[0].close_count == 1


def test_periodic_dev_eval_runner_batches_independent_model_decisions(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    envs: list[_FakeOneStepEvalEnv] = []

    class _FakeModel:
        def __init__(self) -> None:
            self.forward_batch_sizes: list[int] = []

        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 2, 1), device=device)

        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            seat_hidden_state: torch.Tensor | None = None,
            *,
            scoring_mode: str = "auto",
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.forward_batch_sizes.append(int(obs.shape[0]))
            assert scoring_mode == "learner"
            assert seat_hidden_state is not None
            assert tuple(seat_hidden_state.shape) == (int(obs.shape[0]), 2, 1)
            logits = torch.zeros((int(obs.shape[0]), 1), dtype=torch.float32, device=obs.device)
            values = torch.zeros((int(obs.shape[0]),), dtype=torch.float32, device=obs.device)
            return logits, values, seat_hidden_state + 1.0

    def fake_build_env(*_args: object, **_kwargs: object) -> _FakeOneStepEvalEnv:
        env = _FakeOneStepEvalEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(train_script, "_build_ids_eval_env", fake_build_env)
    model = _FakeModel()
    runner = train_script._PeriodicDevEvalRunner(
        stack=stack,
        model=model,
        opponent_policy_id="b0_randomlegal",
        observation_dim=4,
        action_dim=1,
        pass_action_id=0,
        artifact_dir=tmp_path,
        focal_policy_id="policy",
        require_sorted_legal_ids=True,
        eval_device="cpu",
    )
    scheduled_games = [
        ScheduledGame(
            pair_index=index,
            swap_index=0,
            episode_index=index,
            episode_seed=seed,
            focal_policy_id="policy",
            opponent_policy_id="b0_randomlegal",
            seat0_policy_id="policy",
            seat1_policy_id="b0_randomlegal",
            focal_seat=0,
        )
        for index, seed in enumerate((303, 404))
    ]

    completed_games = runner.run_scheduled_games_batched(scheduled_games)

    assert model.forward_batch_sizes == [2]
    assert [game.episode_seed for game, _result in completed_games] == [303, 404]
    counters = runner.drain_runtime_counters()
    assert counters["counts"]["model_forward_calls"] == 1
    assert counters["counts"]["model_forward_rows"] == 2
    runner.close()
    assert len(envs) == 2
    assert [env.close_count for env in envs] == [1, 1]


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
    assert env.closed is False
    runner.close()
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
    assert env.closed is False
    runner.close()
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
        stack=SimpleNamespace(config=SimpleNamespace(curriculum=None)),
        policies={
            "candidate": ResolvedEvalPolicy(
                policy_id="candidate",
                kind="snapshot_registry",
                model=model,
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
    monkeypatch.setattr(runner, "_build_ids_eval_env", lambda *, seed: env)

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
    assert env.closed is False
    runner.close()
    assert env.closed is True


def test_simulator_eval_runner_reuses_env_when_replay_capture_disabled(tmp_path: Path, monkeypatch) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    envs: list[_FakeOneStepEvalEnv] = []

    def fake_build_env(_self: SimulatorEvalRunner, *, seed: int) -> _FakeOneStepEvalEnv:
        env = _FakeOneStepEvalEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(SimulatorEvalRunner, "_build_ids_eval_env", fake_build_env)
    runner = SimulatorEvalRunner(
        stack=stack,
        policies={"policy": ResolvedEvalPolicy(policy_id="policy", kind="random_legal")},
        artifact_layout=ArtifactLayout.from_run_dir(tmp_path),
        run_id256="12" * 32,
        spec_hash256="34" * 32,
        action_dim=1,
        pass_action_id=0,
        require_sorted_legal_ids=True,
        replay_capture_rate=0.0,
        regression_capture_count=0,
        eval_device="cpu",
    )

    for index, seed in enumerate((101, 202)):
        runner.run_game(
            ScheduledGame(
                pair_index=index,
                swap_index=0,
                episode_index=index,
                episode_seed=seed,
                focal_policy_id="policy",
                opponent_policy_id="policy",
                seat0_policy_id="policy",
                seat1_policy_id="policy",
                focal_seat=0,
            )
        )

    assert len(envs) == 1
    assert envs[0].reset_seeds == [101, 202]
    assert envs[0].close_count == 0
    runner.close()
    assert envs[0].close_count == 1
    runner.close()
    assert envs[0].close_count == 1


def test_simulator_eval_runner_does_not_reuse_env_when_replay_capture_enabled(tmp_path: Path, monkeypatch) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    envs: list[_FakeOneStepEvalEnv] = []

    def fake_build_env(_self: SimulatorEvalRunner, *, seed: int) -> _FakeOneStepEvalEnv:
        env = _FakeOneStepEvalEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(SimulatorEvalRunner, "_build_ids_eval_env", fake_build_env)
    runner = SimulatorEvalRunner(
        stack=stack,
        policies={"policy": ResolvedEvalPolicy(policy_id="policy", kind="random_legal")},
        artifact_layout=ArtifactLayout.from_run_dir(tmp_path),
        run_id256="12" * 32,
        spec_hash256="34" * 32,
        action_dim=1,
        pass_action_id=0,
        require_sorted_legal_ids=True,
        replay_capture_rate=1.0,
        regression_capture_count=0,
        eval_device="cpu",
    )

    first = runner._env_for_game(seed=101)
    second = runner._env_for_game(seed=202)

    assert first is not second
    assert len(envs) == 2
    first.close()
    second.close()


def test_simulator_eval_runner_honors_cuda_auto_eval_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_calls: list[str] = []

        def to(self, device: torch.device | str) -> FakeModel:
            self.to_calls.append(str(device))
            return self

        def eval(self) -> FakeModel:
            return self

    monkeypatch.setattr("weiss_rl.eval.simulator_runner.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("weiss_rl.eval.simulator_runner.torch.cuda.device_count", lambda: 3)

    model = FakeModel()
    layout = ArtifactLayout.from_run_dir(tmp_path)
    layout.ensure_directories()
    runner = SimulatorEvalRunner(
        stack=SimpleNamespace(
            config=SimpleNamespace(curriculum=None, evaluation=SimpleNamespace(eval_device="cuda:auto"))
        ),
        policies={
            "candidate": ResolvedEvalPolicy(
                policy_id="candidate",
                kind="snapshot_registry",
                model=model,
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

    assert model.to_calls[-1] == "cuda:0"
    assert str(cast(Any, runner)._device) == "cuda:0"
