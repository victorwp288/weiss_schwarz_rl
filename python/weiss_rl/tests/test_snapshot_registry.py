from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from weiss_rl.config import load_stack_config
from weiss_rl.league import PromotionGatePosterior, PromotionGateRate, PromotionGateResult
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.model import PolicyValueModel

REPO_ROOT = Path(__file__).resolve().parents[3]


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
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_7.pt"
    torch.save({"format": "checkpoint_stub"}, checkpoint_path)

    model = torch.nn.Linear(3, 2)
    train_script._persist_snapshot_registry_entry(
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


def test_ensure_noleague_baseline_anchor_bootstraps_frozen_snapshot_once(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.model is not None

    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    baseline_model = PolicyValueModel(
        observation_dim=512,
        config=stack.config.model,
        action_dim=9,
    )
    bootstrap_learner = SimpleNamespace(
        model=baseline_model,
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
    )

    assert policy_id == "b1_noleague_baseline"
    registry_path = training_paths.snapshots_dir / "registry.json"
    registry = SnapshotRegistry.load(registry_path)
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [policy_id]

    snapshot = registry.snapshots[0]
    weights_path = run_dir / snapshot_weights_relpath(policy_id)
    metadata_path = training_paths.snapshots_dir / policy_id / "policy_meta.json"
    checkpoint_path = training_paths.checkpoints_dir / "baseline_checkpoint.pt"

    assert snapshot.update == 0
    assert checkpoint_path.is_file()
    assert weights_path.is_file()
    assert snapshot.weights_sha256 == train_script._sha256_file(weights_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "format": "minimal_train_snapshot_metadata_v1",
        "policy_id": policy_id,
        "source_checkpoint_path": "training/checkpoints/baseline_checkpoint.pt",
        "update": 0,
        "weights_path": snapshot_weights_relpath(policy_id),
        "weights_sha256": snapshot.weights_sha256,
    }

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert payload["policy_id"] == policy_id
    assert payload["update"] == 0
    assert payload["config_hash256"] == "ab" * 32

    second_policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=bootstrap_learner,
        device=torch.device("cpu"),
        config_hash256="ff" * 32,
    )

    assert second_policy_id == policy_id
    reloaded = SnapshotRegistry.load(registry_path)
    assert [snapshot.policy_id for snapshot in reloaded.snapshots] == [policy_id]
    assert reloaded.snapshots[0].weights_sha256 == snapshot.weights_sha256


def test_run_minimal_training_bootstraps_noleague_baseline_before_env_start(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")

    bootstrap_calls: list[dict[str, object]] = []

    def fake_ensure_noleague_baseline_anchor(**kwargs):
        bootstrap_calls.append(kwargs)
        return "b1_noleague_baseline"

    def stop_before_env(*args, **kwargs):
        raise RuntimeError("stop after bootstrap")

    monkeypatch.setattr(train_script, "_ensure_noleague_baseline_anchor", fake_ensure_noleague_baseline_anchor)
    monkeypatch.setattr(train_script, "_build_env", stop_before_env)

    run_dir = tmp_path / "run"
    try:
        train_script._run_minimal_training(
            stack=stack,
            contract=SimpleNamespace(
                spec_bundle={
                    "observation": {"obs_len": 512},
                    "action": {"action_space_size": 9, "pass_action_id": 8},
                }
            ),
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
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after bootstrap"
    else:
        raise AssertionError("expected _build_env to stop the test after baseline bootstrap")

    assert len(bootstrap_calls) == 1
    bootstrap_call = bootstrap_calls[0]
    assert bootstrap_call["run_dir"] == run_dir
    assert bootstrap_call["training_paths"].snapshots_dir == run_dir / "training" / "snapshots"
    assert bootstrap_call["device"] == torch.device("cpu")
    assert bootstrap_call["config_hash256"] == train_script.compute_config_hash256(stack)


def test_run_snapshot_promotion_gate_marks_passed_candidate_as_champion(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.model is not None

    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)

    baseline_model = PolicyValueModel(
        observation_dim=512,
        config=stack.config.model,
        action_dim=9,
    )
    train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=SimpleNamespace(
            model=baseline_model,
            update_count=0,
            optimizer=None,
            get_policy_version=lambda: 0,
        ),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
    )
    registry_path = training_paths.snapshots_dir / "registry.json"

    learner_model = PolicyValueModel(
        observation_dim=512,
        config=stack.config.model,
        action_dim=9,
    )
    candidate_checkpoint_path = training_paths.checkpoints_dir / "checkpoint_7.pt"
    torch.save({"format": "checkpoint_stub"}, candidate_checkpoint_path)
    candidate_policy_id = train_script._persist_snapshot_registry_entry(
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
        }
        return PromotionGateResult(
            focal_policy_id=candidate_policy_id,
            ordered_opponents=("B0 RandomLegal", "B1 NoLeague baseline"),
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
        contract=SimpleNamespace(
            spec_bundle={
                "observation": {"obs_len": 512},
                "action": {"action_space_size": 9, "pass_action_id": 8},
            }
        ),
        artifacts=SimpleNamespace(run_dir=run_dir),
        training_paths=training_paths,
        learner=SimpleNamespace(model=learner_model),
        candidate_policy_id=candidate_policy_id,
        update_count=7,
        policy_version=7,
        run_id256="12" * 32,
        config_hash256="34" * 32,
        spec_hash256="56" * 32,
    )

    assert promoted is True
    registry = SnapshotRegistry.load(registry_path)
    assert registry.champion_snapshots == [candidate_policy_id]
