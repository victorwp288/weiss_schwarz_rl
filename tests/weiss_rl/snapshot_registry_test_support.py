from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast

import torch
from weiss_rl.config import canonical_config_dict, load_stack_config
from weiss_rl.league.registry import (
    SnapshotRegistry,
    snapshot_weights_relpath,
)
from weiss_rl.model import PolicyValueModel
from weiss_rl.training import train_entrypoint as train_entrypoint_module

from ._config_paths import canonical_stack_config_path

REPO_ROOT = Path(__file__).resolve().parents[2]


class _TrainingPathsLike(Protocol):
    snapshots_dir: Path


@lru_cache(maxsize=1)
def _load_train_script_module() -> ModuleType:
    return train_entrypoint_module


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


def _make_bootstrap_learner(stack: Any, *, update_count: int = 0, policy_version: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        model=_make_policy_value_model(stack),
        update_count=update_count,
        optimizer=None,
        get_policy_version=lambda: policy_version,
    )


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


def _import_seed_snapshot_pool_for_test(
    tmp_path: Path,
    *,
    stack: Any,
    seed_run_dir: Path,
    consumer_name: str,
    expected_spec_hash256: str = "cd" * 32,
) -> SimpleNamespace:
    train_script = _load_train_script_module()
    consumer_run_dir = tmp_path / consumer_name
    training_paths = train_script._training_paths(consumer_run_dir)
    imported_policy_ids = train_script._import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=consumer_run_dir,
        seed_snapshot_run_dir=seed_run_dir,
        expected_model_state_dict=_make_policy_value_model(stack).state_dict(),
        expected_config_canonical=canonical_config_dict(stack),
        expected_spec_hash256=expected_spec_hash256,
    )
    return SimpleNamespace(
        train_script=train_script,
        consumer_run_dir=consumer_run_dir,
        training_paths=training_paths,
        imported_policy_ids=imported_policy_ids,
    )


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
