from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from weiss_rl.config import load_stack_config
from weiss_rl.league import PromotionGatePosterior, PromotionGateRate, PromotionGateResult
from weiss_rl.league.registry import (
    SnapshotRegistry,
)

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _heuristic_public_contract_bundle,
    _load_train_script_module,
    _make_policy_value_model,
    _write_b1_baseline_run_fixture,
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
