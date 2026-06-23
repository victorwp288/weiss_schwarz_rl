from __future__ import annotations

from pathlib import Path

import pytest


def test_canonical_eval_runtime_phase_persists_selection_before_loading_policies(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.phases import (
        CanonicalEvalRunState,
        resolve_canonical_eval_runtime_state,
    )

    calls: list[str] = []
    observed: dict[str, object] = {}
    layout = SimpleNamespace()
    evaluation = SimpleNamespace(
        eval_assert_sorted_legal_ids=True,
        replay_capture_rate_eval=0.25,
        regression_capture_count=2,
        final_matrix_stage1_paired_seeds=2,
        final_matrix_stage2_adaptive_max_paired_seeds=3,
    )
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"report_eval": tmp_path / "report_eval_seeds.txt"},
        config=SimpleNamespace(evaluation=evaluation),
    )
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=evaluation,
        study_config=None,
    )

    class FakeContract:
        spec_bundle = {
            "observation": {"obs_len": 512},
            "action": {"action_space_size": 9, "pass_action_id": 8},
        }

    def fake_resolve_policy_ids_for_run_fn(**_kwargs: object) -> tuple[list[str], dict[str, object], None, None]:
        calls.append("resolve_policy_ids")
        return ["B0 RandomLegal", "policy_000100"], {"status": "resolved"}, None, None

    def fake_persist_policy_selection_in_manifest_fn(**kwargs: object) -> None:
        calls.append("persist_selection")
        observed["persisted"] = kwargs

    def fake_load_verified_simulator_contract_fn(*_args: object, **_kwargs: object) -> FakeContract:
        calls.append("load_contract")
        return FakeContract()

    def fake_resolve_eval_policies_fn(**kwargs: object) -> list[str]:
        calls.append("resolve_eval_policies")
        observed["policy_resolution"] = kwargs
        return ["policy-object"]

    def fake_simulator_eval_runner_cls(**kwargs: object) -> object:
        calls.append("build_runner")
        observed["runner"] = kwargs
        return object()

    def fake_parse_seed_file_fn(path: Path) -> list[int]:
        calls.append("parse_seeds")
        observed["seed_file"] = path
        return [101, 202, 303]

    dependencies = SimpleNamespace(
        resolve_policy_ids_for_run_fn=fake_resolve_policy_ids_for_run_fn,
        persist_policy_selection_in_manifest_fn=fake_persist_policy_selection_in_manifest_fn,
        load_verified_simulator_contract_fn=fake_load_verified_simulator_contract_fn,
        resolve_eval_policies_fn=fake_resolve_eval_policies_fn,
        simulator_eval_runner_cls=fake_simulator_eval_runner_cls,
        parse_seed_file_fn=fake_parse_seed_file_fn,
    )

    runtime_state = resolve_canonical_eval_runtime_state(
        stack=stack,
        run_dir=tmp_path / "run",
        policy_ids=[],
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        b1_baseline_run_dir=tmp_path / "b1",
        paired_seed_limit=2,
        stage1_paired_seeds=None,
        max_paired_seeds=None,
        run_state=run_state,
        dependencies=dependencies,
    )

    assert calls == [
        "resolve_policy_ids",
        "persist_selection",
        "load_contract",
        "resolve_eval_policies",
        "build_runner",
        "parse_seeds",
    ]
    assert runtime_state.policy_ids == ["B0 RandomLegal", "policy_000100"]
    assert runtime_state.paired_seeds == [101, 202]
    assert runtime_state.paired_seed_limit == 2
    assert runtime_state.stage1_paired_seeds == 2
    assert runtime_state.max_paired_seeds == 2
    assert observed["policy_resolution"]["observation_dim"] == 512
    assert observed["policy_resolution"]["action_dim"] == 9
    assert observed["policy_resolution"]["b1_baseline_run_dir"] == tmp_path / "b1"
    assert observed["runner"]["pass_action_id"] == 8
    assert observed["runner"]["require_sorted_legal_ids"] is True


def test_canonical_eval_seed_budget_preserves_limit_defaults_and_errors(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.seed_budget import resolve_canonical_eval_seed_budget

    seed_path = tmp_path / "report_eval_seeds.txt"
    stack = SimpleNamespace(seed_sets={"report_eval": seed_path})
    evaluation = SimpleNamespace(
        final_matrix_stage1_paired_seeds=4,
        final_matrix_stage2_adaptive_max_paired_seeds=8,
    )
    dependencies = SimpleNamespace(parse_seed_file_fn=lambda path: [11, 22, 33] if path == seed_path else [])

    seed_budget = resolve_canonical_eval_seed_budget(
        stack=stack,
        evaluation=evaluation,
        paired_seed_limit=2,
        stage1_paired_seeds=None,
        max_paired_seeds=None,
        dependencies=dependencies,
    )

    assert seed_budget.seed_file_path == seed_path
    assert seed_budget.paired_seeds == [11, 22]
    assert seed_budget.paired_seed_limit == 2
    assert seed_budget.stage1_paired_seeds == 2
    assert seed_budget.max_paired_seeds == 2

    with pytest.raises(ValueError, match=r"stage1 paired seeds \(3\) cannot exceed max paired seeds \(2\)"):
        resolve_canonical_eval_seed_budget(
            stack=stack,
            evaluation=evaluation,
            paired_seed_limit=None,
            stage1_paired_seeds=3,
            max_paired_seeds=2,
            dependencies=dependencies,
        )

    empty_dependencies = SimpleNamespace(parse_seed_file_fn=lambda _path: [])
    with pytest.raises(ValueError, match="report_eval seed file produced no usable seeds"):
        resolve_canonical_eval_seed_budget(
            stack=stack,
            evaluation=evaluation,
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
            dependencies=empty_dependencies,
        )
