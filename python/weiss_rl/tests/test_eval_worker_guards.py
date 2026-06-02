from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]

ENTRYPOINT_MODULES = {
    "parallel_final_eval": "weiss_rl.eval.parallel_final_eval_entrypoint",
    "targeted_confirm_eval": "weiss_rl.eval.targeted_confirm.entrypoint",
}


def _load_script_module(name: str):
    return import_module(ENTRYPOINT_MODULES[name])


def _script_env() -> dict[str, str]:
    env = dict(os.environ)
    python_path = str(REPO_ROOT / "python")
    env["PYTHONPATH"] = python_path if not env.get("PYTHONPATH") else python_path + os.pathsep + env["PYTHONPATH"]
    return env


def _script_env_without_hash_seed() -> dict[str, str]:
    env = _script_env()
    env.pop("PYTHONHASHSEED", None)
    return env


def test_targeted_confirm_eval_rejects_parallel_workers_without_escape_hatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["targeted_confirm_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--snapshot-registry-json",
            "missing_registry.json",
            "--b1-baseline-run-dir",
            "runs/missing_b1",
            "--focal-policy-id",
            "policy_000001",
            "--workers",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_script_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "deterministic only with --workers 1" in result.stderr


def test_parallel_final_eval_rejects_parallel_workers_without_escape_hatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["parallel_final_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--policy-id",
            "policy_000001",
            "--workers",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_script_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "deterministic only with --workers 1" in result.stderr


def test_targeted_confirm_eval_requires_fixed_pythonhashseed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["targeted_confirm_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--snapshot-registry-json",
            "missing_registry.json",
            "--b1-baseline-run-dir",
            "runs/missing_b1",
            "--focal-policy-id",
            "policy_000001",
            "--workers",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_script_env_without_hash_seed(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr


def test_parallel_final_eval_requires_fixed_pythonhashseed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            ENTRYPOINT_MODULES["parallel_final_eval"],
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--policy-id",
            "policy_000001",
            "--workers",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_script_env_without_hash_seed(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr


def test_parallel_final_eval_worker_accepts_noncanonical_output_dir() -> None:
    from weiss_rl.eval import parallel_final_eval_entrypoint as package_module

    module = _load_script_module("parallel_final_eval")
    layout = SimpleNamespace(final_eval_dir=Path("runs/demo/eval/final_eval"))

    assert module._worker is package_module._worker
    assert module._worker_output_dir({"output_dir": "runs/demo/eval/confirm16"}, layout=layout) == Path(
        "runs/demo/eval/confirm16"
    )
    assert package_module._worker_output_dir({"output_dir": "runs/demo/eval/confirm16"}, layout=layout) == Path(
        "runs/demo/eval/confirm16"
    )
    assert module._worker_output_dir({}, layout=layout) == Path("runs/demo/eval/final_eval")


def test_parallel_final_eval_entrypoint_facade_reexports_core_helpers() -> None:
    from weiss_rl.eval import (
        final_eval_worker,
        parallel_final_eval_core,
        parallel_final_eval_entrypoint,
        parallel_final_eval_plan,
    )

    assert parallel_final_eval_entrypoint.main is parallel_final_eval_core.main
    assert parallel_final_eval_entrypoint._load_json is final_eval_worker.load_json_object
    assert parallel_final_eval_entrypoint._run_jobs is parallel_final_eval_core._run_jobs
    assert parallel_final_eval_entrypoint._worker is final_eval_worker.run_final_eval_worker
    assert parallel_final_eval_entrypoint._worker_output_dir is final_eval_worker.worker_output_dir
    assert parallel_final_eval_entrypoint.ParallelFinalEvalPlan is parallel_final_eval_plan.ParallelFinalEvalPlan
    assert parallel_final_eval_entrypoint.build_arg_parser is parallel_final_eval_plan.build_arg_parser
    assert parallel_final_eval_entrypoint.build_parallel_final_eval_jobs is (
        parallel_final_eval_plan.build_parallel_final_eval_jobs
    )
    assert parallel_final_eval_entrypoint.parse_args is parallel_final_eval_plan.parse_args
    assert parallel_final_eval_entrypoint.prepare_parallel_final_eval_plan is (
        parallel_final_eval_plan.prepare_parallel_final_eval_plan
    )
    assert parallel_final_eval_entrypoint.validate_parallel_final_eval_request is (
        parallel_final_eval_plan.validate_parallel_final_eval_request
    )
    assert parallel_final_eval_entrypoint.write_parallel_final_eval_artifacts is (
        parallel_final_eval_plan.write_parallel_final_eval_artifacts
    )


def test_final_eval_worker_facade_reexports_runtime_helpers() -> None:
    from weiss_rl.eval import final_eval_worker, final_eval_worker_runtime

    assert final_eval_worker.FinalEvalWorkerContext is final_eval_worker_runtime.FinalEvalWorkerContext
    assert final_eval_worker.build_final_eval_worker_runner is final_eval_worker_runtime.build_final_eval_worker_runner
    assert (
        final_eval_worker.final_eval_worker_policy_resolution_kwargs
        is final_eval_worker_runtime.final_eval_worker_policy_resolution_kwargs
    )
    assert final_eval_worker.load_final_eval_worker_context is final_eval_worker_runtime.load_final_eval_worker_context
    assert final_eval_worker.load_json_object is final_eval_worker_runtime.load_json_object
    assert final_eval_worker.optional_job_path is final_eval_worker_runtime.optional_job_path
    assert (
        final_eval_worker.resolve_final_eval_worker_policies
        is final_eval_worker_runtime.resolve_final_eval_worker_policies
    )
    assert final_eval_worker.run_final_eval_worker is final_eval_worker_runtime.run_final_eval_worker
    assert final_eval_worker.run_final_eval_worker_matchup is final_eval_worker_runtime.run_final_eval_worker_matchup
    assert final_eval_worker.unique_policy_resolution_ids is final_eval_worker_runtime.unique_policy_resolution_ids
    assert final_eval_worker.worker_output_dir is final_eval_worker_runtime.worker_output_dir


def test_final_eval_worker_uses_unique_policy_resolution_ids_for_diagonal_jobs() -> None:
    from weiss_rl.eval.final.worker import unique_policy_resolution_ids

    assert unique_policy_resolution_ids(
        {
            "focal_policy_id": "policy_000021",
            "opponent_policy_id": "policy_000021",
        }
    ) == ["policy_000021"]
    assert unique_policy_resolution_ids(
        {
            "focal_policy_id": "policy_000021",
            "opponent_policy_id": "B2 HeuristicPublic",
        }
    ) == ["policy_000021", "B2 HeuristicPublic"]


def test_final_eval_worker_policy_resolution_kwargs_preserve_registry_inputs(monkeypatch) -> None:
    from weiss_rl.eval.final.worker_runtime import (
        final_eval_worker_policy_resolution_kwargs,
        optional_job_path,
        resolve_final_eval_worker_policies,
    )

    spec_bundle = {
        "observation": {"obs_len": 42},
        "action": {"action_space_size": 17, "pass_action_id": 3},
    }
    context = SimpleNamespace(
        stack=SimpleNamespace(name="stack"),
        run_dir=Path("runs/main"),
        observation_dim=42,
        action_dim=17,
        contract=SimpleNamespace(spec_bundle=spec_bundle),
    )
    job = {
        "focal_policy_id": "policy_000021",
        "opponent_policy_id": "policy_000021",
        "snapshot_registry_json": "runs/main/training/snapshots/registry.json",
        "b1_baseline_run_dir": "runs/b1_baseline",
    }

    kwargs = final_eval_worker_policy_resolution_kwargs(job, context)

    assert optional_job_path(job, "snapshot_registry_json") == Path("runs/main/training/snapshots/registry.json")
    assert optional_job_path({"snapshot_registry_json": "   "}, "snapshot_registry_json") is None
    assert kwargs == {
        "stack": context.stack,
        "policy_ids": ["policy_000021"],
        "run_dir": Path("runs/main"),
        "observation_dim": 42,
        "action_dim": 17,
        "spec_bundle": spec_bundle,
        "snapshot_registry_path": Path("runs/main/training/snapshots/registry.json"),
        "b1_baseline_run_dir": Path("runs/b1_baseline"),
    }

    captured: dict[str, object] = {}

    def fake_resolve_eval_policies(**resolve_kwargs: object) -> dict[str, object]:
        captured.update(resolve_kwargs)
        return {"policy_000021": object()}

    from weiss_rl.eval import final_eval_worker_runtime

    monkeypatch.setattr(final_eval_worker_runtime, "resolve_eval_policies", fake_resolve_eval_policies)
    policies = resolve_final_eval_worker_policies(job, context)

    assert list(policies) == ["policy_000021"]
    assert captured == kwargs


def test_parallel_final_eval_plan_builds_upper_triangular_jobs() -> None:
    from weiss_rl.eval.parallel_final_eval_plan import build_parallel_final_eval_jobs

    args = SimpleNamespace(
        stack_config=Path("configs/thesis/final_eval.yaml"),
        run_dir=Path("runs/demo"),
        snapshot_registry_json=None,
        b1_baseline_run_dir=Path("runs/b1"),
        stage1_paired_seeds=8,
        max_paired_seeds=16,
        bootstrap_samples=1000,
    )

    jobs = build_parallel_final_eval_jobs(
        args=args,
        policy_ids=["policy_a", "policy_b", "policy_c"],
        paired_seeds=[10, 11, 12],
        scheme="S0",
    )

    assert [(job["focal_policy_id"], job["opponent_policy_id"]) for job in jobs] == [
        ("policy_a", "policy_a"),
        ("policy_a", "policy_b"),
        ("policy_a", "policy_c"),
        ("policy_b", "policy_b"),
        ("policy_b", "policy_c"),
        ("policy_c", "policy_c"),
    ]
    assert jobs[0]["snapshot_registry_json"] is None
    assert jobs[0]["b1_baseline_run_dir"] == "runs/b1"
    assert jobs[0]["paired_seeds"] == [10, 11, 12]
    assert jobs[0]["stage1_paired_seeds"] == 8
    assert jobs[0]["max_paired_seeds"] == 16


def test_targeted_confirm_jobs_write_into_requested_output_subdir() -> None:
    module = _load_script_module("targeted_confirm_eval")
    args = SimpleNamespace(
        stack_config=Path("configs/thesis/final_eval.yaml"),
        run_dir=Path("runs/demo"),
        snapshot_registry_json=Path("runs/demo/training/snapshots/registry.json"),
        b1_baseline_run_dir=Path("runs/demo"),
        paired_seeds=16,
        bootstrap_samples=1000,
        focal_policy_id="policy_000003",
    )

    job = module._targeted_eval_job(
        args=args,
        paired_seeds=[1, 2, 3],
        opponent_index=2,
        opponent="B2 HeuristicPublic",
        output_dir=Path("runs/demo/eval/confirm16_argmax"),
    )

    assert job["output_dir"] == "runs/demo/eval/confirm16_argmax"
    assert job["opponent_policy_id"] == "B2 HeuristicPublic"
    assert job["paired_seeds"] == [1, 2, 3]


def test_targeted_confirm_entrypoint_facade_reexports_core_helpers() -> None:
    from weiss_rl.eval import (
        final_eval_worker,
        targeted_confirm_core,
        targeted_confirm_entrypoint,
        targeted_confirm_jobs,
        targeted_confirm_opponents,
        targeted_confirm_plan,
        targeted_confirm_summary,
    )

    assert targeted_confirm_entrypoint.main is targeted_confirm_core.main
    assert targeted_confirm_entrypoint._run_targeted_jobs is targeted_confirm_core._run_targeted_jobs
    assert targeted_confirm_entrypoint._targeted_worker is targeted_confirm_core._targeted_worker
    assert targeted_confirm_core.run_final_eval_worker is final_eval_worker.run_final_eval_worker
    assert (
        targeted_confirm_entrypoint.TargetedConfirmPlan
        is targeted_confirm_plan.TargetedConfirmPlan
        is targeted_confirm_jobs.TargetedConfirmPlan
    )
    assert targeted_confirm_entrypoint.DEFAULT_OPPONENTS is targeted_confirm_opponents.DEFAULT_OPPONENTS
    assert targeted_confirm_entrypoint.OPPONENT_SETS is targeted_confirm_opponents.OPPONENT_SETS
    assert targeted_confirm_entrypoint.FAST_LOOP_EXACT_PAIRED_SEEDS is (
        targeted_confirm_opponents.FAST_LOOP_EXACT_PAIRED_SEEDS
    )
    assert targeted_confirm_entrypoint.parse_args is targeted_confirm_plan.parse_args
    assert targeted_confirm_entrypoint.build_arg_parser is targeted_confirm_plan.build_arg_parser
    assert (
        targeted_confirm_entrypoint.build_targeted_confirm_jobs
        is targeted_confirm_plan.build_targeted_confirm_jobs
        is targeted_confirm_jobs.build_targeted_confirm_jobs
    )
    assert (
        targeted_confirm_entrypoint.build_targeted_confirm_summary
        is targeted_confirm_summary.build_targeted_confirm_summary
        is targeted_confirm_plan.build_targeted_confirm_summary
    )
    assert (
        targeted_confirm_entrypoint.prepare_targeted_confirm_plan
        is targeted_confirm_plan.prepare_targeted_confirm_plan
        is targeted_confirm_jobs.prepare_targeted_confirm_plan
    )
    assert (
        targeted_confirm_entrypoint.write_targeted_confirm_summary
        is targeted_confirm_summary.write_targeted_confirm_summary
        is targeted_confirm_plan.write_targeted_confirm_summary
    )
    assert (
        targeted_confirm_entrypoint._targeted_eval_job
        is targeted_confirm_plan._targeted_eval_job
        is targeted_confirm_jobs.targeted_eval_job
    )
    assert (
        targeted_confirm_entrypoint._resolve_paired_seed_file
        is targeted_confirm_plan._resolve_paired_seed_file
        is targeted_confirm_jobs.resolve_paired_seed_file
    )
    assert (
        targeted_confirm_entrypoint._resolve_opponents
        is targeted_confirm_plan._resolve_opponents
        is targeted_confirm_opponents.resolve_opponents
    )
    assert (
        targeted_confirm_entrypoint._require_fast_loop_gate
        is targeted_confirm_plan._require_fast_loop_gate
        is targeted_confirm_opponents.require_fast_loop_gate
    )
    assert (
        targeted_confirm_entrypoint._validate_fast_loop_eval_request
        is targeted_confirm_plan._validate_fast_loop_eval_request
        is targeted_confirm_opponents.validate_fast_loop_eval_request
    )
    assert (
        targeted_confirm_entrypoint._require_exact_opponent_panel
        is targeted_confirm_plan._require_exact_opponent_panel
        is targeted_confirm_opponents.require_exact_opponent_panel
    )
    assert (
        targeted_confirm_entrypoint.validate_targeted_confirm_request
        is targeted_confirm_plan.validate_targeted_confirm_request
        is targeted_confirm_opponents.validate_targeted_confirm_request
    )
    assert (
        targeted_confirm_entrypoint.MAIN_LEAGUE_FULL13_OPPONENTS
        is targeted_confirm_plan.MAIN_LEAGUE_FULL13_OPPONENTS
        is targeted_confirm_opponents.MAIN_LEAGUE_FULL13_OPPONENTS
    )


def test_targeted_confirm_summary_helpers_preserve_worker_and_summary_payloads(tmp_path: Path) -> None:
    from weiss_rl.eval import targeted_confirm_core, targeted_confirm_plan, targeted_confirm_summary

    matchup_dir = tmp_path / "eval" / "confirm16" / "matchups" / "00_policy__vs__01_b2"
    worker_result = targeted_confirm_summary.targeted_worker_summary_from_result(
        {
            "focal_policy_id": "policy_000021",
            "opponent_policy_id": "B2 HeuristicPublic",
            "matchup_dir": matchup_dir,
            "used_paired_seeds": (1, 2, 3),
            "summary": {
                "summary": {
                    "games": 6,
                    "wins": 4,
                    "losses": 1,
                    "draws": 1,
                    "truncations": 2,
                    "engine_errors": 0,
                },
                "uncertainty": {
                    "paired_seed_count": 3,
                    "mean": 0.75,
                    "ci_low": 0.6,
                    "ci_high": 0.9,
                    "prob_gt_half": 0.95,
                },
            },
        }
    )
    seed_file = tmp_path / "report_eval.txt"
    seed_file.write_text("1\n2\n3\n", encoding="utf-8")
    args = SimpleNamespace(
        focal_policy_id="policy_000021",
        paired_seeds=16,
        stack_config=Path("configs/thesis/final_eval.yaml"),
        god_search_mode="same_world_prefix_rollout",
        god_search_top_k=3,
        god_search_rollouts_per_action=2,
        god_search_max_rollout_decisions=4,
        god_search_max_search_decisions_per_game=5,
        god_search_rollout_policy="argmax",
        god_search_no_prefix_verify=True,
        god_search_soft_prefix_fail=True,
        god_search_trace_limit=7,
    )
    plan = SimpleNamespace(
        args=args,
        out_dir=tmp_path / "eval" / "confirm16",
        seed_file_path=seed_file,
        seed_source="report_eval",
        eval_sampling_algorithm="argmax",
        model_sampling_temperature=0.5,
        jobs=[
            {"opponent_policy_id": "B2 HeuristicPublic"},
            {"opponent_policy_id": "B4 HeuristicPublicControl"},
            {"opponent_policy_id": "seed_1"},
            {"opponent_policy_id": "seed_2"},
            {"opponent_policy_id": "seed_3"},
            {"opponent_policy_id": "seed_4"},
        ],
    )
    results_by_opp = {
        "B2 HeuristicPublic": worker_result,
        "B4 HeuristicPublicControl": {**worker_result, "wins": 3, "games": 6},
        "seed_1": {**worker_result, "wins": 2, "games": 6},
        "seed_2": {**worker_result, "wins": 1, "games": 6},
        "seed_3": {**worker_result, "wins": 0, "games": 6},
        "seed_4": {**worker_result, "wins": 6, "games": 6},
    }

    summary = targeted_confirm_summary.build_targeted_confirm_summary(
        plan=plan,
        results_by_opp=results_by_opp,
        started_unix=0.0,
    )

    assert targeted_confirm_core.targeted_worker_summary_from_result is (
        targeted_confirm_summary.targeted_worker_summary_from_result
    )
    assert (
        targeted_confirm_plan.build_targeted_confirm_summary is targeted_confirm_summary.build_targeted_confirm_summary
    )
    assert (
        targeted_confirm_plan.write_targeted_confirm_summary is targeted_confirm_summary.write_targeted_confirm_summary
    )
    assert worker_result == {
        "focal_policy_id": "policy_000021",
        "opponent_policy_id": "B2 HeuristicPublic",
        "paired_seeds": 3,
        "games": 6,
        "wins": 4,
        "losses": 1,
        "draws": 1,
        "mean": 0.75,
        "ci_low": 0.6,
        "ci_high": 0.9,
        "prob_gt_half": 0.95,
        "truncations": 2,
        "engine_errors": 0,
        "summary_path": (matchup_dir / "matchup_summary.json").as_posix(),
        "diagnostics_path": (matchup_dir / "diagnostics.json").as_posix(),
    }
    assert summary["games_per_row"] == 32
    assert summary["seed_file"]["source"] == "report_eval"
    assert summary["overall"] == {"games": 36, "mean": 16 / 36, "wins": 16}
    assert summary["anchor_subset"] == {"games": 30, "mean": 10 / 30, "wins": 10}
    assert summary["legacy_subset"] == {"games": 6, "mean": 1.0, "wins": 6}
    assert summary["god_search"] == {
        "mode": "same_world_prefix_rollout",
        "top_k": 3,
        "rollouts_per_action": 2,
        "max_rollout_decisions": 4,
        "max_search_decisions_per_game": 5,
        "rollout_policy": "argmax",
        "apply_to_focal_only": True,
        "verify_prefix_replay": False,
        "fail_on_prefix_mismatch": False,
        "trace_limit": 7,
    }


def test_targeted_confirm_plan_builds_ordered_jobs_with_shared_output_dir() -> None:
    from weiss_rl.eval.targeted_confirm.plan import build_targeted_confirm_jobs

    args = SimpleNamespace(
        stack_config=Path("configs/thesis/final_eval.yaml"),
        run_dir=Path("runs/demo"),
        snapshot_registry_json=Path("runs/demo/training/snapshots/registry.json"),
        b1_baseline_run_dir=Path("runs/b1"),
        paired_seeds=32,
        bootstrap_samples=1000,
        focal_policy_id="policy_000021",
        god_search_mode="disabled",
    )

    jobs = build_targeted_confirm_jobs(
        args=args,
        paired_seeds=[1, 2, 3],
        opponents=["B2 HeuristicPublic", "B4 HeuristicPublicControl"],
        output_dir=Path("runs/demo/eval/targeted_confirm32"),
    )

    assert [(job["opponent_index"], job["opponent_policy_id"]) for job in jobs] == [
        (1, "B2 HeuristicPublic"),
        (2, "B4 HeuristicPublicControl"),
    ]
    assert jobs[0]["focal_index"] == 0
    assert jobs[0]["focal_policy_id"] == "policy_000021"
    assert jobs[0]["output_dir"] == "runs/demo/eval/targeted_confirm32"
    assert jobs[0]["stage1_paired_seeds"] == 32
    assert jobs[0]["max_paired_seeds"] == 32


def test_targeted_confirm_plan_preparation_uses_seed_source_output_dir_and_eval_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from weiss_rl.eval import targeted_confirm_jobs, targeted_confirm_plan

    seed_file = tmp_path / "custom_seeds.txt"
    seed_file.write_text("10\n11\n12\n13\n", encoding="utf-8")
    fake_stack = SimpleNamespace(
        seed_sets={"custom_eval": seed_file},
        config=SimpleNamespace(
            evaluation=SimpleNamespace(
                eval_sampling_algorithm="model_argmax_pinned_v1",
                model_sampling_temperature=0.25,
            )
        ),
    )
    monkeypatch.setattr(targeted_confirm_jobs, "load_stack_config", lambda _path: fake_stack)
    args = SimpleNamespace(
        stack_config=Path("configs/thesis/final_eval.yaml"),
        run_dir=tmp_path / "run",
        snapshot_registry_json=Path("runs/demo/training/snapshots/registry.json"),
        b1_baseline_run_dir=Path("runs/b1"),
        paired_seed_file=None,
        seed_set="custom_eval",
        paired_seeds=3,
        output_subdir="confirm3_custom",
        bootstrap_samples=99,
        focal_policy_id="policy_000021",
        god_search_mode="disabled",
    )

    plan = targeted_confirm_plan.prepare_targeted_confirm_plan(
        args,
        opponents=["B2 HeuristicPublic", "B4 HeuristicPublicControl"],
    )

    assert plan.__class__ is targeted_confirm_jobs.TargetedConfirmPlan
    assert plan.opponents == ["B2 HeuristicPublic", "B4 HeuristicPublicControl"]
    assert plan.paired_seeds == [10, 11, 12]
    assert plan.seed_file_path == seed_file
    assert plan.seed_source == "custom_eval"
    assert plan.out_dir == tmp_path / "run" / "eval" / "confirm3_custom"
    assert plan.out_dir.is_dir()
    assert plan.eval_sampling_algorithm == "model_argmax_pinned_v1"
    assert plan.model_sampling_temperature == 0.25
    assert [(job["opponent_index"], job["opponent_policy_id"]) for job in plan.jobs] == [
        (1, "B2 HeuristicPublic"),
        (2, "B4 HeuristicPublicControl"),
    ]
    assert plan.jobs[0]["paired_seeds"] == [10, 11, 12]
    assert plan.jobs[0]["bootstrap_samples"] == 99
    assert plan.jobs[0]["output_dir"] == (tmp_path / "run" / "eval" / "confirm3_custom").as_posix()


def test_targeted_confirm_eval_resolves_explicit_or_named_seed_file(tmp_path: Path) -> None:
    module = _load_script_module("targeted_confirm_eval")
    explicit = tmp_path / "explicit.txt"
    explicit.write_text("1\n2\n", encoding="utf-8")
    named = tmp_path / "named.txt"
    named.write_text("3\n4\n", encoding="utf-8")
    stack = SimpleNamespace(seed_sets={"report_eval": named, "hardneg_repair_train": named})

    explicit_path, explicit_source = module._resolve_paired_seed_file(
        SimpleNamespace(paired_seed_file=explicit, seed_set="report_eval"),
        stack,
    )
    named_path, named_source = module._resolve_paired_seed_file(
        SimpleNamespace(paired_seed_file=None, seed_set="hardneg_repair_train"),
        stack,
    )

    assert explicit_path == explicit.resolve()
    assert explicit_source == "explicit"
    assert named_path == named
    assert named_source == "hardneg_repair_train"


def test_targeted_confirm_eval_resolves_main_league_sentinel_opponent_set() -> None:
    module = _load_script_module("targeted_confirm_eval")

    sentinel = module._resolve_opponents(SimpleNamespace(opponent=[], opponent_set="main_league_sentinel"))
    explicit = module._resolve_opponents(
        SimpleNamespace(opponent=[" B2 HeuristicPublic ", "seed_custom"], opponent_set="main_league_sentinel")
    )

    assert sentinel == module.MAIN_LEAGUE_SENTINEL_OPPONENTS
    assert sentinel[:2] == ["B2 HeuristicPublic", "B4 HeuristicPublicControl"]
    assert "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004" in sentinel
    assert explicit == ["B2 HeuristicPublic", "seed_custom"]


def test_targeted_confirm_eval_resolves_current_main_league_full13_opponent_set() -> None:
    module = _load_script_module("targeted_confirm_eval")

    full13 = module._resolve_opponents(SimpleNamespace(opponent=[], opponent_set="main_league_full13"))

    assert full13 == module.MAIN_LEAGUE_FULL13_OPPONENTS
    assert full13[:5] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    ]
    assert len(full13) == 13
    assert "seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025" in full13
    assert "seed_b8c698d26a_seed_c3aac2f9dc_policy_000005" in full13


def test_targeted_confirm_eval_fast_loop_rejects_wrong_sentinel_panel() -> None:
    module = _load_script_module("targeted_confirm_eval")

    try:
        module._validate_fast_loop_eval_request(
            stage="sentinel",
            paired_seeds=16,
            opponents=["B2 HeuristicPublic"],
        )
    except SystemExit as exc:
        assert "main_league_sentinel" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SystemExit")


def test_targeted_confirm_eval_fast_loop_rejects_wrong_confirm_seed_depth() -> None:
    module = _load_script_module("targeted_confirm_eval")

    try:
        module._validate_fast_loop_eval_request(
            stage="confirm128",
            paired_seeds=64,
            opponents=module.MAIN_LEAGUE_FULL13_OPPONENTS,
        )
    except SystemExit as exc:
        assert "exactly 128 paired seeds" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SystemExit")


def test_targeted_confirm_eval_fast_loop_accepts_full13_confirm64_request() -> None:
    module = _load_script_module("targeted_confirm_eval")

    assert (
        module._validate_fast_loop_eval_request(
            stage="full_confirm64",
            paired_seeds=64,
            opponents=module.MAIN_LEAGUE_FULL13_OPPONENTS,
        )
        is None
    )


def test_targeted_confirm_eval_fast_loop_publish_requires_full13_panel() -> None:
    module = _load_script_module("targeted_confirm_eval")

    try:
        module._validate_fast_loop_eval_request(
            stage="publish",
            paired_seeds=256,
            opponents=module.MAIN_LEAGUE_SENTINEL_OPPONENTS,
        )
    except SystemExit as exc:
        assert "main_league_full13" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected SystemExit")
