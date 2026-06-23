from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace


def test_parallel_final_eval_worker_accepts_noncanonical_output_dir() -> None:
    from weiss_rl.eval.final.worker_runtime import run_final_eval_worker, worker_output_dir

    layout = SimpleNamespace(final_eval_dir=Path("runs/demo/eval/final_eval"))

    assert callable(run_final_eval_worker)
    assert worker_output_dir({"output_dir": "runs/demo/eval/confirm16"}, layout=layout) == Path(
        "runs/demo/eval/confirm16"
    )
    assert worker_output_dir({}, layout=layout) == Path("runs/demo/eval/final_eval")


def test_parallel_final_eval_entrypoint_exposes_only_cli_main() -> None:
    from weiss_rl.eval import parallel_final_eval_core, parallel_final_eval_entrypoint

    assert parallel_final_eval_entrypoint.main is parallel_final_eval_core.main
    retired_helper_exports = {
        "ParallelFinalEvalPlan",
        "_load_json",
        "_run_jobs",
        "_worker",
        "_worker_output_dir",
        "build_arg_parser",
        "build_parallel_final_eval_jobs",
        "parse_args",
        "prepare_parallel_final_eval_plan",
        "validate_parallel_final_eval_request",
        "write_parallel_final_eval_artifacts",
    }

    assert parallel_final_eval_entrypoint.__all__ == ["main"]
    assert not any(hasattr(parallel_final_eval_entrypoint, name) for name in retired_helper_exports)


def test_parallel_final_eval_core_does_not_reexport_worker_helpers() -> None:
    from weiss_rl.eval import parallel_final_eval_core

    assert not hasattr(parallel_final_eval_core, "_load_json")
    assert not hasattr(parallel_final_eval_core, "_worker")
    assert not hasattr(parallel_final_eval_core, "_worker_output_dir")


def test_final_eval_worker_facade_stays_removed() -> None:
    assert find_spec("weiss_rl.eval.final.worker") is None


def test_eval_root_does_not_export_final_eval_worker_module_aliases() -> None:
    import weiss_rl.eval as eval_package

    assert not hasattr(eval_package, "final_eval_worker")
    assert not hasattr(eval_package, "final_eval_worker_runtime")


def test_final_eval_worker_uses_unique_policy_resolution_ids_for_diagonal_jobs() -> None:
    from weiss_rl.eval.final.worker_runtime import unique_policy_resolution_ids

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

    from weiss_rl.eval.final import worker_runtime as final_eval_worker_runtime

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
