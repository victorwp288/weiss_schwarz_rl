from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from weiss_rl.eval import build_paper_readiness_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "python" / "scripts"
WORKSPACE_ROOT = REPO_ROOT.parent


def _load_script_module(script_name: str):
    script_path = SCRIPTS_ROOT / script_name
    spec = importlib.util.spec_from_file_location(f"test_script_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "package_module"),
    [
        ("god_search_scorecard.py", "weiss_rl.experiments.god_search_scorecard_entrypoint"),
        ("league_progress_summary.py", "weiss_rl.experiments.league_progress_summary_entrypoint"),
        ("main_league_fast_loop_gate.py", "weiss_rl.experiments.main_league_fast_loop_gate_entrypoint"),
        ("main_league_frontier_audit.py", "weiss_rl.experiments.main_league_frontier_audit_entrypoint"),
        ("main_league_frontier_scorecard.py", "weiss_rl.experiments.main_league_frontier_scorecard_entrypoint"),
        ("main_league_live_probe_gate.py", "weiss_rl.experiments.main_league_live_probe_gate_entrypoint"),
        ("main_league_multiobjective_gate.py", "weiss_rl.experiments.main_league_multiobjective_gate_entrypoint"),
        ("paired_flip_targets_gate.py", "weiss_rl.experiments.paired_flip_targets_gate_entrypoint"),
        ("paired_outcome_compare_gate.py", "weiss_rl.experiments.paired_outcome_compare_gate_entrypoint"),
        (
            "paired_outcome_preference_mechanistic_gate.py",
            "weiss_rl.experiments.paired_outcome_preference_mechanistic_gate_entrypoint",
        ),
        ("paired_swing_mechanistic_gate.py", "weiss_rl.experiments.paired_swing_mechanistic_gate_entrypoint"),
        ("trajectory_policy_drift_gate.py", "weiss_rl.experiments.trajectory_policy_drift_gate_entrypoint"),
    ],
)
def test_gate_and_scorecard_script_shims_delegate_to_package_entrypoints(
    script_name: str,
    package_module: str,
) -> None:
    script_module = _load_script_module(script_name)
    package = importlib.import_module(package_module)

    assert script_module.main is package.main


@pytest.mark.parametrize(
    ("script_name", "package_module"),
    [
        ("paired_flip_targets.py", "weiss_rl.experiments.paired_flip_targets_entrypoint"),
        ("paired_outcome_compare.py", "weiss_rl.experiments.paired_outcome_compare_entrypoint"),
        ("paired_outcome_overlap_report.py", "weiss_rl.experiments.paired_outcome_overlap_report_entrypoint"),
        (
            "paired_outcome_preference_decisions.py",
            "weiss_rl.experiments.paired_outcome_preference_decisions_entrypoint",
        ),
        (
            "paired_outcome_preference_edge_margins.py",
            "weiss_rl.experiments.paired_outcome_preference_edge_margins_entrypoint",
        ),
        (
            "paired_outcome_preference_margins.py",
            "weiss_rl.experiments.paired_outcome_preference_margins_entrypoint",
        ),
        (
            "paired_outcome_preference_row_guard.py",
            "weiss_rl.experiments.paired_outcome_preference_row_guard_entrypoint",
        ),
        (
            "paired_outcome_preference_span_audit.py",
            "weiss_rl.experiments.paired_outcome_preference_span_audit_entrypoint",
        ),
        (
            "paired_outcome_preference_surface_clusters.py",
            "weiss_rl.experiments.paired_outcome_preference_surface_clusters_entrypoint",
        ),
        (
            "paired_outcome_preference_surface_prototypes.py",
            "weiss_rl.experiments.paired_outcome_preference_surface_prototypes_entrypoint",
        ),
        ("paired_swing_context_margins.py", "weiss_rl.experiments.paired_swing_context_margins_entrypoint"),
        ("paired_swing_report.py", "weiss_rl.experiments.paired_swing_report_entrypoint"),
        ("trajectory_policy_drift.py", "weiss_rl.experiments.trajectory_policy_drift_entrypoint"),
    ],
)
def test_analysis_report_script_shims_delegate_to_package_entrypoints(
    script_name: str,
    package_module: str,
) -> None:
    script_module = _load_script_module(script_name)
    package = importlib.import_module(package_module)

    assert script_module.main is package.main


@pytest.mark.parametrize(
    ("script_name", "package_module"),
    [
        ("paper_readiness_check.py", "weiss_rl.eval.paper_readiness_check_entrypoint"),
        ("write_paper_readiness_fixture.py", "weiss_rl.eval.paper_readiness_fixture_entrypoint"),
        ("publish_checkpoint_snapshot.py", "weiss_rl.training.checkpoint_publish_entrypoint"),
        ("interpolate_checkpoints.py", "weiss_rl.training.checkpoint_interpolation_entrypoint"),
        ("metagame.py", "weiss_rl.metagame.metagame_entrypoint"),
        ("augment_eval_snapshot_registry.py", "weiss_rl.experiments.eval_registry_augmentation_entrypoint"),
        ("select_b1_candidate.py", "weiss_rl.experiments.select_b1_candidate_entrypoint"),
        ("teacher_action_overrides_from_inspections.py", "weiss_rl.experiments.teacher_action_overrides_entrypoint"),
        ("targeted_confirm_prefix_summary.py", "weiss_rl.experiments.targeted_confirm_prefix_entrypoint"),
        ("merge_targeted_confirm_summaries.py", "weiss_rl.experiments.targeted_confirm_merge_entrypoint"),
        ("make_god_search_figures.py", "weiss_rl.experiments.god_search_figures_entrypoint"),
        ("make_main_search_extra_figures.py", "weiss_rl.experiments.main_search_extra_figures_entrypoint"),
        ("replay_inspector.py", "weiss_rl.replay.inspector_entrypoint"),
        ("reward_component_probe.py", "weiss_rl.diagnostics.reward_component_probe_entrypoint"),
        ("trajectory_audit_compare.py", "weiss_rl.diagnostics.trajectory_audit_compare_entrypoint"),
    ],
)
def test_utility_workflow_script_shims_delegate_to_package_entrypoints(
    script_name: str,
    package_module: str,
) -> None:
    script_module = _load_script_module(script_name)
    package = importlib.import_module(package_module)

    assert script_module.main is package.main


@pytest.mark.parametrize(
    ("script_name", "package_module"),
    [
        ("champion_hardneg_trajectory_bc_dataset.py", "weiss_rl.experiments.champion_hardneg_trajectory_bc_entrypoint"),
        ("paired_flip_trajectory_bc_dataset.py", "weiss_rl.experiments.paired_flip_trajectory_bc_entrypoint"),
        ("paired_outcome_contrastive_dataset.py", "weiss_rl.experiments.paired_outcome_contrastive_entrypoint"),
        ("paired_outcome_preference_dataset.py", "weiss_rl.experiments.paired_outcome_preference_dataset_entrypoint"),
        (
            "filter_paired_outcome_preference_dataset.py",
            "weiss_rl.experiments.paired_outcome_preference_dataset_filter_entrypoint",
        ),
        (
            "filter_paired_outcome_preference_rows.py",
            "weiss_rl.experiments.paired_outcome_preference_row_filter_entrypoint",
        ),
        (
            "filter_paired_outcome_preference_spans.py",
            "weiss_rl.experiments.paired_outcome_preference_span_filter_entrypoint",
        ),
        ("filter_paired_swing_dataset.py", "weiss_rl.experiments.paired_swing_filter_entrypoint"),
        ("paired_swing_conflict_report.py", "weiss_rl.experiments.paired_swing_conflict_report_entrypoint"),
        ("replay_trajectory_bc_dataset.py", "weiss_rl.replay.trajectory_bc_dataset_entrypoint"),
        ("merge_replay_trajectory_bc_datasets.py", "weiss_rl.replay.trajectory_bc_merge_entrypoint"),
    ],
)
def test_dataset_script_shims_delegate_to_package_entrypoints(
    script_name: str,
    package_module: str,
) -> None:
    script_module = _load_script_module(script_name)
    package = importlib.import_module(package_module)

    assert script_module.main is package.main


@pytest.mark.parametrize(
    ("script_name", "package_module"),
    [
        ("trajectory_bc_warmstart.py", "weiss_rl.training.trajectory_bc_warmstart_entrypoint"),
        ("paired_swing_warmstart.py", "weiss_rl.training.paired_swing_warmstart_entrypoint"),
        (
            "paired_outcome_preference_warmstart.py",
            "weiss_rl.training.paired_outcome_preference_warmstart_entrypoint",
        ),
    ],
)
def test_warmstart_script_shims_delegate_to_package_entrypoints(
    script_name: str,
    package_module: str,
) -> None:
    script_module = _load_script_module(script_name)
    package = importlib.import_module(package_module)

    assert script_module.main is package.main


@pytest.mark.parametrize(
    ("script_name", "package_module"),
    [
        ("artifact_scan.py", "weiss_rl.diagnostics.artifact_scan_entrypoint"),
        ("bias_checkpoint_family_logits.py", "weiss_rl.training.family_logit_bias_entrypoint"),
        ("check_core_placeholders.py", "weiss_rl.diagnostics.core_placeholder_check_entrypoint"),
        ("heuristic_sanity_scan.py", "weiss_rl.diagnostics.heuristic_sanity_scan_entrypoint"),
        ("learning_run_compare.py", "weiss_rl.experiments.learning_run_compare_entrypoint"),
        ("play_vs_model.py", "weiss_rl.human_play.play_vs_model_entrypoint"),
        ("profile_structured_hotpaths.py", "weiss_rl.diagnostics.profile_structured_hotpaths_entrypoint"),
        ("profile_train_job.py", "weiss_rl.diagnostics.profile_train_job_entrypoint"),
        ("structured_v2_baseline.py", "weiss_rl.experiments.structured_v2_baseline_entrypoint"),
        ("structured_v2_campaign.py", "weiss_rl.experiments.structured_v2_campaign_entrypoint"),
        ("verify_repo.py", "weiss_rl.workflows.verify_repo_entrypoint"),
    ],
)
def test_profile_play_and_verification_script_shims_delegate_to_package_entrypoints(
    script_name: str,
    package_module: str,
) -> None:
    script_module = _load_script_module(script_name)
    package = importlib.import_module(package_module)

    assert script_module.main is package.main


@pytest.mark.parametrize(
    ("script_name", "package_module"),
    [
        ("b2_disagreement_audit.py", "weiss_rl.diagnostics.b2_disagreement_audit"),
        ("guarded_league_bootstrap.py", "weiss_rl.experiments.guarded_league_bootstrap_entrypoint"),
        ("learning_progress_diagnostic.py", "weiss_rl.diagnostics.learning_progress"),
        ("make_figures.py", "weiss_rl.workflows.figures_entrypoint"),
        ("segmented_b1_guided_bootstrap.py", "weiss_rl.experiments.segmented_b1_guided_bootstrap_entrypoint"),
        ("targeted_confirm_eval.py", "weiss_rl.eval.targeted_confirm_entrypoint"),
        ("thesis_workflow.py", "weiss_rl.cli"),
    ],
)
def test_legacy_workflow_script_shims_delegate_to_package_entrypoints(
    script_name: str,
    package_module: str,
) -> None:
    script_module = _load_script_module(script_name)
    package = importlib.import_module(package_module)

    assert script_module.main is package.main


def test_eval_script_shim_aliases_package_entrypoint_module() -> None:
    module_name = "test_script_eval_alias"
    script_path = SCRIPTS_ROOT / "eval.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    script_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = script_module
    spec.loader.exec_module(script_module)

    package = importlib.import_module("weiss_rl.workflows.eval_entrypoint")

    assert sys.modules[module_name] is package


def test_check_core_placeholders_entrypoint_passes(monkeypatch) -> None:
    module = _load_script_module("check_core_placeholders.py")

    monkeypatch.chdir(REPO_ROOT)

    assert module.main() == 0


def test_verify_repo_entrypoint_runs_release_verification_steps(monkeypatch) -> None:
    module = _load_script_module("verify_repo.py")
    observed: list[tuple[list[str], Path, bool]] = []

    def _fake_run(command: list[str], cwd: Path, check: bool):
        observed.append((command, cwd, check))
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(module._impl.subprocess, "run", _fake_run)
    monkeypatch.setattr(module._impl.sys, "executable", "C:/Python/python.exe")

    module.main()

    assert observed == [
        (
            ["C:/Python/python.exe", "-m", "weiss_rl.diagnostics.core_placeholder_check_entrypoint"],
            REPO_ROOT,
            True,
        ),
        (
            ["C:/Python/python.exe", "-m", "ruff", "check", "python", "tests", "examples", "python/scripts"],
            REPO_ROOT,
            True,
        ),
        (
            [
                "C:/Python/python.exe",
                "-m",
                "ruff",
                "format",
                "--check",
                "python",
                "tests",
                "examples",
                "python/scripts",
            ],
            REPO_ROOT,
            True,
        ),
        (
            [
                "C:/Python/python.exe",
                "-m",
                "mypy",
                "python/weiss_rl/workflows/thesis_wrapper.py",
                "python/weiss_rl/workflows/eval_entrypoint.py",
                "python/weiss_rl/human_play/play_vs_model_entrypoint.py",
            ],
            REPO_ROOT,
            True,
        ),
        (
            [
                "C:/Python/python.exe",
                "-m",
                "vulture",
                "python/weiss_rl",
                "python/scripts",
                "examples",
                "--min-confidence",
                "80",
            ],
            REPO_ROOT,
            True,
        ),
        (["C:/Python/python.exe", "-m", "pytest", "-q", "python/weiss_rl/tests"], REPO_ROOT, True),
        (
            [
                "C:/Python/python.exe",
                "-m",
                "weiss_rl.workflows.thesis_wrapper",
                "--preset",
                "standard",
                "--run-label",
                "standard_surface_ci",
                "--dry-run",
                "--skip-compare",
            ],
            REPO_ROOT,
            True,
        ),
        (
            [
                "C:/Python/python.exe",
                "-m",
                "weiss_rl.workflows.thesis_wrapper",
                "--preset",
                "standard-auto-gpu",
                "--run-label",
                "standard_auto_gpu_surface_ci",
                "--dry-run",
                "--skip-compare",
            ],
            REPO_ROOT,
            True,
        ),
        (
            [
                "C:/Python/python.exe",
                "-m",
                "weiss_rl.workflows.thesis_wrapper",
                "--preset",
                "standard-multideck",
                "--run-label",
                "standard_multideck_surface_ci",
                "--dry-run",
                "--skip-compare",
            ],
            REPO_ROOT,
            True,
        ),
    ]


def test_verify_repo_plan_builder_preserves_release_verification_surface() -> None:
    from weiss_rl.workflows.verify_repo_plan import (
        VerificationRequest,
        build_release_verification_steps,
        build_release_verification_steps_for_request,
        render_verification_plan,
        render_verification_plan_for_request,
        verification_request,
    )

    request = verification_request(repo_root=REPO_ROOT, python_exe="C:/Python/python.exe")
    steps = build_release_verification_steps_for_request(request)
    legacy_steps = build_release_verification_steps(python_exe="C:/Python/python.exe")
    rendered = render_verification_plan(steps)

    assert isinstance(request, VerificationRequest)
    assert steps == legacy_steps
    assert render_verification_plan_for_request(request) == rendered
    assert [label for label, _command in rendered] == [
        "Core placeholder gate",
        "Ruff check",
        "Ruff format check",
        "Mypy",
        "Vulture",
        "Pytest",
        "Standard wrapper dry-run",
        "Standard auto-gpu wrapper dry-run",
        "Standard multideck wrapper dry-run",
    ]
    assert rendered[0][1] == ["C:/Python/python.exe", "-m", "weiss_rl.diagnostics.core_placeholder_check_entrypoint"]
    assert rendered[1][1] == [
        "C:/Python/python.exe",
        "-m",
        "ruff",
        "check",
        "python",
        "tests",
        "examples",
        "python/scripts",
    ]
    assert rendered[3][1] == [
        "C:/Python/python.exe",
        "-m",
        "mypy",
        "python/weiss_rl/workflows/thesis_wrapper.py",
        "python/weiss_rl/workflows/eval_entrypoint.py",
        "python/weiss_rl/human_play/play_vs_model_entrypoint.py",
    ]
    assert rendered[6][1] == [
        "C:/Python/python.exe",
        "-m",
        "weiss_rl.workflows.thesis_wrapper",
        "--preset",
        "standard",
        "--run-label",
        "standard_surface_ci",
        "--dry-run",
        "--skip-compare",
    ]


def test_verify_repo_script_shim_reexports_plan_helpers() -> None:
    module = _load_script_module("verify_repo.py")
    from weiss_rl.workflows import verify_repo_entrypoint
    from weiss_rl.workflows.verify_repo_plan import (
        VerificationRequest,
        VerificationStep,
        build_release_verification_steps,
        build_release_verification_steps_for_request,
        verification_request,
    )

    assert module.VerificationRequest is VerificationRequest
    assert module.VerificationStep is VerificationStep
    assert module.build_release_verification_steps is build_release_verification_steps
    assert module.build_release_verification_steps_for_request is build_release_verification_steps_for_request
    assert module.run_verification_request is verify_repo_entrypoint.run_verification_request
    assert module.run_verification_steps is verify_repo_entrypoint.run_verification_steps
    assert module.verification_request is verification_request


def test_train_entrypoint_helper_resolves_central_actor_torch_threads(monkeypatch) -> None:
    module = _load_script_module("train.py")

    stack = SimpleNamespace(
        config=SimpleNamespace(
            system=SimpleNamespace(actor_device="cpu", actor_torch_threads=16, learner_torch_threads=4)
        )
    )
    central_runtime = SimpleNamespace(_use_process_collectors=False, _use_central_batched_collection=True)
    process_runtime = SimpleNamespace(_use_process_collectors=True, _use_central_batched_collection=False)

    assert module._central_runtime_actor_torch_threads(stack, central_runtime) == 16
    assert module._central_runtime_actor_torch_threads(stack, process_runtime) is None


def test_train_script_shim_reexports_startup_phase_helpers() -> None:
    module = _load_script_module("train.py")
    from weiss_rl.training.train_entrypoint_phases import (
        TrainCliState,
        TrainManifestState,
        TrainStartupState,
        execute_train_run,
        prepare_train_manifest_state,
        prepare_train_startup_state,
        resolve_train_cli_state,
    )

    assert module.TrainCliState is TrainCliState
    assert module.TrainManifestState is TrainManifestState
    assert module.TrainStartupState is TrainStartupState
    assert module.execute_train_run is execute_train_run
    assert module.prepare_train_manifest_state is prepare_train_manifest_state
    assert module.resolve_train_cli_state is resolve_train_cli_state
    assert module.prepare_train_startup_state is prepare_train_startup_state


def test_train_script_shim_uses_shared_script_compat_helpers() -> None:
    module = _load_script_module("train.py")
    from weiss_rl.training import train_entrypoint
    from weiss_rl.workflows import script_compat

    assert module.install_package_entrypoint_exports is script_compat.install_package_entrypoint_exports
    assert module.bind_package_script_api is script_compat.bind_package_script_api
    assert train_entrypoint._SCRIPT_COMPAT_API is module


def test_script_compat_helpers_install_exports_bind_api_and_restore_overrides() -> None:
    from types import ModuleType

    from weiss_rl.workflows.script_compat import (
        bind_package_script_api,
        install_package_entrypoint_exports,
        install_package_override_entrypoint_facade,
        run_package_main_with_script_overrides,
    )

    package = ModuleType("package")
    package.public_value = "from-package"
    package.__private_dunder__ = "hidden"
    script = ModuleType("script")
    target_globals: dict[str, object] = {}

    install_package_entrypoint_exports(target_globals, package)
    bind_package_script_api(package, script)

    assert target_globals["public_value"] == "from-package"
    assert "__private_dunder__" not in target_globals
    assert package._SCRIPT_COMPAT_API is script

    calls: list[object] = []
    package.hook = lambda: "package-hook"
    package.main = lambda: calls.append(package.hook())
    script_globals = {"hook": lambda: "script-hook"}

    run_package_main_with_script_overrides(package, script_globals, ("hook",))

    assert calls == ["script-hook"]
    assert package.hook() == "package-hook"

    facade_globals: dict[str, object] = {}
    install_package_override_entrypoint_facade(facade_globals, package, ("hook",))
    facade_globals["hook"] = lambda: "facade-hook"
    facade_globals["main"]()

    assert facade_globals["public_value"] == "from-package"
    assert calls == ["script-hook", "facade-hook"]
    assert package.hook() == "package-hook"


def test_train_script_shim_reexports_legacy_alias_table() -> None:
    module = _load_script_module("train.py")
    from weiss_rl.training import checkpoint_guard, checkpoint_resolution, train_entrypoint
    from weiss_rl.training.dev_eval import persist_periodic_dev_eval_summary
    from weiss_rl.training.environments import build_ids_eval_env
    from weiss_rl.training.manifest_layout import manifest_actor_device_layout
    from weiss_rl.training.run_metadata import hardware_summary
    from weiss_rl.training.script_entrypoint_current_checkpoint_hooks import EnsureCurrentCheckpointRequest
    from weiss_rl.training.script_entrypoint_dev_eval_hooks import PeriodicDevEvalRequest
    from weiss_rl.training.script_entrypoint_hooks import (
        EnsureCurrentCheckpointRequest as FacadeEnsureCurrentCheckpointRequest,
    )
    from weiss_rl.training.script_entrypoint_promotion_hooks import SnapshotPromotionGateRequest
    from weiss_rl.training.train_entrypoint_aliases import NAMESPACE_ALIAS_FAMILIES
    from weiss_rl.training.train_entrypoint_best_checkpoint_wrappers import install_best_checkpoint_wrappers
    from weiss_rl.training.train_entrypoint_checkpoint_hooks import (
        EnsureNoLeagueBaselineAnchorRequest as FacadeEnsureNoLeagueBaselineAnchorRequest,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_io_hooks import WriteCheckpointRequest
    from weiss_rl.training.train_entrypoint_checkpoint_io_wrappers import install_checkpoint_io_wrappers
    from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks import (
        BuildTrainingLearnerRequest as LifecycleBuildTrainingLearnerRequest,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks import (
        EnsureNoLeagueBaselineAnchorRequest as LifecycleEnsureNoLeagueBaselineAnchorRequest,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks import (
        WriteCheckpointRequest as LifecycleWriteCheckpointRequest,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_wrappers import (
        install_checkpoint_io_wrappers as lifecycle_install_checkpoint_io_wrappers,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_wrappers import (
        install_checkpoint_wrappers as lifecycle_install_checkpoint_wrappers,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_wrappers import (
        install_learner_wrappers as lifecycle_install_learner_wrappers,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_wrappers import (
        install_snapshot_wrappers as lifecycle_install_snapshot_wrappers,
    )
    from weiss_rl.training.train_entrypoint_checkpoint_wrappers import install_checkpoint_wrappers
    from weiss_rl.training.train_entrypoint_compat_exports import install_train_entrypoint_compat_exports
    from weiss_rl.training.train_entrypoint_core_aliases import CORE_NAMESPACE_ALIASES
    from weiss_rl.training.train_entrypoint_core_exports import CORE_COMPAT_EXPORTS
    from weiss_rl.training.train_entrypoint_current_checkpoint_wrapper import install_current_checkpoint_wrapper
    from weiss_rl.training.train_entrypoint_dev_eval_wrappers import install_dev_eval_wrappers
    from weiss_rl.training.train_entrypoint_eval_aliases import EVAL_NAMESPACE_ALIASES
    from weiss_rl.training.train_entrypoint_eval_exports import EVAL_COMPAT_EXPORTS
    from weiss_rl.training.train_entrypoint_guard_aliases import CHECKPOINT_GUARD_ALIASES
    from weiss_rl.training.train_entrypoint_learner_hooks import BuildTrainingLearnerRequest
    from weiss_rl.training.train_entrypoint_learner_wrappers import install_learner_wrappers
    from weiss_rl.training.train_entrypoint_metadata_wrappers import install_metadata_wrappers
    from weiss_rl.training.train_entrypoint_minimal_wrapper import install_minimal_training_wrapper
    from weiss_rl.training.train_entrypoint_namespace import (
        COMPAT_EXPORT_FAMILIES,
    )
    from weiss_rl.training.train_entrypoint_namespace import (
        install_train_entrypoint_aliases as namespace_install_train_entrypoint_aliases,
    )
    from weiss_rl.training.train_entrypoint_namespace import (
        install_train_entrypoint_compat_exports as namespace_install_train_entrypoint_compat_exports,
    )
    from weiss_rl.training.train_entrypoint_promotion_wrapper import install_promotion_wrapper
    from weiss_rl.training.train_entrypoint_runner_wrappers import install_runner_wrappers
    from weiss_rl.training.train_entrypoint_runtime_wrappers import (
        install_best_checkpoint_wrappers as runtime_install_best_checkpoint_wrappers,
    )
    from weiss_rl.training.train_entrypoint_runtime_wrappers import (
        install_current_checkpoint_wrapper as runtime_install_current_checkpoint_wrapper,
    )
    from weiss_rl.training.train_entrypoint_runtime_wrappers import (
        install_dev_eval_wrappers as runtime_install_dev_eval_wrappers,
    )
    from weiss_rl.training.train_entrypoint_runtime_wrappers import (
        install_minimal_training_wrapper as runtime_install_minimal_training_wrapper,
    )
    from weiss_rl.training.train_entrypoint_runtime_wrappers import (
        install_promotion_wrapper as runtime_install_promotion_wrapper,
    )
    from weiss_rl.training.train_entrypoint_script_wrappers import install_script_wrappers
    from weiss_rl.training.train_entrypoint_snapshot_hooks import EnsureNoLeagueBaselineAnchorRequest
    from weiss_rl.training.train_entrypoint_snapshot_wrappers import install_snapshot_wrappers
    from weiss_rl.training.train_entrypoint_training_aliases import TRAINING_NAMESPACE_ALIASES
    from weiss_rl.training.train_entrypoint_training_exports import TRAINING_COMPAT_EXPORTS
    from weiss_rl.training.train_entrypoint_wrappers import install_train_entrypoint_wrappers

    assert train_entrypoint.MinimalRollout is train_entrypoint._TrainingMinimalRollout
    assert module.MinimalRollout is train_entrypoint.MinimalRollout
    assert module.build_train_parser is train_entrypoint.build_train_parser
    assert module._training_paths is train_entrypoint._training_paths_impl
    assert module._checkpoint_candidate_metric is checkpoint_guard.checkpoint_candidate_metric
    assert module._resolve_resume_checkpoint_path is checkpoint_resolution.resolve_resume_checkpoint_path
    assert module._build_ids_eval_env is build_ids_eval_env
    assert module._manifest_actor_device_layout is manifest_actor_device_layout
    assert module._hardware_summary is hardware_summary
    assert module._persist_periodic_dev_eval_summary is persist_periodic_dev_eval_summary
    assert module.install_train_entrypoint_compat_exports is install_train_entrypoint_compat_exports
    assert install_train_entrypoint_compat_exports is namespace_install_train_entrypoint_compat_exports
    assert module.install_train_entrypoint_aliases is namespace_install_train_entrypoint_aliases
    assert module.install_train_entrypoint_wrappers is install_train_entrypoint_wrappers
    assert module._write_checkpoint.__module__ == "weiss_rl.training.train_entrypoint_checkpoint_io_wrappers"
    assert module._build_training_learner.__module__ == "weiss_rl.training.train_entrypoint_learner_wrappers"
    assert module._ensure_noleague_baseline_anchor.__module__ == (
        "weiss_rl.training.train_entrypoint_snapshot_wrappers"
    )
    assert module._git_commit.__module__ == "weiss_rl.training.train_entrypoint_metadata_wrappers"
    assert module._ensure_current_checkpoint.__module__ == "weiss_rl.training.train_entrypoint_runtime_wrappers"
    assert module._run_periodic_dev_eval.__module__ == "weiss_rl.training.train_entrypoint_runtime_wrappers"
    assert module._maybe_rollback_to_best_checkpoint.__module__ == "weiss_rl.training.train_entrypoint_runtime_wrappers"
    assert module._run_snapshot_promotion_gate.__module__ == "weiss_rl.training.train_entrypoint_runtime_wrappers"
    assert module._run_minimal_training.__module__ == "weiss_rl.training.train_entrypoint_runtime_wrappers"
    assert module._PeriodicDevEvalRunner.__name__ == "_PeriodicDevEvalRunner"
    assert install_checkpoint_wrappers.__name__ == "install_checkpoint_wrappers"
    assert install_checkpoint_wrappers.__module__ == "weiss_rl.training.train_entrypoint_checkpoint_wrappers"
    assert install_checkpoint_io_wrappers.__name__ == "install_checkpoint_io_wrappers"
    assert install_learner_wrappers.__name__ == "install_learner_wrappers"
    assert install_snapshot_wrappers.__name__ == "install_snapshot_wrappers"
    assert install_checkpoint_wrappers is lifecycle_install_checkpoint_wrappers
    assert install_checkpoint_io_wrappers is lifecycle_install_checkpoint_io_wrappers
    assert install_learner_wrappers is lifecycle_install_learner_wrappers
    assert install_snapshot_wrappers is lifecycle_install_snapshot_wrappers
    assert WriteCheckpointRequest.__module__ == "weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks"
    assert BuildTrainingLearnerRequest.__module__ == "weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks"
    assert EnsureNoLeagueBaselineAnchorRequest.__module__ == (
        "weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks"
    )
    assert WriteCheckpointRequest is LifecycleWriteCheckpointRequest
    assert BuildTrainingLearnerRequest is LifecycleBuildTrainingLearnerRequest
    assert EnsureNoLeagueBaselineAnchorRequest is LifecycleEnsureNoLeagueBaselineAnchorRequest
    assert FacadeEnsureNoLeagueBaselineAnchorRequest is EnsureNoLeagueBaselineAnchorRequest
    assert EnsureCurrentCheckpointRequest.__module__ == "weiss_rl.training.script_entrypoint_current_checkpoint_hooks"
    assert PeriodicDevEvalRequest.__module__ == "weiss_rl.training.script_entrypoint_dev_eval_hooks"
    assert SnapshotPromotionGateRequest.__module__ == "weiss_rl.training.script_entrypoint_promotion_hooks"
    assert FacadeEnsureCurrentCheckpointRequest is EnsureCurrentCheckpointRequest
    assert install_current_checkpoint_wrapper.__name__ == "install_current_checkpoint_wrapper"
    assert install_dev_eval_wrappers.__name__ == "install_dev_eval_wrappers"
    assert install_best_checkpoint_wrappers.__name__ == "install_best_checkpoint_wrappers"
    assert install_promotion_wrapper.__name__ == "install_promotion_wrapper"
    assert install_current_checkpoint_wrapper is runtime_install_current_checkpoint_wrapper
    assert install_dev_eval_wrappers is runtime_install_dev_eval_wrappers
    assert install_best_checkpoint_wrappers is runtime_install_best_checkpoint_wrappers
    assert install_promotion_wrapper is runtime_install_promotion_wrapper
    assert install_metadata_wrappers.__name__ == "install_metadata_wrappers"
    assert install_minimal_training_wrapper.__name__ == "install_minimal_training_wrapper"
    assert install_minimal_training_wrapper is runtime_install_minimal_training_wrapper
    assert install_runner_wrappers.__name__ == "install_runner_wrappers"
    assert install_script_wrappers.__name__ == "install_script_wrappers"
    assert CORE_COMPAT_EXPORTS["build_train_parser"] is train_entrypoint.build_train_parser
    assert TRAINING_COMPAT_EXPORTS["_training_paths_impl"] is train_entrypoint._training_paths_impl
    assert EVAL_COMPAT_EXPORTS["build_ids_eval_env"] is train_entrypoint.build_ids_eval_env
    assert COMPAT_EXPORT_FAMILIES == (
        CORE_COMPAT_EXPORTS,
        TRAINING_COMPAT_EXPORTS,
        EVAL_COMPAT_EXPORTS,
    )
    assert NAMESPACE_ALIAS_FAMILIES == (
        CORE_NAMESPACE_ALIASES,
        TRAINING_NAMESPACE_ALIASES,
        EVAL_NAMESPACE_ALIASES,
    )
    assert CORE_NAMESPACE_ALIASES["_manifest_scaffold_only_reason"] == "manifest_scaffold_only_reason"
    assert TRAINING_NAMESPACE_ALIASES["_training_paths"] == "_training_paths_impl"
    assert EVAL_NAMESPACE_ALIASES["_build_ids_eval_env"] == "build_ids_eval_env"
    assert CHECKPOINT_GUARD_ALIASES["_checkpoint_candidate_metric"] == "checkpoint_candidate_metric"


def test_train_script_checkpoint_hook_uses_active_script_api(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("train.py")
    observed: dict[str, object] = {}
    training_paths = SimpleNamespace()
    learner = SimpleNamespace()
    stack = SimpleNamespace()
    device = SimpleNamespace()

    def fake_write_checkpoint(**kwargs: object) -> dict[str, object]:
        observed["write_checkpoint"] = kwargs
        return {"checkpoint": kwargs["checkpoint_path"]}

    def fake_ensure_current_checkpoint(*, training_paths, learner, write_checkpoint):
        observed["ensure_current_checkpoint"] = {
            "training_paths": training_paths,
            "learner": learner,
        }
        observed["write_result"] = write_checkpoint(tmp_path / "current.pt")
        return tmp_path / "current.pt"

    monkeypatch.setattr(module, "_write_checkpoint", fake_write_checkpoint)
    monkeypatch.setattr(module, "ensure_current_checkpoint", fake_ensure_current_checkpoint)

    result = module._ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256="spec-hash",
        algorithm="impala",
    )

    assert result == tmp_path / "current.pt"
    assert observed["ensure_current_checkpoint"] == {
        "training_paths": training_paths,
        "learner": learner,
    }
    assert observed["write_checkpoint"] == {
        "checkpoint_path": tmp_path / "current.pt",
        "learner": learner,
        "stack": stack,
        "device": device,
        "spec_hash256": "spec-hash",
        "algorithm": "impala",
    }
    assert observed["write_result"] == {"checkpoint": tmp_path / "current.pt"}


def test_train_script_write_checkpoint_uses_active_script_helpers(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("train.py")
    observed: dict[str, object] = {}
    model = SimpleNamespace()
    learner = SimpleNamespace(model=model)
    stack = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(recurrent_core="gru")))
    device = SimpleNamespace()

    def fake_compute_config_hash256(arg_stack: object) -> str:
        observed["compute_stack"] = arg_stack
        return "config-hash"

    def fake_model_guidance_payload(arg_model: object) -> dict[str, object]:
        observed["guidance_model"] = arg_model
        return {"bias": 0.25}

    def fake_write_minimal_train_checkpoint(**kwargs: object) -> dict[str, object]:
        observed["checkpoint"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(module, "compute_config_hash256", fake_compute_config_hash256)
    monkeypatch.setattr(module, "_model_guidance_payload", fake_model_guidance_payload)
    monkeypatch.setattr(module, "write_minimal_train_checkpoint", fake_write_minimal_train_checkpoint)

    payload = module._write_checkpoint(
        checkpoint_path=tmp_path / "checkpoint.pt",
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256="spec-hash",
        algorithm="impala",
    )

    assert payload == {"ok": True}
    assert observed["compute_stack"] is stack
    assert observed["guidance_model"] is model
    assert observed["checkpoint"] == {
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "learner": learner,
        "device": device,
        "config_hash256": "config-hash",
        "spec_hash256": "spec-hash",
        "algorithm": "impala",
        "recurrent_core": "gru",
        "guidance_payload": {"bias": 0.25},
    }


def test_train_script_snapshot_registry_entry_uses_active_script_guidance(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("train.py")
    observed: dict[str, object] = {}
    stack = SimpleNamespace()
    training_paths = SimpleNamespace()
    device = SimpleNamespace()
    model = SimpleNamespace()
    model_state_dict = {"weight": object()}

    def fake_model_guidance_payload(arg_model: object) -> dict[str, object]:
        observed["guidance_model"] = arg_model
        return {
            "public_heuristic_logit_bias_scale": 0.25,
            "public_heuristic_actor_logit_bias_scale": 0.5,
        }

    def fake_persist_snapshot_registry_entry(**kwargs: object) -> str:
        observed["registry"] = kwargs
        return "policy-v7"

    monkeypatch.setattr(module, "_model_guidance_payload", fake_model_guidance_payload)
    monkeypatch.setattr(module, "persist_snapshot_registry_entry", fake_persist_snapshot_registry_entry)

    result = module._persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path / "run",
        checkpoint_path=tmp_path / "checkpoint.pt",
        model_state_dict=model_state_dict,
        config_hash256="config-hash",
        device=device,
        update=12,
        policy_version=7,
        model=model,
    )

    assert result == "policy-v7"
    assert observed["guidance_model"] is model
    assert observed["registry"] == {
        "stack": stack,
        "training_paths": training_paths,
        "run_dir": tmp_path / "run",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "model_state_dict": model_state_dict,
        "config_hash256": "config-hash",
        "device": device,
        "update": 12,
        "policy_version": 7,
        "public_heuristic_logit_bias_scale": 0.25,
        "public_heuristic_actor_logit_bias_scale": 0.5,
    }


def test_train_script_git_metadata_helpers_use_active_script_api(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("train.py")
    observed: dict[str, object] = {}
    repo = tmp_path / "repo"

    def fake_repo_root(source_file: Path) -> Path:
        observed["repo_source_file"] = source_file
        return repo

    def fake_git_output(args: list[str], *, cwd: Path) -> str:
        observed["git_output"] = {"args": args, "cwd": cwd}
        return "git-output"

    def fake_git_commit(*, cwd: Path) -> str:
        observed["git_commit"] = cwd
        return "abc123"

    def fake_git_dirty(*, cwd: Path) -> bool:
        observed["git_dirty"] = cwd
        return True

    monkeypatch.setattr(module, "repo_root", fake_repo_root)
    monkeypatch.setattr(module, "git_output", fake_git_output)
    monkeypatch.setattr(module, "git_commit", fake_git_commit)
    monkeypatch.setattr(module, "git_dirty", fake_git_dirty)

    assert module._repo_root() == repo
    assert module._git_output(["status", "--short"]) == "git-output"
    assert module._git_commit() == "abc123"
    assert module._git_dirty() is True

    assert Path(cast(Path, observed["repo_source_file"])).name == "train_entrypoint.py"
    assert observed["git_output"] == {"args": ["status", "--short"], "cwd": repo}
    assert observed["git_commit"] == repo
    assert observed["git_dirty"] == repo


def test_train_script_noleague_anchor_hook_uses_active_script_callbacks(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("train.py")
    observed: dict[str, object] = {}
    sentinels = {
        "write_checkpoint": object(),
        "import_anchor": object(),
        "model_guidance": object(),
        "write_snapshot": object(),
        "experiment_role": object(),
    }
    stack = SimpleNamespace()
    training_paths = SimpleNamespace()
    learner = SimpleNamespace()
    device = SimpleNamespace()

    def fake_ensure_noleague_baseline_anchor(**kwargs: object) -> str:
        observed["anchor"] = kwargs
        return "b1_noleague_baseline"

    monkeypatch.setattr(module, "ensure_noleague_baseline_anchor", fake_ensure_noleague_baseline_anchor)
    monkeypatch.setattr(module, "_write_checkpoint", sentinels["write_checkpoint"])
    monkeypatch.setattr(module, "_import_noleague_baseline_anchor", sentinels["import_anchor"])
    monkeypatch.setattr(module, "_model_guidance_payload", sentinels["model_guidance"])
    monkeypatch.setattr(module, "_write_snapshot_artifact", sentinels["write_snapshot"])
    monkeypatch.setattr(module, "_experiment_role", sentinels["experiment_role"])

    result = module._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path / "run",
        learner=learner,
        device=device,
        config_hash256="config-hash",
        spec_hash256="spec-hash",
        baseline_run_dir=tmp_path / "b1",
        permit_current_run_alias=True,
        source_checkpoint_path=tmp_path / "source.pt",
        update=12,
    )

    assert result == "b1_noleague_baseline"
    assert observed["anchor"] == {
        "stack": stack,
        "training_paths": training_paths,
        "run_dir": tmp_path / "run",
        "learner": learner,
        "device": device,
        "config_hash256": "config-hash",
        "spec_hash256": "spec-hash",
        "baseline_run_dir": tmp_path / "b1",
        "permit_current_run_alias": True,
        "source_checkpoint_path": tmp_path / "source.pt",
        "update": 12,
        "write_checkpoint_fn": sentinels["write_checkpoint"],
        "import_noleague_baseline_anchor_fn": sentinels["import_anchor"],
        "model_guidance_payload_fn": sentinels["model_guidance"],
        "write_snapshot_artifact_fn": sentinels["write_snapshot"],
        "experiment_role_fn": sentinels["experiment_role"],
    }


def test_train_entrypoint_applies_profile_flags_before_hashing() -> None:
    module = _load_script_module("train.py")
    stack = module.load_stack_config(REPO_ROOT / "configs" / "presets" / "typed_thesis_locked.yaml")

    base_hash = module.compute_config_hash256(stack)
    updated = module._apply_training_flag_overrides(
        stack,
        enable_profile_timers=True,
        enable_torch_profiler=True,
    )

    assert updated.config.training is not None
    assert updated.config.training.profile_timers is True
    assert updated.config.training.torch_profiler is True
    assert module.compute_config_hash256(updated) != base_hash


def test_train_metadata_helpers_preserve_manifest_shapes(tmp_path: Path) -> None:
    module = _load_script_module("train.py")
    stack = module.load_stack_config(REPO_ROOT / "configs" / "presets" / "typed_thesis_locked.yaml")

    relative = module._manifest_source_path(
        REPO_ROOT / "configs" / "presets" / "typed_thesis_locked.yaml", root=REPO_ROOT
    )
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"ok": true}\n', encoding="utf-8")

    assert relative == "configs/presets/typed_thesis_locked.yaml"
    assert module._load_json_object(payload_path, label="metadata") == {"ok": True}
    assert module._evaluation_pinning(stack)["seat_swap"] is True
    assert (
        module._hardware_summary(
            "cuda:0",
            actor_device="cuda:1",
            actor_device_layout=("cuda:1", "cuda:2", "cuda:1"),
        )["actor_device_unique_count"]
        == 2
    )


def test_train_noleague_import_contract_helpers_preserve_role_rules() -> None:
    module = _load_script_module("train.py")

    modern = {"config": {"experiment": {"role": "baseline_noleague"}}}
    legacy = {"training_family_a": {"mode": "b1_no_league"}}
    bad_role = {"config": {"experiment": {"role": "league_training"}}}

    assert module._config_marks_noleague_baseline(modern) is True
    assert module._config_marks_noleague_baseline(legacy) is True
    assert module._config_marks_noleague_baseline(bad_role) is False
    module._assert_noleague_baseline_config(modern)
    module._assert_noleague_baseline_config(legacy)
    with pytest.raises(RuntimeError, match="baseline_noleague"):
        module._assert_noleague_baseline_config(bad_role)


def test_train_cli_parser_preserves_public_defaults_and_aliases() -> None:
    from weiss_rl.training.cli import build_train_parser

    parser = build_train_parser()
    args = parser.parse_args(
        [
            "--stack-config",
            "configs/stack_smoke.yaml",
            "--run-id",
            "legacy_alias",
            "--config-override",
            "training.optimizer.learning_rate=0.0001",
        ]
    )

    assert args.stack_config == Path("configs/stack_smoke.yaml")
    assert args.run_id_alias == "legacy_alias"
    assert args.config_override == ["training.optimizer.learning_rate=0.0001"]
    assert args.num_envs is None
    assert args.unroll_length is None
    assert args.max_updates == 1
    assert args.runtime_mode is None


def test_train_entrypoint_resolves_cuda_auto_to_first_visible_gpu(monkeypatch) -> None:
    module = _load_script_module("train.py")

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 4)
    stack = SimpleNamespace(config=SimpleNamespace(system=SimpleNamespace(learner_device="cuda:auto")))

    resolved = module._resolve_device(stack, "")

    assert resolved == module.torch.device("cuda:0")


def test_launch_sweep_compare_entrypoint_facades_use_split_cli_helpers() -> None:
    from weiss_rl.experiments import (
        launch_entrypoint_runtime,
        launch_experiments_cli,
        launch_experiments_entrypoint,
        sweep_experiments_cli,
        sweep_experiments_entrypoint,
    )
    from weiss_rl.workflows import compare_runs_cli, compare_runs_entrypoint, script_compat

    launch_script = _load_script_module("launch_experiments.py")
    sweep_script = _load_script_module("sweep_experiments.py")
    compare_script = _load_script_module("compare_runs.py")

    assert launch_experiments_entrypoint.build_launch_experiments_parser is (
        launch_experiments_cli.build_launch_experiments_parser
    )
    assert launch_experiments_entrypoint.run_launch_experiments_from_args is (
        launch_experiments_cli.run_launch_experiments_from_args
    )
    assert launch_experiments_entrypoint.launch_summary_line is launch_experiments_cli.launch_summary_line
    assert sweep_experiments_entrypoint.build_sweep_experiments_parser is (
        sweep_experiments_cli.build_sweep_experiments_parser
    )
    assert sweep_experiments_entrypoint.run_sweep_experiments_from_args is (
        sweep_experiments_cli.run_sweep_experiments_from_args
    )
    assert sweep_experiments_entrypoint.sweep_summary_line is sweep_experiments_cli.sweep_summary_line
    assert compare_runs_entrypoint.build_compare_runs_parser is compare_runs_cli.build_compare_runs_parser
    assert compare_runs_entrypoint.CompareRunsRequest is compare_runs_cli.CompareRunsRequest
    assert compare_runs_entrypoint.compare_runs_request_from_args is compare_runs_cli.compare_runs_request_from_args
    assert compare_runs_entrypoint.run_compare_from_args is compare_runs_cli.run_compare_from_args
    assert compare_runs_entrypoint.run_compare_request is compare_runs_cli.run_compare_request
    assert compare_runs_entrypoint.compare_summary_line is compare_runs_cli.compare_summary_line
    assert launch_experiments_entrypoint.run_launch_entrypoint_main is (
        launch_entrypoint_runtime.run_launch_entrypoint_main
    )
    assert sweep_experiments_entrypoint.run_sweep_entrypoint_main is launch_entrypoint_runtime.run_sweep_entrypoint_main
    assert launch_script.install_package_override_entrypoint_facade is (
        script_compat.install_package_override_entrypoint_facade
    )
    assert sweep_script.install_package_override_entrypoint_facade is (
        script_compat.install_package_override_entrypoint_facade
    )
    assert compare_script.install_package_override_entrypoint_facade is (
        script_compat.install_package_override_entrypoint_facade
    )
    assert launch_script.main is not launch_experiments_entrypoint.main
    assert sweep_script.main is not sweep_experiments_entrypoint.main
    assert compare_script.main is not compare_runs_entrypoint.main


def test_launch_entrypoint_runtime_plumbs_common_entrypoint_state(tmp_path: Path) -> None:
    from weiss_rl.experiments.launch_entrypoint_runtime import (
        entrypoint_repo_root,
        run_launch_entrypoint_main,
        run_sweep_entrypoint_main,
        torch_cuda_state,
    )

    entrypoint_file = tmp_path / "repo" / "python" / "weiss_rl" / "experiments" / "entrypoint.py"
    entrypoint_file.parent.mkdir(parents=True)
    entrypoint_file.write_text("", encoding="utf-8")
    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
    launch_args = SimpleNamespace(group_label="launch_group")
    sweep_args = SimpleNamespace(preset="impala_compact")
    observed: dict[str, object] = {"printed": []}

    assert entrypoint_repo_root(entrypoint_file) == tmp_path
    assert torch_cuda_state(torch_module) == (True, 2)

    run_launch_entrypoint_main(
        entrypoint_file=entrypoint_file,
        torch_module=torch_module,
        build_parser_fn=lambda: SimpleNamespace(parse_args=lambda: launch_args),
        run_from_args_fn=lambda args, **kwargs: (
            observed.setdefault("launch", (args, kwargs)) or {"group_label": "unreachable"}
        ),
        summary_line_fn=lambda summary: f"launch:{summary[0].group_label}",
        resolve_devices_fn=object(),
        build_launch_plan_fn=object(),
        execute_launch_plan_fn=object(),
        print_fn=cast(list[str], observed["printed"]).append,
    )
    run_sweep_entrypoint_main(
        entrypoint_file=entrypoint_file,
        torch_module=torch_module,
        build_parser_fn=lambda: SimpleNamespace(parse_args=lambda: sweep_args),
        run_from_args_fn=lambda args, **kwargs: (
            observed.setdefault("sweep_call", (args, kwargs)),
            Path("plan.json"),
        )[0:2],
        summary_line_fn=lambda *, preset, summary, plan_path: f"sweep:{preset}:{plan_path}",
        resolve_devices_fn=object(),
        build_sweep_launch_plan_fn=object(),
        execute_launch_plan_fn=object(),
        print_fn=cast(list[str], observed["printed"]).append,
    )

    launch_call = cast(tuple[object, dict[str, object]], observed["launch"])
    sweep_call = cast(tuple[object, dict[str, object]], observed["sweep_call"])
    assert launch_call[0] is launch_args
    assert launch_call[1]["repo_root"] == tmp_path
    assert launch_call[1]["cuda_available"] is True
    assert launch_call[1]["cuda_count"] == 2
    assert sweep_call[0] is sweep_args
    assert sweep_call[1]["repo_root"] == tmp_path
    assert sweep_call[1]["cuda_available"] is True
    assert sweep_call[1]["cuda_count"] == 2
    assert observed["printed"] == ["launch:launch_group", "sweep:impala_compact:plan.json"]


def test_compare_runs_entrypoint_expands_launch_group_and_deduplicates(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("compare_runs.py")
    run_a = (tmp_path / "runs" / "a").resolve()
    run_b = (tmp_path / "runs" / "b").resolve()
    launch_group_summary = tmp_path / "summary.json"
    launch_group_summary.write_text(
        json.dumps(
            {
                "jobs": [
                    {"expected_run_dir": run_a.as_posix()},
                    {"expected_run_dir": run_b.as_posix()},
                ]
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def _fake_render(*, run_dirs, out_dir):
        observed["run_dirs"] = list(run_dirs)
        observed["out_dir"] = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "dummy.png"
        output.write_bytes(b"png")
        return (output,)

    monkeypatch.setattr(module, "render_benchmark_figures", _fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_runs.py",
            "--run-dir",
            run_a.as_posix(),
            "--launch-group-summary",
            str(launch_group_summary),
        ],
    )

    module.main()

    assert observed["run_dirs"] == [run_a, run_b]
    assert observed["out_dir"] == run_a / "figures" / "benchmark_compare"


def test_compare_runs_request_normalizes_inputs_and_matches_legacy_runtime(tmp_path: Path) -> None:
    from weiss_rl.workflows.compare_runs_cli import (
        CompareRunsRequest,
        compare_runs_request_from_args,
        run_compare_from_args,
        run_compare_request,
    )

    run_a = (tmp_path / "runs" / "a").resolve()
    run_b = (tmp_path / "runs" / "b").resolve()
    out_dir = (tmp_path / "compare_out").resolve()
    launch_group_summary = tmp_path / "summary.json"
    launch_group_summary.write_text(
        json.dumps({"jobs": [{"expected_run_dir": run_a.as_posix()}, {"expected_run_dir": run_b.as_posix()}]}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        run_dir=[run_a.as_posix()],
        launch_group_summary=launch_group_summary,
        out_dir=out_dir,
    )
    observed: list[tuple[list[Path], Path]] = []

    def fake_render(*, run_dirs: list[Path], out_dir: Path) -> tuple[Path, Path]:
        observed.append((run_dirs, out_dir))
        return (out_dir / "a.png", out_dir / "b.png")

    request = compare_runs_request_from_args(args)
    request_result = run_compare_request(request, render_benchmark_figures_fn=fake_render)
    legacy_result = run_compare_from_args(args, render_benchmark_figures_fn=fake_render)

    assert isinstance(request, CompareRunsRequest)
    assert request.run_dirs == (run_a, run_b)
    assert request.out_dir == out_dir
    assert request_result == (2, out_dir)
    assert legacy_result == request_result
    assert observed == [([run_a, run_b], out_dir), ([run_a, run_b], out_dir)]


def test_structured_v2_baseline_entrypoint_writes_contract(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("structured_v2_baseline.py")
    baseline_run_dir = tmp_path / "runs" / "baseline"
    summary_path = baseline_run_dir / "eval" / "dev_eval" / "update_300" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "aggregate_score": 0.375,
                "anchor_scores": {
                    "B0 RandomLegal": 0.6875,
                    "B1 NoLeague baseline": 0.4375,
                    "B2 HeuristicPublic": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    audit_summary = tmp_path / "audit_summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "top_family_pairs": [
                    {"policy_a_family": "main_move", "policy_b_family": "pass", "count": 101},
                    {
                        "policy_a_family": "main_move",
                        "policy_b_family": "main_play_character",
                        "count": 98,
                    },
                ],
                "top_action_label_pairs": [
                    {
                        "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
                        "policy_b_action_label": "pass",
                        "count": 78,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "baseline_contract.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "structured_v2_baseline.py",
            "--baseline-run-dir",
            str(baseline_run_dir),
            "--audit-summary",
            str(audit_summary),
            "--out-json",
            str(out_path),
        ],
    )

    module.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mismatch_baseline"]["main_move_to_pass"] == 101
    assert payload["acceptance_targets"]["u120"]["max_main_move_to_main_play_character"] == 39


def test_structured_v2_campaign_entrypoint_writes_dry_run_summary(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("structured_v2_campaign.py")
    repo_root = tmp_path / "weiss_schwarz_rl"
    stack_config = repo_root / "configs" / "presets" / "typed_structured_v2.yaml"
    stack_config.parent.mkdir(parents=True, exist_ok=True)
    stack_config.write_text("config: {}\n", encoding="utf-8")
    baseline_run_dir = repo_root / "runs" / "baseline_ref"
    summary_path = baseline_run_dir / "eval" / "dev_eval" / "update_300" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "aggregate_score": 0.375,
                "anchor_scores": {
                    "B0 RandomLegal": 0.6875,
                    "B1 NoLeague baseline": 0.4375,
                    "B2 HeuristicPublic": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    audit_summary = tmp_path / "audit_summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "top_family_pairs": [
                    {"policy_a_family": "main_move", "policy_b_family": "pass", "count": 101},
                    {
                        "policy_a_family": "main_move",
                        "policy_b_family": "main_play_character",
                        "count": 98,
                    },
                ],
                "top_action_label_pairs": [
                    {
                        "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
                        "policy_b_action_label": "pass",
                        "count": 78,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "structured_v2_campaign.py",
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--campaign-label",
            "structured_campaign_cli",
            "--seed",
            "7",
            "--frozen-baseline-run-dir",
            str(baseline_run_dir),
            "--frozen-audit-summary",
            str(audit_summary),
            "--dry-run",
        ],
    )

    assert module.main() == 0

    campaign_summary = tmp_path / "runs" / "launch_groups" / "structured_campaign_cli" / "summary.json"
    payload = json.loads(campaign_summary.read_text(encoding="utf-8"))
    assert payload["status"] == "planned"
    assert payload["baseline_contract"]["mismatch_baseline"]["main_move_to_pass"] == 101
    assert [step["label"] for step in payload["steps"]] == [
        "baseline_seed_7",
        "canary_seed_7",
        "audit_seed_7",
    ]
    assert payload["steps"][0]["command"][1:3] == ["-m", "weiss_rl.training.train_entrypoint"]
    assert payload["steps"][1]["command"][1:3] == ["-m", "weiss_rl.training.train_entrypoint"]
    assert payload["steps"][2]["command"][1:3] == ["-m", "weiss_rl.diagnostics.b2_disagreement_audit"]
    assert payload["steps"][0]["command"][payload["steps"][0]["command"].index("--stack-config") + 1].endswith(
        "configs/presets/typed_structured_v2.yaml"
    )
    assert payload["steps"][1]["command"][payload["steps"][1]["command"].index("--b1-baseline-run-dir") + 1].endswith(
        "runs/structured_campaign_cli_b1_seed7_u120"
    )


def test_structured_v2_campaign_entrypoint_facade_uses_split_core_helpers() -> None:
    from weiss_rl.experiments import structured_v2_campaign_core, structured_v2_campaign_entrypoint

    assert structured_v2_campaign_entrypoint._build_parser is (
        structured_v2_campaign_core.build_structured_v2_campaign_parser
    )
    assert structured_v2_campaign_entrypoint._run_label is structured_v2_campaign_core.run_label
    assert structured_v2_campaign_entrypoint._repo_run_dir is structured_v2_campaign_core.repo_run_dir
    assert structured_v2_campaign_entrypoint._command_env is structured_v2_campaign_core.command_env
    assert structured_v2_campaign_entrypoint._write_summary is structured_v2_campaign_core.write_summary
    assert structured_v2_campaign_entrypoint._freeze_baseline_contract is (
        structured_v2_campaign_core.freeze_baseline_contract
    )
    assert structured_v2_campaign_entrypoint.run_structured_v2_campaign_from_args is (
        structured_v2_campaign_core.run_structured_v2_campaign_from_args
    )


def test_structured_v2_campaign_entrypoint_records_failed_step(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("structured_v2_campaign.py")
    repo_root = tmp_path / "weiss_schwarz_rl"
    stack_config = repo_root / "configs" / "presets" / "typed_structured_v2.yaml"
    stack_config.parent.mkdir(parents=True, exist_ok=True)
    stack_config.write_text("config: {}\n", encoding="utf-8")
    baseline_run_dir = repo_root / "runs" / "baseline_ref"
    summary_path = baseline_run_dir / "eval" / "dev_eval" / "update_300" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "aggregate_score": 0.375,
                "anchor_scores": {
                    "B0 RandomLegal": 0.6875,
                    "B1 NoLeague baseline": 0.4375,
                    "B2 HeuristicPublic": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    audit_summary = tmp_path / "audit_summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "top_family_pairs": [
                    {"policy_a_family": "main_move", "policy_b_family": "pass", "count": 101},
                    {
                        "policy_a_family": "main_move",
                        "policy_b_family": "main_play_character",
                        "count": 98,
                    },
                ],
                "top_action_label_pairs": [
                    {
                        "policy_a_action_label": "main_move(from_slot=0, to_slot=2)",
                        "policy_b_action_label": "pass",
                        "count": 78,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=17, cmd=kwargs.get("args", args[0] if args else "boom"))

    monkeypatch.setattr(module.subprocess, "run", _boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "structured_v2_campaign.py",
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--campaign-label",
            "structured_campaign_fail",
            "--seed",
            "7",
            "--frozen-baseline-run-dir",
            str(baseline_run_dir),
            "--frozen-audit-summary",
            str(audit_summary),
        ],
    )

    with pytest.raises(subprocess.CalledProcessError):
        module.main()

    campaign_summary = tmp_path / "runs" / "launch_groups" / "structured_campaign_fail" / "summary.json"
    payload = json.loads(campaign_summary.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failed_step"] == "baseline_seed_7"
    assert payload["failed_seed"] == 7
    assert payload["current_stage"] == "baseline_seed_7"
    assert payload["current_seed"] == 7
    assert payload["steps"][0]["status"] == "failed"


def test_launch_experiments_entrypoint_dry_run_plumbs_devices(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("launch_experiments.py")
    observed: dict[str, object] = {}

    def _fake_resolve_devices(*, requested_devices, cuda_available, cuda_count):
        observed["resolve_devices"] = {
            "requested_devices": requested_devices,
            "cuda_available": cuda_available,
            "cuda_count": cuda_count,
        }
        return ("cpu",)

    def _fake_build_launch_plan(*, group_label, stack_configs, seeds, devices, run_label_prefix, extra_args):
        observed["build_launch_plan"] = {
            "group_label": group_label,
            "stack_configs": list(stack_configs),
            "seeds": list(seeds),
            "devices": tuple(devices),
            "run_label_prefix": run_label_prefix,
            "extra_args": extra_args,
        }
        return object()

    def _fake_execute_launch_plan(*, repo_root, plan, dry_run):
        observed["execute_launch_plan"] = {
            "repo_root": repo_root,
            "plan": plan,
            "dry_run": dry_run,
        }
        return {"group_label": "cli_group", "jobs": [{}, {}], "max_parallel_jobs": 1, "dry_run": True}

    monkeypatch.setattr(module, "resolve_devices", _fake_resolve_devices)
    monkeypatch.setattr(module, "build_launch_plan", _fake_build_launch_plan)
    monkeypatch.setattr(module, "execute_launch_plan", _fake_execute_launch_plan)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_experiments.py",
            "--group-label",
            "cli_group",
            "--stack-config",
            str(tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml"),
            "--seed",
            "7",
            "--train-arg=--max-updates",
            "--train-arg=2",
            "--dry-run",
        ],
    )

    module.main()

    assert observed["resolve_devices"] == {
        "requested_devices": None,
        "cuda_available": False,
        "cuda_count": 0,
    }
    assert observed["build_launch_plan"] == {
        "group_label": "cli_group",
        "stack_configs": [str((tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml").resolve())],
        "seeds": [7],
        "devices": ("cpu",),
        "run_label_prefix": None,
        "extra_args": ["--max-updates", "2"],
    }
    assert observed["execute_launch_plan"] == {
        "repo_root": WORKSPACE_ROOT,
        "plan": cast(dict[str, Any], observed["execute_launch_plan"])["plan"],
        "dry_run": True,
    }


def test_sweep_experiments_entrypoint_writes_plan_and_summary(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("sweep_experiments.py")
    group_dir = WORKSPACE_ROOT / "runs" / "launch_groups" / "cli_sweep"
    if group_dir.exists():
        import shutil

        shutil.rmtree(group_dir)
    observed: dict[str, object] = {}

    def _fake_resolve_devices(*, requested_devices, cuda_available, cuda_count):
        observed["resolve_devices"] = {
            "requested_devices": requested_devices,
            "cuda_available": cuda_available,
            "cuda_count": cuda_count,
        }
        return ("cpu",)

    def _fake_build_sweep_launch_plan(*, preset_id, repo_root, group_label, seeds, devices, train_args):
        observed["build_sweep_launch_plan"] = {
            "preset_id": preset_id,
            "repo_root": repo_root,
            "group_label": group_label,
            "seeds": list(seeds),
            "devices": tuple(devices),
            "train_args": train_args,
        }
        return object(), {"preset_id": preset_id, "jobs": [{"run_label": "job_a"}]}

    def _fake_execute_launch_plan(*, repo_root, plan, dry_run):
        observed["execute_launch_plan"] = {
            "repo_root": repo_root,
            "plan": plan,
            "dry_run": dry_run,
        }
        return {"group_label": "cli_sweep", "jobs": [{}], "max_parallel_jobs": 1, "dry_run": True}

    monkeypatch.setattr(module, "resolve_devices", _fake_resolve_devices)
    monkeypatch.setattr(module, "build_sweep_launch_plan", _fake_build_sweep_launch_plan)
    monkeypatch.setattr(module, "execute_launch_plan", _fake_execute_launch_plan)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_experiments.py",
            "--preset",
            "impala_compact",
            "--group-label",
            "cli_sweep",
            "--seed",
            "3",
            "--train-arg=--max-updates",
            "--train-arg=2",
            "--dry-run",
        ],
    )

    module.main()

    plan_path = group_dir / "sweep_plan.json"
    assert plan_path.is_file()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["preset_id"] == "impala_compact"
    assert observed["build_sweep_launch_plan"] == {
        "preset_id": "impala_compact",
        "repo_root": WORKSPACE_ROOT,
        "group_label": "cli_sweep",
        "seeds": [3],
        "devices": ("cpu",),
        "train_args": ["--max-updates", "2"],
    }
    assert observed["execute_launch_plan"] == {
        "repo_root": WORKSPACE_ROOT,
        "plan": cast(dict[str, Any], observed["execute_launch_plan"])["plan"],
        "dry_run": True,
    }
    import shutil

    shutil.rmtree(group_dir)


def test_write_paper_readiness_fixture_entrypoint_writes_ready_run(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("write_paper_readiness_fixture.py")
    run_dir = tmp_path / "fixture_run"

    monkeypatch.setattr(sys, "argv", ["write_paper_readiness_fixture.py", "--run-dir", str(run_dir)])

    module.main()

    payload = build_paper_readiness_summary(run_dir=run_dir)
    assert payload["passed"] is True
