from __future__ import annotations

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

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module.sys, "executable", "C:/Python/python.exe")

    module.main()

    assert observed == [
        (["C:/Python/python.exe", "python/scripts/check_core_placeholders.py"], REPO_ROOT, True),
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
                "python/scripts/thesis_run.py",
                "python/scripts/eval.py",
                "python/scripts/play_vs_model.py",
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
                "python/scripts/thesis_run.py",
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
                "python/scripts/thesis_run.py",
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
                "python/scripts/thesis_run.py",
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
