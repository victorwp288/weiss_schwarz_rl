from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
    spec.loader.exec_module(module)
    return module


def test_check_core_placeholders_entrypoint_passes(monkeypatch) -> None:
    module = _load_script_module("check_core_placeholders.py")

    monkeypatch.chdir(REPO_ROOT)

    assert module.main() == 0


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
        "plan": observed["execute_launch_plan"]["plan"],
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
        "plan": observed["execute_launch_plan"]["plan"],
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
