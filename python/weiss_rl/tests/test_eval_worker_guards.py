from __future__ import annotations

import os
import subprocess
import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script_module(name: str):
    module_path = REPO_ROOT / "python" / "scripts" / f"{name}.py"
    spec = util.spec_from_file_location(f"test_{name}_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
            "python/scripts/targeted_confirm_eval.py",
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
            "python/scripts/parallel_final_eval.py",
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
            "python/scripts/targeted_confirm_eval.py",
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
            "python/scripts/parallel_final_eval.py",
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
    module = _load_script_module("parallel_final_eval")
    layout = SimpleNamespace(final_eval_dir=Path("runs/demo/eval/final_eval"))

    assert module._worker_output_dir({"output_dir": "runs/demo/eval/confirm16"}, layout=layout) == Path(
        "runs/demo/eval/confirm16"
    )
    assert module._worker_output_dir({}, layout=layout) == Path("runs/demo/eval/final_eval")


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
