from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.spec import spec_bundle_hash

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mismatched_sha256(value: str) -> str:
    return ("0" if value[0] != "0" else "1") + value[1:]


def _write_stub_weiss_sim(tmp_path: Path, *, spec_hash: int = 123) -> dict[str, object]:
    bundle: dict[str, object] = {
        "policy_version": 3,
        "spec_hash": spec_hash,
        "observation": {
            "obs_encoding_version": 2,
            "dtype": "i32",
            "obs_len": 512,
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 9,
            "pass_action_id": 8,
        },
    }
    (tmp_path / "weiss_sim.py").write_text(
        "\n".join(
            (
                "def build_info():",
                "    return 'stub-build'",
                "",
                "def db_info():",
                "    return 'stub-db'",
                "",
                "def export_spec_bundle():",
                f"    return {bundle!r}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return bundle


def _copy_repo_configs(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    return tmp_path / "configs" / "rl_stack_locked.yaml"


def _run_entrypoint(
    tmp_path: Path,
    *,
    script_name: str,
    stack_config: Path,
    spec_hash: str,
    run_label: str = "",
    run_id_alias: str = "",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO_ROOT / "python")])
    env["WEISS_SIM_PYTHONPATH"] = str(tmp_path)
    env["WEISS_SIM_PYTHON"] = sys.executable

    command = [sys.executable, str(REPO_ROOT / "python" / "scripts" / script_name), "--stack-config", str(stack_config)]
    if spec_hash:
        command.extend(["--spec-hash", spec_hash])
    if run_label:
        command.extend(["--run-label", run_label])
    if run_id_alias:
        command.extend(["--run-id", run_id_alias])
    if extra_args:
        command.extend(extra_args)

    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)


def test_train_entrypoint_fails_fast_on_runtime_spec_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="999",
        run_label="mismatch_run",
    )

    assert result.returncode != 0
    assert "Spec mismatch" in result.stderr


def test_train_entrypoint_rejects_invalid_runtime_spec_bundle_before_claiming_verification(tmp_path: Path) -> None:
    invalid_bundle = {
        "policy_version": 3,
        "spec_hash": 123,
        "observation": {"obs_encoding_version": 2, "dtype": "i32", "obs_len": 512},
        "action": {"action_encoding_version": 1, "pass_action_id": 8},
    }
    (tmp_path / "weiss_sim.py").write_text(
        "\n".join(
            (
                "def build_info():",
                "    return 'stub-build'",
                "",
                "def db_info():",
                "    return 'stub-db'",
                "",
                "def export_spec_bundle():",
                f"    return {invalid_bundle!r}",
                "",
            )
        ),
        encoding="utf-8",
    )
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="123",
        run_label="invalid_spec_bundle",
    )

    assert result.returncode != 0
    assert "invalid spec_bundle payload" in result.stderr
    assert "Verified runtime spec bundle" not in result.stdout


def test_train_entrypoint_persists_runtime_spec_bundle(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="spec_bundle_run",
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "spec_bundle_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["simulator"]["compatibility_hash"] == "123"
    assert manifest["spec_bundle"] == bundle
    assert (manifest_path.parent / "spec_bundle.json").is_file()
    assert (manifest_path.parent / "spec_hash256.txt").read_text(encoding="utf-8").strip() == spec_bundle_hash(bundle)
    assert "computed_run_id64:" in result.stdout
    assert "computed_run_id256:" in result.stdout
    assert "run_label:              spec_bundle_run" in result.stdout
    assert "run_dir_name:           spec_bundle_run" in result.stdout
    assert "Manifest scaffold only: no learner training or rollout collection was executed." in result.stdout
    assert "active weiss_sim runtime is missing stepping APIs" in result.stdout


def test_train_entrypoint_uses_default_run_dir_when_no_label_override(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="123",
    )

    assert result.returncode == 0, result.stderr
    manifest_path_line = next(line for line in result.stdout.splitlines() if line.startswith("Wrote manifest: "))
    manifest_path = Path(manifest_path_line.removeprefix("Wrote manifest: ").strip())
    assert manifest_path.name == "manifest.json"
    assert manifest_path.parent.name.startswith("run_")
    assert "run_label:              (default)" in result.stdout
    assert f"run_dir_name:           {manifest_path.parent.name}" in result.stdout


def test_train_entrypoint_accepts_deprecated_run_id_alias(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="123",
        run_id_alias="compat_alias_run",
    )

    assert result.returncode == 0, result.stderr
    assert "deprecated; use --run-label instead" in result.stderr
    assert (tmp_path / "runs" / "compat_alias_run" / "manifest.json").is_file()


def test_eval_entrypoint_honors_explicit_spec_hash_without_reproducibility_config(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    _copy_repo_configs(tmp_path)
    stack_config = tmp_path / "configs" / "stack_smoke.yaml"

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash=_mismatched_sha256(spec_bundle_hash(bundle)),
    )

    assert result.returncode != 0
    assert "Spec bundle hash mismatch" in result.stderr


def test_eval_entrypoint_accepts_spec_bundle_sha256(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash=spec_bundle_hash(bundle),
    )

    assert result.returncode == 0, result.stderr
    assert "Verified runtime spec bundle" in result.stdout
    assert "run_label:              (default)" in result.stdout
    assert "computed_run_id64:" not in result.stdout


def test_eval_entrypoint_reports_run_label_without_claiming_run_identity(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash="",
        run_label="eval_report_label",
    )

    assert result.returncode == 0, result.stderr
    assert "run_label:              eval_report_label" in result.stdout
    assert "computed_run_id64:" not in result.stdout
    assert "computed_run_id256:" not in result.stdout


def test_eval_entrypoint_fails_fast_on_config_hash_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    config_hash256 = compute_config_hash256(load_stack_config(stack_config))

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash="",
        extra_args=["--config-hash", _mismatched_sha256(config_hash256)],
    )

    assert result.returncode != 0
    assert "--config-hash mismatch" in result.stderr


def test_eval_entrypoint_exports_summary_json_and_csv(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    episodes_path = tmp_path / "episodes.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    diagnostics_json = tmp_path / "diagnostics.json"
    episodes_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "pair_index": 0,
                        "swap_index": 0,
                        "episode_index": 0,
                        "episode_seed": 7,
                        "episode_key": "01" * 32,
                        "episode_key64": 1,
                        "config_hash256": "ab" * 32,
                        "spec_hash256": "cd" * 32,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "champion",
                        "seat1_policy_id": "baseline",
                        "focal_seat": 0,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "pair_index": 0,
                        "swap_index": 1,
                        "episode_index": 1,
                        "episode_seed": 7,
                        "episode_key": "02" * 32,
                        "episode_key64": 2,
                        "config_hash256": "ab" * 32,
                        "spec_hash256": "cd" * 32,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "baseline",
                        "seat1_policy_id": "champion",
                        "focal_seat": 1,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                    },
                    sort_keys=True,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash="",
        extra_args=[
            "--episodes-jsonl",
            str(episodes_path),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_csv),
            "--diagnostics-json",
            str(diagnostics_json),
            "--bootstrap-samples",
            "16",
            "--bootstrap-seed",
            "7",
        ],
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "decisive"
    assert payload["summary"]["wins"] == 2
    assert diagnostics["seat_results"]["seat0_wins"] == 1
    assert diagnostics["seat_results"]["seat1_wins"] == 1
    assert summary_csv.read_text(encoding="utf-8").splitlines()[0].startswith("focal_policy_id,")
