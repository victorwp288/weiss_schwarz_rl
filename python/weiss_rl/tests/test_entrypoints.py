from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from weiss_rl.spec import spec_bundle_hash

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_stub_weiss_sim(tmp_path: Path, *, spec_hash: int = 123) -> dict[str, object]:
    bundle: dict[str, object] = {
        "action_space_size": 527,
        "obs_len": 378,
        "spec_hash": spec_hash,
    }
    (tmp_path / "weiss_sim.py").write_text(
        "\n".join(
            [
                "def export_spec_bundle():",
                f"    return {bundle!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return bundle


def _run_entrypoint(
    tmp_path: Path,
    *,
    script_name: str,
    stack_config: Path,
    spec_hash: str,
    run_id: str = "",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO_ROOT / "python")])

    command = [sys.executable, str(REPO_ROOT / "python" / "scripts" / script_name), "--stack-config", str(stack_config)]
    if spec_hash:
        command.extend(["--spec-hash", spec_hash])
    if run_id:
        command.extend(["--run-id", run_id])

    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)


def test_train_entrypoint_fails_fast_on_runtime_spec_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=REPO_ROOT / "configs" / "rl_stack_locked.yaml",
        spec_hash="999",
        run_id="mismatch_run",
    )

    assert result.returncode != 0
    assert "Spec mismatch" in result.stderr


def test_train_entrypoint_persists_runtime_spec_bundle(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=REPO_ROOT / "configs" / "rl_stack_locked.yaml",
        spec_hash=str(bundle["spec_hash"]),
        run_id="spec_bundle_run",
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "spec_bundle_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["observed_spec_hash"] == "123"
    assert manifest["observed_spec_bundle_hash"] == spec_bundle_hash(bundle)
    assert manifest["spec_bundle"] == bundle
    assert (manifest_path.parent / "spec_bundle.json").is_file()
    assert (manifest_path.parent / "spec_hash256.txt").read_text(encoding="utf-8").strip() == spec_bundle_hash(bundle)


def test_eval_entrypoint_accepts_spec_bundle_sha256(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=REPO_ROOT / "configs" / "rl_stack_locked.yaml",
        spec_hash=spec_bundle_hash(bundle),
    )

    assert result.returncode == 0, result.stderr
    assert "Verified runtime spec bundle" in result.stdout
