from __future__ import annotations

from pathlib import Path

from .entrypoints_test_support import (
    _copy_repo_configs,
    _run_entrypoint,
    _write_stub_weiss_sim,
)


def test_train_entrypoint_fails_fast_on_runtime_spec_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
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
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash="123",
        run_label="invalid_spec_bundle",
    )

    assert result.returncode != 0
    assert "invalid spec_bundle payload" in result.stderr
    assert "Verified runtime spec bundle" not in result.stdout


def test_train_entrypoint_locked_stack_fails_on_incomplete_runtime(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="locked_stack_requires_runtime",
    )

    assert result.returncode != 0
    assert "Canonical simulator-backed training requires a weiss_sim runtime with stepping support" in result.stderr
    assert "active weiss_sim runtime is missing stepping APIs" in result.stderr
