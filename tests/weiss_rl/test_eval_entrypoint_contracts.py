from __future__ import annotations

from .entrypoints_test_support import (
    Path,
    _copy_repo_configs,
    _mismatched_sha256,
    _run_entrypoint,
    _write_manifest_only_stack_config,
    _write_stub_weiss_sim,
    compute_config_hash256,
    load_stack_config,
    spec_bundle_hash,
)


def test_eval_entrypoint_honors_explicit_spec_hash_without_reproducibility_config(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
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
        module_name="weiss_rl.workflows.eval_entrypoint",
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
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash="",
        run_label="eval_report_label",
    )

    assert result.returncode == 0, result.stderr
    assert "run_label:              eval_report_label" in result.stdout
    assert "Verified runtime spec bundle" in result.stdout
    assert "computed_run_id64:" not in result.stdout
    assert "computed_run_id256:" not in result.stdout


def test_eval_entrypoint_fails_fast_on_config_hash_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    config_hash256 = compute_config_hash256(load_stack_config(stack_config))

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash="",
        extra_args=["--config-hash", _mismatched_sha256(config_hash256)],
    )

    assert result.returncode != 0
    assert "--config-hash mismatch" in result.stderr


def test_eval_entrypoint_requires_skip_readiness_when_skipping_required_outputs(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash="",
        extra_args=["--run-dir", str(tmp_path / "runs" / "candidate"), "--skip-metagame"],
    )

    assert result.returncode != 0
    assert "--skip-metagame or --skip-figures requires --skip-readiness" in result.stderr
