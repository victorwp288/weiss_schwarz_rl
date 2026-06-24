from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.core.spec import spec_bundle_hash

from .entrypoints_test_support import (
    _copy_repo_configs,
    _run_entrypoint,
    _write_eval_only_stack_config,
    _write_manifest_only_stack_config,
    _write_policy_set_inputs,
    _write_stub_weiss_sim,
)


def test_train_entrypoint_persists_runtime_spec_bundle(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="spec_bundle_run",
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "spec_bundle_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["simulator"]["compatibility_hash"] == "123"
    assert manifest["spec_bundle"] == bundle
    assert manifest["policy_set_selection"] == []
    assert manifest["policy_set_selection_details"] == {
        "mode": "not_configured",
        "status": "not_configured",
        "source_paths": {
            "snapshot_registry_json": None,
            "dev_eval_summaries_json": None,
        },
    }
    assert (manifest_path.parent / "spec_bundle.json").is_file()
    assert (manifest_path.parent / "spec_hash256.txt").read_text(encoding="utf-8").strip() == spec_bundle_hash(bundle)
    assert "computed_run_id64:" in result.stdout
    assert "computed_run_id256:" in result.stdout
    assert "run_label:              spec_bundle_run" in result.stdout
    assert "run_dir_name:           spec_bundle_run" in result.stdout
    assert "Manifest scaffold only: no learner training or rollout collection was executed." in result.stdout
    assert "missing config blocks: environment, training, model" in result.stdout


def test_train_entrypoint_resolves_policy_set_selection_when_inputs_are_supplied(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="resolved_policy_set_run",
        extra_args=[
            "--snapshot-registry-json",
            str(snapshot_registry_path),
            "--dev-eval-summaries-json",
            str(dev_eval_summaries_path),
        ],
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "resolved_policy_set_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["policy_set_selection"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000400",
        "policy_000100",
        "policy_000200",
        "policy_000300",
        "policy_000150",
        "policy_000250",
        "policy_000350",
    ]
    details = manifest["policy_set_selection_details"]
    assert details == {
        "mode": "deterministic_v1",
        "status": "resolved",
        "version": "deterministic_v1",
        "final_policy_set_size": 10,
        "source_paths": {
            "snapshot_registry_json": "policy_set_snapshot_registry.json",
            "dev_eval_summaries_json": "policy_set_dev_eval_summaries.json",
        },
        "missing_inputs": [],
        "selected_policy_count": 10,
        "selection_trace": details["selection_trace"],
    }
    assert len(details["selection_trace"]) == 10
    assert details["selection_trace"][0]["reason"] == "random_legal_baseline_b0"
    assert details["selection_trace"][-1]["reason"] == "top_dev_performer_vs_anchor_set"


def test_train_entrypoint_uses_default_run_dir_when_no_label_override(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
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
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash="123",
        run_id_alias="compat_alias_run",
    )

    assert result.returncode == 0, result.stderr
    assert "deprecated; use --run-label instead" in result.stderr
    assert (tmp_path / "runs" / "compat_alias_run" / "manifest.json").is_file()
