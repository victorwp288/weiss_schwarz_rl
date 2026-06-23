from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.toy_public_demo import public_demo_spec_hash256

from .entrypoints_test_support import (
    _copy_repo_configs,
    _run_entrypoint,
    _run_public_demo_train,
)


def test_train_entrypoint_public_demo_accepts_profile_timers_flag(tmp_path: Path) -> None:
    stack_config = _copy_repo_configs(tmp_path)
    run_label = "toy_public_demo_profile_timers"
    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        run_label=run_label,
        extra_args=["--public-demo", "--profile-timers"],
    )

    assert result.returncode == 0, result.stderr
    assert "Staged public-demo toy catalog and policy bundle" in result.stdout
    run_summary = json.loads((tmp_path / "runs" / run_label / "run_summary.json").read_text(encoding="utf-8"))
    training_controls = run_summary["training_controls"]
    assert training_controls["profile_timers"] is True
    assert training_controls["torch_profiler"] is False


def test_train_entrypoint_public_demo_stages_public_safe_catalog_without_weiss_sim(tmp_path: Path) -> None:
    result, run_dir = _run_public_demo_train(tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((run_dir / "public_demo" / "catalog.json").read_text(encoding="utf-8"))
    policies = json.loads((run_dir / "public_demo" / "policy_manifest.json").read_text(encoding="utf-8"))
    scalars_lines = (run_dir / "training" / "logs" / "scalars.jsonl").read_text(encoding="utf-8").splitlines()

    assert manifest["simulator"]["runtime"] == "public_demo"
    assert manifest["simulator"]["public_safe"] is True
    assert manifest["spec_bundle"]["action"]["action_space_size"] == 9
    assert catalog["public_safe"] is True
    assert len(catalog["card_pool"]) == 12
    assert len(catalog["decks"]) == 3
    assert policies["policy_ids"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "toy_policy_000100",
        "toy_policy_000200",
    ]
    assert len(scalars_lines) == 1
    assert "Loaded synthetic public-demo spec bundle" in result.stdout
    assert "Verified runtime spec bundle" not in result.stdout
    assert "Staged public-demo toy catalog and policy bundle" in result.stdout
    assert "demo-only" in result.stdout
