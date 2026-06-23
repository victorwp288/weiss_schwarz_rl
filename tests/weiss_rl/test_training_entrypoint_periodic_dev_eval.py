from __future__ import annotations

import json
from pathlib import Path

import torch

from .entrypoints_test_support import (
    _copy_repo_configs,
    _patch_periodic_dev_eval_config,
    _run_entrypoint,
    _write_b1_baseline_run_fixture,
    _write_runtime_weiss_sim,
)


def test_train_entrypoint_runs_periodic_dev_eval_and_handles_empty_ids_pass_fallback(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(
        tmp_path,
        spec_hash=123,
        pass_action_id=3,
        empty_eval_legal_row=True,
    )
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="periodic_dev_eval_run",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--checkpoint-interval-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "Periodic dev eval: update=1 opponent=b0_randomlegal" in result.stdout

    eval_root = tmp_path / "runs" / "periodic_dev_eval_run" / "eval" / "dev_eval" / "update_1"
    seed_usage = json.loads((eval_root / "b0_randomlegal" / "seed_usage.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((eval_root / "b0_randomlegal" / "matchup_summary.json").read_text(encoding="utf-8"))
    diagnostics_payload = json.loads((eval_root / "b0_randomlegal" / "diagnostics.json").read_text(encoding="utf-8"))
    episodes_lines = (eval_root / "b0_randomlegal" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()

    assert seed_usage["seed_file"]["path"] == "configs/seeds/dev_eval_seeds.txt"
    assert seed_usage["paired_seed_count"] == 1
    assert seed_usage["paired_seeds"] == [7]
    assert seed_usage["focal_policy"]["update_count"] == 1
    assert seed_usage["focal_policy"]["policy_version"] == 1
    assert seed_usage["focal_policy"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert len(episodes_lines) == 2
    assert summary_payload["summary"]["games"] == 2
    assert summary_payload["evaluation_context"] == {
        "artifact_scope": "periodic_dev_eval",
        "update_count": 1,
        "policy_version": 1,
        "checkpoint_path": "training/checkpoints/checkpoint_1.pt",
        "seed_usage_path": "eval/dev_eval/update_1/b0_randomlegal/seed_usage.json",
        "anchor_display_name": "B0 RandomLegal",
    }
    assert diagnostics_payload["seat_results"]["seat0_wins"] == 2
    assert diagnostics_payload["seat_results"]["seat1_wins"] == 0
    assert (eval_root / "b0_randomlegal" / "matchup_summary.csv").is_file()


def test_train_entrypoint_periodic_dev_eval_writes_exact_current_checkpoint(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="periodic_dev_eval_checkpoint_traceability",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--checkpoint-interval-updates",
            "2",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr

    run_root = tmp_path / "runs" / "periodic_dev_eval_checkpoint_traceability"
    eval_root = run_root / "eval" / "dev_eval" / "update_1"
    checkpoint_path = run_root / "training" / "checkpoints" / "checkpoint_1.pt"
    seed_usage = json.loads((eval_root / "b0_randomlegal" / "seed_usage.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((eval_root / "b0_randomlegal" / "matchup_summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert checkpoint_path.is_file()
    assert seed_usage["focal_policy"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert summary_payload["evaluation_context"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert checkpoint_payload["update_count"] == 1


def test_train_entrypoint_uses_configured_checkpoint_interval_by_default(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="checkpoint_default_from_config",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / "checkpoint_default_from_config"
    registry = json.loads((run_root / "training" / "snapshots" / "registry.json").read_text(encoding="utf-8"))
    assert [snapshot["policy_id"] for snapshot in registry["snapshots"]] == ["b1_noleague_baseline"]
    assert not (run_root / "eval" / "promotion_gate" / "update_1").exists()
