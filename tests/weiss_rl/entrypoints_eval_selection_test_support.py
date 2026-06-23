from __future__ import annotations

from typing import Any

from .entrypoints_test_support import (
    ArtifactLayout,
    Path,
    _copy_repo_configs,
    _write_eval_only_stack_config,
    _write_policy_set_inputs,
    load_stack_config,
    shutil,
)

EXPECTED_DETERMINISTIC_POLICY_IDS = [
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


def prepare_policy_selection_run(tmp_path: Path, run_name: str) -> tuple[Any, ArtifactLayout]:
    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    layout = ArtifactLayout.from_run_dir(tmp_path / "runs" / run_name)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(
        dev_eval_summaries_path,
        layout.training_logs_dir / "periodic_dev_eval_summaries.json",
    )
    return stack, layout
