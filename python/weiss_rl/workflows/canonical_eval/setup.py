from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState


def prepare_canonical_eval_run_state(
    *,
    parser: argparse.ArgumentParser,
    stack: Any,
    run_dir: Path,
    final_eval_dir: Path | None,
    skip_metagame: bool,
    study_config_path: Path | None,
    git_commit_override: str,
    dependencies: Any,
) -> CanonicalEvalRunState:
    layout = dependencies.artifact_layout_cls.from_run_dir(run_dir)
    layout.ensure_directories()
    tensorboard_logger = dependencies.tensorboard_logger_cls(layout.tensorboard_dir)
    if final_eval_dir is not None and final_eval_dir.resolve() != layout.final_eval_dir.resolve():
        parser.error(
            f"--final-eval-dir must match the canonical run directory layout for non-demo runs: {layout.final_eval_dir}"
        )

    manifest = cast(dict[str, Any], dependencies.load_json_object_fn(layout.manifest_path, label="run manifest"))
    effective_git_commit = dependencies.effective_manifest_git_commit_fn(
        manifest=manifest,
        git_commit_override=git_commit_override,
    )
    if effective_git_commit:
        manifest_git_commit = dependencies.normalize_git_commit_fn(str(manifest.get("git_commit", "")))
        if manifest_git_commit:
            print(f"Eval provenance git commit: {manifest_git_commit}")
        else:
            print(f"Eval provenance git commit override (not persisted): {effective_git_commit}")
    run_id256 = str(manifest.get("run_id256", "")).strip()
    if len(run_id256) != 64:
        raise ValueError(f"run manifest is missing a valid run_id256: {layout.manifest_path}")

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    study_config = None
    if not skip_metagame:
        resolved_study_config = (
            (stack.root / "configs" / "study" / "metagame_sensitivity.yaml")
            if study_config_path is None
            else study_config_path.resolve()
        )
        study_config = dependencies.load_study_config_fn(resolved_study_config)

    return CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=tensorboard_logger,
        manifest=manifest,
        run_id256=run_id256,
        evaluation=evaluation,
        study_config=study_config,
    )
