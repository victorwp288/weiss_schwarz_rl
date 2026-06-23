from __future__ import annotations

from pathlib import Path

import pytest

from .snapshot_registry_test_support import (
    _load_train_script_module,
)


def test_resolve_resume_checkpoint_path_defaults_to_latest_alias(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    import weiss_rl.training.checkpointing.resolution as checkpoint_resolution
    from weiss_rl.training.checkpoints import resolve_resume_checkpoint_path

    run_dir = tmp_path / "resume_run"
    latest_path = run_dir / "training" / "checkpoints" / "latest.pt"
    best_path = run_dir / "training" / "checkpoints" / "best.pt"
    observed_best_path = run_dir / "training" / "checkpoints" / "observed_best.pt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_bytes(b"checkpoint")
    best_path.write_bytes(b"best")
    observed_best_path.write_bytes(b"observed")
    explicit_checkpoint_path = tmp_path / "manual.pt"
    explicit_checkpoint_path.write_bytes(b"manual")

    resolved = train_script._resolve_resume_checkpoint_path(
        resume_from="",
        resume_run_dir=run_dir,
    )

    assert resolve_resume_checkpoint_path is checkpoint_resolution.resolve_resume_checkpoint_path
    assert resolved == latest_path.resolve()
    assert resolve_resume_checkpoint_path(resume_from="", resume_run_dir=run_dir) == latest_path.resolve()
    assert resolve_resume_checkpoint_path(resume_from=" BEST ", resume_run_dir=run_dir) == best_path.resolve()
    assert (
        train_script._resolve_resume_checkpoint_path(
            resume_from="observed_best",
            resume_run_dir=run_dir,
        )
        == observed_best_path.resolve()
    )
    assert (
        resolve_resume_checkpoint_path(resume_from="observed_best", resume_run_dir=run_dir)
        == observed_best_path.resolve()
    )
    assert (
        resolve_resume_checkpoint_path(resume_from=str(explicit_checkpoint_path), resume_run_dir=None)
        == explicit_checkpoint_path.resolve()
    )
    with pytest.raises(ValueError, match="requires --resume-run-dir"):
        resolve_resume_checkpoint_path(resume_from="latest", resume_run_dir=None)
