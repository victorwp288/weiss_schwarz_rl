from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from weiss_rl.training.execution import resolve_training_execution_settings


def _training_config() -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_interval_updates=7,
        profile_timers=True,
        torch_profiler=False,
    )


def test_resolve_training_execution_settings_uses_config_defaults_and_resolves_paths(tmp_path: Path) -> None:
    b1_dir = tmp_path / "b1"
    seed_dir = tmp_path / "seed"
    init_checkpoint = tmp_path / "seed" / "best.pt"

    settings = resolve_training_execution_settings(
        training_config=_training_config(),
        checkpoint_interval_override=None,
        b1_baseline_run_dir=b1_dir,
        seed_snapshot_run_dir=seed_dir,
        init_from_checkpoint=init_checkpoint,
    )

    assert settings.checkpoint_interval_updates == 7
    assert settings.profile_timers is True
    assert settings.torch_profiler is False
    assert settings.b1_baseline_run_dir == b1_dir.resolve()
    assert settings.seed_snapshot_run_dir == seed_dir.resolve()
    assert settings.init_from_checkpoint_path == init_checkpoint.resolve()


def test_resolve_training_execution_settings_prefers_checkpoint_interval_override() -> None:
    settings = resolve_training_execution_settings(
        training_config=_training_config(),
        checkpoint_interval_override=3,
        b1_baseline_run_dir=None,
        seed_snapshot_run_dir=None,
        init_from_checkpoint=None,
    )

    assert settings.checkpoint_interval_updates == 3
    assert settings.b1_baseline_run_dir is None
    assert settings.seed_snapshot_run_dir is None
    assert settings.init_from_checkpoint_path is None


def test_resolve_training_execution_settings_rejects_non_positive_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="--checkpoint-interval-updates must be >= 1, got 0"):
        resolve_training_execution_settings(
            training_config=_training_config(),
            checkpoint_interval_override=0,
            b1_baseline_run_dir=None,
            seed_snapshot_run_dir=None,
            init_from_checkpoint=None,
        )
