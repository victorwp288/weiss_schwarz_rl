from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from weiss_rl.config import load_stack_config

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _heuristic_public_contract_bundle,
    _load_train_script_module,
    _TrainingPathsLike,
)


def test_run_minimal_training_bootstraps_noleague_baseline_before_env_start(tmp_path: Path, monkeypatch) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())

    bootstrap_calls: list[dict[str, Any]] = []

    def fake_ensure_noleague_baseline_anchor(**kwargs):
        bootstrap_calls.append(kwargs)
        return "b1_noleague_baseline"

    def stop_before_runtime(*args, **kwargs):
        raise RuntimeError("stop after bootstrap")

    monkeypatch.setattr(train_script, "_ensure_noleague_baseline_anchor", fake_ensure_noleague_baseline_anchor)
    monkeypatch.setattr(train_script, "QueueRuntime", stop_before_runtime)

    run_dir = tmp_path / "run"
    try:
        train_script._run_minimal_training(
            stack=stack,
            contract=SimpleNamespace(spec_bundle=_heuristic_public_contract_bundle()),
            artifacts=SimpleNamespace(run_dir=run_dir),
            num_envs=1,
            unroll_length=1,
            max_updates=1,
            profile="fast",
            device=torch.device("cpu"),
            seed=7,
            checkpoint_interval_updates=1,
            run_id256="12" * 32,
            config_hash256="34" * 32,
            spec_hash256="56" * 32,
            runtime_mode="train_ordered",
            b1_baseline_run_dir=None,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after bootstrap"
    else:
        raise AssertionError("expected QueueRuntime to stop the test after baseline bootstrap")

    assert len(bootstrap_calls) == 1
    bootstrap_call = bootstrap_calls[0]
    assert bootstrap_call["run_dir"] == run_dir
    training_paths_arg = cast(_TrainingPathsLike, bootstrap_call["training_paths"])
    assert training_paths_arg.snapshots_dir == run_dir / "training" / "snapshots"
    assert bootstrap_call["device"] == torch.device("cpu")
    assert bootstrap_call["config_hash256"] == train_script.compute_config_hash256(stack)
    assert bootstrap_call["baseline_run_dir"] is None
