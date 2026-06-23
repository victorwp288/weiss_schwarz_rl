from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.training.warmstarts.warmstart_replay_support import (
    _opponent_context_indices_for_episodes,
    _sample_episode_indices,
    _source_opponent_policy_ids_by_episode,
)


def test_paired_swing_warmstart_entrypoint_exposes_only_cli_boundary() -> None:
    import weiss_rl.training.warmstarts.paired_swing_warmstart_entrypoint as paired_swing_warmstart_entrypoint

    assert paired_swing_warmstart_entrypoint.__all__ == ["main"]
    assert hasattr(paired_swing_warmstart_entrypoint, "main")
    assert not hasattr(paired_swing_warmstart_entrypoint, "_build_parser")
    assert not hasattr(paired_swing_warmstart_entrypoint, "_opponent_context_indices_for_episodes")
    assert not hasattr(paired_swing_warmstart_entrypoint, "_sample_episode_indices")
    assert not hasattr(paired_swing_warmstart_entrypoint, "parse_paired_swing_warmstart_args")
    assert not hasattr(paired_swing_warmstart_entrypoint, "run_paired_swing_warmstart")


def test_paired_swing_warmstart_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "paired_swing_warmstart_cli")
    assert not hasattr(training, "paired_swing_warmstart_entrypoint")
    assert not hasattr(training, "paired_swing_warmstart_runtime")


def test_paired_swing_warmstart_entrypoint_main_delegates_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import weiss_rl.training.warmstarts.paired_swing_warmstart_entrypoint as paired_swing_warmstart_entrypoint

    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> int:
        calls.append(argv)
        return 23

    monkeypatch.setattr(paired_swing_warmstart_entrypoint, "_run_paired_swing_warmstart", fake_run)

    assert paired_swing_warmstart_entrypoint.main(["--probe"]) == 23
    assert calls == [["--probe"]]


def test_paired_swing_warmstart_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.training.warmstarts.paired_swing_warmstart_cli import build_paired_swing_warmstart_parser

    args = build_paired_swing_warmstart_parser().parse_args(
        [
            "--stack-config",
            str(tmp_path / "stack.yaml"),
            "--dataset",
            str(tmp_path / "swing.npz"),
            "--init-from-checkpoint",
            str(tmp_path / "init.pt"),
            "--output-run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.stack_config == tmp_path / "stack.yaml"
    assert args.dataset == tmp_path / "swing.npz"
    assert args.init_from_checkpoint == tmp_path / "init.pt"
    assert args.output_run_dir == tmp_path / "run"
    assert args.device == "cuda"
    assert args.epochs == 1
    assert args.batch_episodes == 8
    assert args.seed == 20260520
    assert args.margin == 0.35
    assert args.coef == 0.08
    assert args.positive_action_source == "actions"
    assert args.negative_action_source == "teacher_action"
    assert args.loss_scope == "row"
    assert args.compare_to == "negative"
    assert args.margin_retention_coef == 0.0
    assert args.margin_retention_margin == 0.0
    assert args.top_action_retention_coef == 0.0
    assert args.top_action_retention_margin == 0.0
    assert args.full_surface_retention_dataset is None
    assert args.full_surface_retention_coef == 0.0
    assert args.full_surface_retention_margin == 0.0
    assert args.full_surface_retention_batch_episodes == 0
    assert args.full_surface_retention_mode == "reference_top"
    assert args.conflict_filter == "none"
    assert args.allow_missing_context is False


def test_paired_swing_warmstart_parser_validates_numeric_and_source_bounds(tmp_path: Path) -> None:
    from weiss_rl.training.warmstarts.paired_swing_warmstart_cli import parse_paired_swing_warmstart_args

    base_args = [
        "--stack-config",
        str(tmp_path / "stack.yaml"),
        "--dataset",
        str(tmp_path / "swing.npz"),
        "--init-from-checkpoint",
        str(tmp_path / "init.pt"),
        "--output-run-dir",
        str(tmp_path / "run"),
    ]

    with pytest.raises(SystemExit):
        parse_paired_swing_warmstart_args([*base_args, "--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_paired_swing_warmstart_args([*base_args, "--full-surface-retention-coef", "0.1"])
    with pytest.raises(SystemExit):
        parse_paired_swing_warmstart_args(
            [*base_args, "--positive-action-source", "actions", "--negative-action-source", "actions"]
        )


def test_paired_swing_warmstart_runtime_preserves_dataset_train_row_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weiss_rl.training.warmstarts.paired_swing_warmstart_runtime as paired_swing_warmstart_runtime

    monkeypatch.setattr(
        paired_swing_warmstart_runtime,
        "load_replay_trajectory_bc_dataset",
        lambda _path: SimpleNamespace(metadata={"train_rows": 0}),
    )
    monkeypatch.setattr(paired_swing_warmstart_runtime, "paired_swing_distinct_train_row_count", lambda *_a, **_k: 1)

    with pytest.raises(SystemExit) as exc_info:
        paired_swing_warmstart_runtime.run_paired_swing_warmstart(
            [
                "--stack-config",
                str(tmp_path / "stack.yaml"),
                "--dataset",
                str(tmp_path / "swing.npz"),
                "--init-from-checkpoint",
                str(tmp_path / "init.pt"),
                "--output-run-dir",
                str(tmp_path / "run"),
            ]
        )

    assert exc_info.value.code == 2


def test_paired_swing_warmstart_maps_source_opponents_to_context_indices() -> None:
    dataset = SimpleNamespace(
        episode_count=3,
        metadata={
            "selected_bundles": [
                {"source_opponent_policy_id": "B2"},
                {"source_opponent_policy_id": ""},
                {"source_opponent_policy_id": "policy_000004"},
            ]
        },
    )
    model = _ContextModel({"B2": 4, "policy_000004": 9})

    indices = _opponent_context_indices_for_episodes(model, dataset, episode_indices=[2, 0, 1])

    assert indices.tolist() == [9, 4, 0]
    assert _source_opponent_policy_ids_by_episode(dataset) == ["B2", "", "policy_000004"]


def test_paired_swing_warmstart_returns_zero_context_without_model_support() -> None:
    dataset = SimpleNamespace(
        episode_count=2,
        metadata={"selected_bundles": [{"source_opponent_policy_id": "B2"}, {"source_opponent_policy_id": "B4"}]},
    )

    indices = _opponent_context_indices_for_episodes(object(), dataset, episode_indices=[0, 1])

    np.testing.assert_array_equal(indices, np.zeros((2,), dtype=np.int64))


def test_paired_swing_warmstart_samples_retention_indices_with_replacement_when_needed() -> None:
    rng = np.random.default_rng(123)

    indices = _sample_episode_indices(rng, episode_count=2, batch_episodes=5)

    assert len(indices) == 5
    assert set(indices).issubset({0, 1})


class _ContextModel:
    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = dict(mapping)

    def opponent_context_indices_for_policy_ids(self, policy_ids: list[str]) -> list[int]:
        return [self._mapping.get(policy_id, 0) for policy_id in policy_ids]
