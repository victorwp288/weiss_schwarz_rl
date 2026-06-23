from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.training.warmstarts.trajectory_bc_warmstart_runtime import _initial_hidden_state


def test_trajectory_bc_warmstart_entrypoint_exposes_only_cli_boundary() -> None:
    import weiss_rl.training.warmstarts.trajectory_bc_warmstart_entrypoint as trajectory_bc_warmstart_entrypoint

    assert trajectory_bc_warmstart_entrypoint.__all__ == ["main"]
    assert hasattr(trajectory_bc_warmstart_entrypoint, "main")
    assert not hasattr(trajectory_bc_warmstart_entrypoint, "_build_parser")
    assert not hasattr(trajectory_bc_warmstart_entrypoint, "_initial_hidden_state")
    assert not hasattr(trajectory_bc_warmstart_entrypoint, "_publish_trajectory_bc_snapshot")
    assert not hasattr(trajectory_bc_warmstart_entrypoint, "parse_trajectory_bc_warmstart_args")
    assert not hasattr(trajectory_bc_warmstart_entrypoint, "run_trajectory_bc_warmstart")


def test_trajectory_bc_warmstart_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "trajectory_bc_warmstart_cli")
    assert not hasattr(training, "trajectory_bc_warmstart_entrypoint")
    assert not hasattr(training, "trajectory_bc_warmstart_runtime")


def test_trajectory_bc_warmstart_entrypoint_main_delegates_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import weiss_rl.training.warmstarts.trajectory_bc_warmstart_entrypoint as trajectory_bc_warmstart_entrypoint

    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> int:
        calls.append(argv)
        return 29

    monkeypatch.setattr(trajectory_bc_warmstart_entrypoint, "_run_trajectory_bc_warmstart", fake_run)

    assert trajectory_bc_warmstart_entrypoint.main(["--probe"]) == 29
    assert calls == [["--probe"]]


def test_trajectory_bc_warmstart_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.training.warmstarts.trajectory_bc_warmstart_cli import build_trajectory_bc_warmstart_parser

    args = build_trajectory_bc_warmstart_parser().parse_args(
        [
            "--stack-config",
            str(tmp_path / "stack.yaml"),
            "--dataset",
            str(tmp_path / "trajectory.npz"),
            "--init-from-checkpoint",
            str(tmp_path / "init.pt"),
            "--output-run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.stack_config == tmp_path / "stack.yaml"
    assert args.dataset == tmp_path / "trajectory.npz"
    assert args.init_from_checkpoint == tmp_path / "init.pt"
    assert args.output_run_dir == tmp_path / "run"
    assert args.device == "cuda"
    assert args.epochs == 2
    assert args.batch_episodes == 8
    assert args.seed == 20260516
    assert args.mixed_precision is False
    assert args.teacher_family_coef == 0.05
    assert args.teacher_slot_coef == 0.05
    assert args.teacher_move_source_coef == 0.02
    assert args.teacher_attack_type_coef == 0.02
    assert args.teacher_action_coef == 0.20
    assert args.teacher_same_family_action_coef == 0.60
    assert args.teacher_action_margin_coef == 0.0
    assert args.teacher_action_margin == 0.5
    assert args.teacher_same_family_action_margin_coef == 0.10
    assert args.teacher_same_family_action_margin == 0.5
    assert args.exact_action_family is None


def test_trajectory_bc_warmstart_parser_validates_positive_counts(tmp_path: Path) -> None:
    from weiss_rl.training.warmstarts.trajectory_bc_warmstart_cli import parse_trajectory_bc_warmstart_args

    base_args = [
        "--stack-config",
        str(tmp_path / "stack.yaml"),
        "--dataset",
        str(tmp_path / "trajectory.npz"),
        "--init-from-checkpoint",
        str(tmp_path / "init.pt"),
        "--output-run-dir",
        str(tmp_path / "run"),
    ]

    with pytest.raises(SystemExit):
        parse_trajectory_bc_warmstart_args([*base_args, "--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_trajectory_bc_warmstart_args([*base_args, "--batch-episodes", "0"])


def test_trajectory_bc_warmstart_runtime_preserves_dataset_train_row_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weiss_rl.training.warmstarts.trajectory_bc_warmstart_runtime as trajectory_bc_warmstart_runtime

    monkeypatch.setattr(
        trajectory_bc_warmstart_runtime,
        "load_replay_trajectory_bc_dataset",
        lambda _path: SimpleNamespace(metadata={"train_rows": 0}),
    )

    with pytest.raises(SystemExit) as exc_info:
        trajectory_bc_warmstart_runtime.run_trajectory_bc_warmstart(
            [
                "--stack-config",
                str(tmp_path / "stack.yaml"),
                "--dataset",
                str(tmp_path / "trajectory.npz"),
                "--init-from-checkpoint",
                str(tmp_path / "init.pt"),
                "--output-run-dir",
                str(tmp_path / "run"),
            ]
        )

    assert exc_info.value.code == 2


def test_trajectory_bc_warmstart_initial_hidden_state_uses_model_hook() -> None:
    model = _HiddenModel()

    hidden = _initial_hidden_state(model, batch_size=3, device="cpu")

    assert model.calls == [{"batch_size": 3, "device": "cpu"}]
    np.testing.assert_array_equal(hidden, np.asarray([[1, 2], [1, 2], [1, 2]], dtype=np.float32))


class _Hidden:
    def __init__(self, batch_size: int) -> None:
        self._batch_size = int(batch_size)

    def detach(self) -> _Hidden:
        return self

    def cpu(self) -> _Hidden:
        return self

    def numpy(self) -> np.ndarray:
        return np.tile(np.asarray([[1, 2]], dtype=np.float32), (self._batch_size, 1))


class _HiddenModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def initial_seat_hidden(self, batch_size: int, *, device: object) -> _Hidden:
        self.calls.append({"batch_size": int(batch_size), "device": device})
        return _Hidden(batch_size)
