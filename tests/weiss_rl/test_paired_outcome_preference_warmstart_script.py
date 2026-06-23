from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_support import (
    _parse_pair_role_selectors,
    _parse_pair_weights,
    _preference_pair_role_mask,
    _preference_pair_weight_matrix,
    _scale_optimizer_learning_rates,
    _serialize_pair_role_selectors,
)


def test_preference_warmstart_entrypoint_exposes_only_cli_boundary() -> None:
    import weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_entrypoint as preference_entrypoint

    assert preference_entrypoint.__all__ == ["main"]
    assert hasattr(preference_entrypoint, "main")
    assert not hasattr(preference_entrypoint, "_build_parser")
    assert not hasattr(preference_entrypoint, "_parse_pair_weights")
    assert not hasattr(preference_entrypoint, "_preference_pair_role_mask")
    assert not hasattr(preference_entrypoint, "_initial_hidden_state")
    assert not hasattr(preference_entrypoint, "parse_paired_outcome_preference_warmstart_args")
    assert not hasattr(preference_entrypoint, "run_paired_outcome_preference_warmstart")


def test_preference_warmstart_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "paired_outcome_preference_warmstart_cli")
    assert not hasattr(training, "paired_outcome_preference_warmstart_entrypoint")
    assert not hasattr(training, "paired_outcome_preference_warmstart_runtime")


def test_preference_warmstart_entrypoint_main_delegates_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_entrypoint as preference_entrypoint

    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> int:
        calls.append(argv)
        return 17

    monkeypatch.setattr(preference_entrypoint, "_run_paired_outcome_preference_warmstart", fake_run)

    assert preference_entrypoint.main(["--dry-run"]) == 17
    assert calls == [["--dry-run"]]


def test_preference_warmstart_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_cli import (
        build_paired_outcome_preference_warmstart_parser,
    )

    args = build_paired_outcome_preference_warmstart_parser().parse_args(
        [
            "--stack-config",
            str(tmp_path / "stack.yaml"),
            "--dataset",
            str(tmp_path / "preference.npz"),
            "--init-from-checkpoint",
            str(tmp_path / "init.pt"),
            "--output-run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.stack_config == tmp_path / "stack.yaml"
    assert args.dataset == tmp_path / "preference.npz"
    assert args.init_from_checkpoint == tmp_path / "init.pt"
    assert args.output_run_dir == tmp_path / "run"
    assert args.device == "cuda"
    assert args.epochs == 1
    assert args.batch_episodes == 8
    assert args.seed == 20260520
    assert args.beta == 0.2
    assert args.coef == 0.08
    assert args.optimizer_lr_scale == 1.0
    assert args.aggregation == "mean"
    assert args.group_balance is False
    assert args.pair_weight == []
    assert args.target_logp_retention_coef == 0.0
    assert args.target_logp_retention_margin == 0.0
    assert args.target_logp_retention_role == "preferred"
    assert args.target_logp_retention_reference_top_only is False
    assert args.target_logp_retention_pair_role == []
    assert args.top_action_retention_coef == 0.0
    assert args.top_action_retention_margin == 0.0
    assert args.top_action_retention_role == "all"
    assert args.top_action_retention_reference_top_only is False
    assert args.top_action_retention_pair_role == []


def test_preference_warmstart_parser_validates_numeric_and_selector_bounds(tmp_path: Path) -> None:
    from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_cli import (
        parse_paired_outcome_preference_warmstart_args,
    )

    base_args = [
        "--stack-config",
        str(tmp_path / "stack.yaml"),
        "--dataset",
        str(tmp_path / "preference.npz"),
        "--init-from-checkpoint",
        str(tmp_path / "init.pt"),
        "--output-run-dir",
        str(tmp_path / "run"),
    ]

    with pytest.raises(SystemExit):
        parse_paired_outcome_preference_warmstart_args([*base_args, "--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_paired_outcome_preference_warmstart_args([*base_args, "--pair-weight", "9=0"])
    with pytest.raises(SystemExit):
        parse_paired_outcome_preference_warmstart_args([*base_args, "--target-logp-retention-pair-role", "9:maybe"])


def test_preference_warmstart_runtime_preserves_dataset_train_row_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_runtime as paired_outcome_preference_warmstart_runtime

    monkeypatch.setattr(
        paired_outcome_preference_warmstart_runtime,
        "load_replay_trajectory_bc_dataset",
        lambda _path: SimpleNamespace(metadata={"train_rows": 0}),
    )

    with pytest.raises(SystemExit) as exc_info:
        paired_outcome_preference_warmstart_runtime.run_paired_outcome_preference_warmstart(
            [
                "--stack-config",
                str(tmp_path / "stack.yaml"),
                "--dataset",
                str(tmp_path / "preference.npz"),
                "--init-from-checkpoint",
                str(tmp_path / "init.pt"),
                "--output-run-dir",
                str(tmp_path / "run"),
            ]
        )

    assert exc_info.value.code == 2


def test_preference_warmstart_can_scale_optimizer_learning_rates() -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 2e-4}, {"lr": 1e-4, "name": "head"}])

    summary = _scale_optimizer_learning_rates(optimizer, scale=0.25)

    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[1]["lr"] == 2.5e-5
    assert summary == {
        "scale": 0.25,
        "groups": [
            {"index": 0, "original_lr": 2e-4, "scaled_lr": 5e-5},
            {"index": 1, "original_lr": 1e-4, "scaled_lr": 2.5e-5},
        ],
    }


def test_preference_warmstart_parses_pair_weights() -> None:
    assert _parse_pair_weights(["9=8", "14=2.5"]) == {9: 8.0, 14: 2.5}

    with pytest.raises(ValueError, match="PAIR_ID=WEIGHT"):
        _parse_pair_weights(["9"])
    with pytest.raises(ValueError, match="finite and positive"):
        _parse_pair_weights(["9=0"])


def test_preference_warmstart_builds_pair_weight_matrix() -> None:
    pair_ids = np.asarray([[9, 9, 14], [-1, 3, 3]], dtype=np.int64)

    weights = _preference_pair_weight_matrix(pair_ids, {9: 8.0, 14: 2.0})

    np.testing.assert_allclose(weights, np.asarray([[8.0, 8.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float32))


def test_preference_warmstart_parses_and_builds_pair_role_masks() -> None:
    selectors = _parse_pair_role_selectors(["9:preferred", "14:rejected", "15:all", "9:preferred"])
    assert selectors == ((9, 1), (14, 0), (15, None))
    assert _serialize_pair_role_selectors(selectors) == ["9:preferred", "14:rejected", "15:all"]

    pair_ids = np.asarray([[9, 9, 14], [14, 15, 3]], dtype=np.int64)
    roles = np.asarray([[1, 0, 0], [1, 0, 1]], dtype=np.int64)

    mask = _preference_pair_role_mask(pair_ids, roles, selectors)

    np.testing.assert_allclose(mask, np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32))


def test_preference_warmstart_rejects_bad_pair_role_selectors() -> None:
    with pytest.raises(ValueError, match="PAIR_ID:ROLE"):
        _parse_pair_role_selectors(["9=preferred"])
    with pytest.raises(ValueError, match="preferred, rejected, or all"):
        _parse_pair_role_selectors(["9:maybe"])
