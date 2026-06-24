from __future__ import annotations

import pytest
from weiss_rl.config.sections.sections_training import parse_training_config

from tests.weiss_rl.config_training_test_support import training_body


def test_parse_training_config_accepts_trajectory_bc_focus_groups() -> None:
    body = training_body()
    body["structured_aux"] = {
        "trajectory_bc_dataset_path": "runs/bc/dataset.npz",
        "trajectory_bc_every_updates": 1,
        "trajectory_bc_focus_groups": [
            {
                "name": " learned_repair ",
                "source_labels": [" champion_a ", "hard_negative_a"],
                "fraction": 0.30,
            },
            {
                "name": "fixed_repair",
                "source_labels": ["b1_lossstate", "b3_lossstate"],
                "fraction": 0.25,
            },
        ],
    }

    config = parse_training_config(body)

    assert config.trajectory_bc_enabled is True
    assert config.structured_aux.trajectory_bc_focus_source_labels == ()
    assert config.structured_aux.trajectory_bc_focus_fraction == pytest.approx(0.0)
    assert tuple(group.name for group in config.structured_aux.trajectory_bc_focus_groups) == (
        "learned_repair",
        "fixed_repair",
    )
    assert config.structured_aux.trajectory_bc_focus_groups[0].source_labels == (
        "champion_a",
        "hard_negative_a",
    )
    assert config.structured_aux.trajectory_bc_focus_groups[0].fraction == pytest.approx(0.30)
    assert config.structured_aux.trajectory_bc_focus_groups[1].source_labels == (
        "b1_lossstate",
        "b3_lossstate",
    )
    assert config.structured_aux.trajectory_bc_focus_groups[1].fraction == pytest.approx(0.25)
