from __future__ import annotations

import numpy as np
import pytest
from scripts.reward_component_probe import summarize_reward_samples


def test_summarize_reward_samples_checks_component_sum_and_scale() -> None:
    rewards = np.asarray([0.10, -0.05, 1.0], dtype=np.float32)
    components = np.asarray(
        [
            [0.0, 0.10, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.05, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_reward_samples(
        rewards=rewards,
        reward_components=components,
        terminated=np.asarray([False, False, True]),
        truncated=np.asarray([False, False, False]),
        engine_status=np.asarray([0, 0, 0], dtype=np.uint8),
    )

    assert summary["transition_count"] == 3
    assert summary["component_sum_error_max_abs"] == pytest.approx(0.0)
    assert summary["reward"]["positive_fraction"] == pytest.approx(2 / 3)
    assert summary["components"]["terminal"]["nonzero_fraction"] == pytest.approx(1 / 3)
    assert summary["components"]["damage"]["sum"] == pytest.approx(0.10)
    assert summary["components"]["level"]["sum"] == pytest.approx(-0.05)
    assert summary["terminated_fraction"] == pytest.approx(1 / 3)


def test_summarize_reward_samples_rejects_wrong_component_width() -> None:
    with pytest.raises(ValueError, match="reward_components must have shape"):
        summarize_reward_samples(
            rewards=np.zeros((2,), dtype=np.float32),
            reward_components=np.zeros((2, 4), dtype=np.float32),
            terminated=np.zeros((2,), dtype=np.bool_),
            truncated=np.zeros((2,), dtype=np.bool_),
            engine_status=np.zeros((2,), dtype=np.uint8),
        )
