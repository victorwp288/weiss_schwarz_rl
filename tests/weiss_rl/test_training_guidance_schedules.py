from __future__ import annotations

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.training.train_entrypoint import (
    _entropy_coef_for_next_update,
)

from tests.weiss_rl._config_paths import repo_root


def test_entropy_coef_for_next_update_linearly_anneals() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    training = stack.config.training
    assert training is not None

    assert _entropy_coef_for_next_update(training, update_count=0) == pytest.approx(training.entropy_coef)
    midpoint = _entropy_coef_for_next_update(
        training,
        update_count=int(training.entropy_anneal_steps_updates // 2),
    )
    assert midpoint == pytest.approx((training.entropy_coef + training.entropy_anneal_to) / 2.0)
    assert _entropy_coef_for_next_update(
        training,
        update_count=int(training.entropy_anneal_steps_updates * 2),
    ) == pytest.approx(training.entropy_anneal_to)
