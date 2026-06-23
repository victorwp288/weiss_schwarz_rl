from __future__ import annotations

import pytest

from tests.weiss_rl.impala_paired_auxiliary_test_support import make_paired_swing_dense_case


def test_impala_learner_paired_swing_auxiliary_dense_path_preserves_weighted_metrics() -> None:
    _action_catalog, learner, batch = make_paired_swing_dense_case()

    loss, metrics, context = learner._paired_swing_loss_and_metrics(
        batch,
        margin=0.25,
        coef=0.5,
        positive_action_source="teacher_action",
        negative_action_source="actions",
    )

    assert loss.detach().item() == pytest.approx(0.625)
    assert metrics["paired_swing_weighted_loss"] == pytest.approx(0.625)
    assert metrics["paired_swing_margin"] == pytest.approx(0.25)
    assert metrics["paired_swing_coef"] == pytest.approx(0.5)
    assert metrics["paired_swing_positive_action_source_teacher"] == 1.0
    assert metrics["paired_swing_negative_action_source_teacher"] == 0.0
    assert metrics["paired_swing_rows"] == 1.0
    assert context["paired_swing_margins"].tolist() == pytest.approx([-1.0])
