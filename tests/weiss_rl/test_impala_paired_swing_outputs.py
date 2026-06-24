from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.impala.auxiliary.paired_swing_outputs import build_paired_swing_auxiliary_metrics


def test_build_paired_swing_auxiliary_metrics_preserves_flags_and_metric_precedence() -> None:
    metrics = build_paired_swing_auxiliary_metrics(
        weighted_loss=torch.tensor(0.75),
        coef=0.5,
        margin=0.25,
        positive_action_source="teacher_action",
        negative_action_source="actions",
        loss_scope="label_mean",
        compare_to=" Top_Other ",
        margin_retention_coef=0.1,
        margin_retention_margin=0.2,
        top_action_retention_coef=0.3,
        top_action_retention_margin=0.4,
        swing_metrics={"paired_swing_rows": 2.0, "paired_swing_weighted_loss": 99.0},
    )

    assert metrics["loss"] == pytest.approx(0.75)
    assert metrics["paired_swing_weighted_loss"] == 99.0
    assert metrics["paired_swing_coef"] == pytest.approx(0.5)
    assert metrics["paired_swing_margin"] == pytest.approx(0.25)
    assert metrics["paired_swing_positive_action_source_teacher"] == 1.0
    assert metrics["paired_swing_negative_action_source_teacher"] == 0.0
    assert metrics["paired_swing_loss_scope_label_mean"] == 1.0
    assert metrics["paired_swing_compare_to_top_other"] == 1.0
    assert metrics["paired_swing_margin_retention_coef"] == pytest.approx(0.1)
    assert metrics["paired_swing_top_action_retention_margin"] == pytest.approx(0.4)
    assert metrics["paired_swing_rows"] == 2.0
