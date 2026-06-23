from __future__ import annotations

import numpy as np
import numpy.testing as npt
import torch

from .heuristic_public_model_bias_test_support import (
    _actor_packed_scores,
    _public_bias_model,
    _public_bias_rows,
)


def test_structured_model_public_bias_family_gate_only_affects_selected_families() -> None:
    rows = _public_bias_rows()
    torch.manual_seed(0)
    baseline_model = _public_bias_model(
        spec_bundle=rows.spec_bundle,
        public_heuristic_logit_bias_scale=0.0,
        public_heuristic_actor_logit_bias_scale=0.0,
    )
    torch.manual_seed(0)
    gated_model = _public_bias_model(
        spec_bundle=rows.spec_bundle,
        public_heuristic_logit_bias_scale=0.0,
        public_heuristic_actor_logit_bias_scale=100.0,
        public_heuristic_logit_bias_families=("attack",),
    )

    baseline_scores = _actor_packed_scores(baseline_model, rows)
    gated_scores = _actor_packed_scores(gated_model, rows)

    npt.assert_allclose(
        gated_scores[int(rows.offsets[1]) : int(rows.offsets[2])],
        baseline_scores[int(rows.offsets[1]) : int(rows.offsets[2])],
    )
    npt.assert_allclose(
        gated_scores[int(rows.offsets[2]) : int(rows.offsets[3])],
        baseline_scores[int(rows.offsets[2]) : int(rows.offsets[3])],
    )
    assert not np.allclose(
        gated_scores[int(rows.offsets[0]) : int(rows.offsets[1])],
        baseline_scores[int(rows.offsets[0]) : int(rows.offsets[1])],
    )
