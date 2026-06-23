from __future__ import annotations

import numpy.testing as npt
import torch

from .heuristic_public_model_bias_test_support import (
    _actor_packed_scores,
    _best_actions_from_packed_scores,
    _public_bias_model,
    _public_bias_rows,
)


def test_structured_model_public_bias_guides_live_packed_scores_toward_b2_choices() -> None:
    rows = _public_bias_rows()
    torch.manual_seed(0)
    model = _public_bias_model(
        spec_bundle=rows.spec_bundle,
        public_heuristic_logit_bias_scale=0.0,
        public_heuristic_actor_logit_bias_scale=100.0,
    )

    chosen = _best_actions_from_packed_scores(rows, _actor_packed_scores(model, rows))
    expected = rows.policy.choose_actions_from_meta_batch(rows.obs_rows, rows.legal_ids, rows.offsets, rows.meta)
    npt.assert_array_equal(chosen, expected)
