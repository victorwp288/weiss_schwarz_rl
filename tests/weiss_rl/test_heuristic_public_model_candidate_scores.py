from __future__ import annotations

import numpy.testing as npt
import torch

from .heuristic_public_model_bias_test_support import (
    _best_actions_from_packed_scores,
    _public_bias_model,
    _public_bias_rows,
)


def test_structured_model_public_heuristic_scores_match_b2_batch_meta_choices() -> None:
    rows = _public_bias_rows()
    model = _public_bias_model(spec_bundle=rows.spec_bundle)

    scores = (
        model.score_packed_public_heuristic_candidates(
            torch.as_tensor(rows.obs_rows, dtype=torch.float32),
            rows.legal_batch,
        )
        .detach()
        .cpu()
        .numpy()
    )

    chosen = _best_actions_from_packed_scores(rows, scores)
    expected = rows.policy.choose_actions_from_meta_batch(rows.obs_rows, rows.legal_ids, rows.offsets, rows.meta)
    npt.assert_array_equal(chosen, expected)
