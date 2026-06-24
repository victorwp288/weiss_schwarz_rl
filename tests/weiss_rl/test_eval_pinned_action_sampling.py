from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.eval import sample_action_pinned, select_action_argmax_pinned
from weiss_rl.eval.sampling.rng_pcg32 import Pcg32XshRrV1

from .eval_sampler_test_support import PINNED_SAMPLER_ACTIONS, StubFloatRng, expected_single_row_logp


def test_sample_action_pinned_matches_anchor_sequence() -> None:
    logits = np.array([0.0, -3.0, 0.5, -2.0, 1.0, 1.5, -4.0, -0.5], dtype=np.float32)
    legal_ids = np.array([0, 2, 4, 5, 7], dtype=np.uint32)
    rng = Pcg32XshRrV1(20260316)

    actions = [sample_action_pinned(logits, legal_ids, rng=rng)[0] for _ in range(10)]

    assert actions == PINNED_SAMPLER_ACTIONS


def test_sample_action_pinned_returns_masking_core_logp() -> None:
    logits = np.array([0.25, -9.0, -0.5, 8.0, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([0, 2, 4, 5], dtype=np.uint32)
    rng = StubFloatRng(0.72)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 4
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_temperature_scales_model_distribution_and_logp() -> None:
    logits = np.array([0.0, 1.0, -10.0], dtype=np.float32)
    legal_ids = np.array([0, 1], dtype=np.uint32)
    rng = StubFloatRng(0.2)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng, temperature=0.25)

    assert action == 1
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action, temperature=0.25), abs=1e-6)
    assert rng.calls == 1


def test_select_action_argmax_pinned_uses_legal_argmax_without_rng() -> None:
    logits = np.array([2.0, 99.0, 0.5, 3.0, 3.0], dtype=np.float32)
    legal_ids = np.array([0, 2, 3, 4], dtype=np.uint32)

    action, logp = select_action_argmax_pinned(logits, legal_ids)

    assert action == 3
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action), abs=1e-6)


def test_select_action_argmax_pinned_empty_legal_returns_pass() -> None:
    logits = np.array([0.25, -0.5, 1.5, 0.75], dtype=np.float32)

    action, logp = select_action_argmax_pinned(logits, np.array([], dtype=np.uint32), pass_action_id=3)

    assert action == 3
    assert logp == pytest.approx(0.0)


def test_sample_action_pinned_rejects_non_finite_legal_logits() -> None:
    logits = np.array([0.25, np.nan, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([1, 2], dtype=np.uint32)
    rng = StubFloatRng(0.25)

    with pytest.raises(ValueError, match="legal logits must be finite"):
        sample_action_pinned(logits, legal_ids, rng=rng)

    assert rng.calls == 0


def test_sample_action_pinned_ignores_non_finite_illegal_logits() -> None:
    logits = np.array([0.25, np.nan, 1.5, np.inf, 0.75], dtype=np.float32)
    legal_ids = np.array([0, 2, 4], dtype=np.uint32)
    rng = StubFloatRng(0.6)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 2
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_empty_legal_returns_pass_without_rng_draw() -> None:
    logits = np.array([0.25, -0.5, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([], dtype=np.uint32)
    rng = StubFloatRng(0.4)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng, pass_action_id=3)

    assert action == 3
    assert logp == pytest.approx(0.0)
    assert rng.calls == 0


def test_sample_action_pinned_singleton_legal_set_consumes_rng() -> None:
    logits = np.array([0.25, -0.5, 1.5, 0.75], dtype=np.float32)
    legal_ids = np.array([2], dtype=np.uint32)
    rng = StubFloatRng(0.999999)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 2
    assert logp == pytest.approx(0.0)
    assert rng.calls == 1


def test_sample_action_pinned_skips_zero_prob_plateau_at_draw_zero() -> None:
    logits = np.array([-1000.0, 0.0, 0.0], dtype=np.float32)
    legal_ids = np.array([0, 1, 2], dtype=np.uint32)
    rng = StubFloatRng(0.0)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 1
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_pins_final_bin_at_draw_one_with_zero_prob_tail() -> None:
    logits = np.array([0.0, -1000.0, -1000.0], dtype=np.float32)
    legal_ids = np.array([0, 1, 2], dtype=np.uint32)
    rng = StubFloatRng(1.0)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 2
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_rounding_guard_pins_final_cdf_bin() -> None:
    logits = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    legal_ids = np.array([0, 1, 2, 3], dtype=np.uint32)
    rng = StubFloatRng(1.0)

    action, logp = sample_action_pinned(logits, legal_ids, rng=rng)

    assert action == 3
    assert logp == pytest.approx(expected_single_row_logp(logits, legal_ids, action), abs=1e-6)
    assert rng.calls == 1


def test_sample_action_pinned_empty_rows_do_not_consume_rng_but_non_empty_rows_do() -> None:
    logits = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    rng = StubFloatRng(0.2)

    pass_action, pass_logp = sample_action_pinned(
        logits,
        np.array([], dtype=np.uint32),
        rng=rng,
        pass_action_id=1,
    )
    sampled_action, sampled_logp = sample_action_pinned(
        logits,
        np.array([0, 2, 3], dtype=np.uint32),
        rng=rng,
    )

    assert pass_action == 1
    assert pass_logp == pytest.approx(0.0)
    assert sampled_action == 2
    assert sampled_logp == pytest.approx(
        expected_single_row_logp(logits, np.array([0, 2, 3], dtype=np.uint32), sampled_action),
        abs=1e-6,
    )
    assert rng.calls == 1
