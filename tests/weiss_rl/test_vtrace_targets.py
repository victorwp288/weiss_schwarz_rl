from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.learners.vtrace import compute_vtrace_targets

from .vtrace_test_support import fixture_array, load_vtrace_fixture


def test_compute_vtrace_targets_matches_golden_fixture() -> None:
    fixture = load_vtrace_fixture()

    result = compute_vtrace_targets(
        fixture_array(fixture["rewards"]),
        fixture_array(fixture["values"]),
        fixture_array(fixture["discounts"]),
        fixture_array(fixture["behavior_logp"]),
        fixture_array(fixture["target_logp"]),
        rho_bar=fixture["rho_bar"],
        c_bar=fixture["c_bar"],
    )

    np.testing.assert_allclose(result.vs, fixture_array(fixture["expected_vs"]), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        result.pg_advantages, fixture_array(fixture["expected_pg_advantages"]), rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(result.rhos, fixture_array(fixture["expected_rhos"]), rtol=0.0, atol=1e-6)


def test_compute_vtrace_targets_uses_rho_bar_for_values_and_policy_advantages() -> None:
    result = compute_vtrace_targets(
        rewards=np.array([[1.0]], dtype=np.float32),
        values=np.array([[0.5], [0.2]], dtype=np.float32),
        discounts=np.array([[0.0]], dtype=np.float32),
        behavior_logp=np.array([[-10.0]], dtype=np.float32),
        target_logp=np.array([[0.0]], dtype=np.float32),
        rho_bar=1.0,
        c_bar=1.0,
    )

    assert result.rhos[0, 0] > 1_000.0
    assert result.vs[0, 0] == pytest.approx(1.0)
    assert result.pg_advantages[0, 0] == pytest.approx(0.5)


def test_compute_vtrace_targets_caps_extreme_importance_weights_before_float32_overflow() -> None:
    result = compute_vtrace_targets(
        rewards=np.array([[1.0]], dtype=np.float32),
        values=np.array([[0.5], [0.0]], dtype=np.float32),
        discounts=np.array([[0.0]], dtype=np.float32),
        behavior_logp=np.array([[-1_000.0]], dtype=np.float32),
        target_logp=np.array([[1_000.0]], dtype=np.float32),
        rho_bar=1.0,
        c_bar=1.0,
    )

    assert np.isfinite(result.rhos).all()
    assert result.rhos[0, 0] == np.finfo(np.float32).max
    assert result.vs[0, 0] == pytest.approx(1.0)
    assert result.pg_advantages[0, 0] == pytest.approx(0.5)


def test_compute_vtrace_targets_distinguishes_c_bar_recursion_from_rho_bar_policy_clip() -> None:
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    discounts = np.ones((2, 1), dtype=np.float32)
    behavior_logp = np.full((2, 1), -np.log(4.0), dtype=np.float32)
    target_logp = np.zeros((2, 1), dtype=np.float32)

    low_trace_clip = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=0.5,
    )
    high_trace_clip = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=1.0,
        c_bar=1.0,
    )
    swapped_clips = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        behavior_logp,
        target_logp,
        rho_bar=0.5,
        c_bar=1.0,
    )

    expected_rhos = np.full((2, 1), 4.0, dtype=np.float32)
    np.testing.assert_allclose(low_trace_clip.rhos, expected_rhos, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(low_trace_clip.vs, np.array([[2.0], [2.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(high_trace_clip.vs, np.array([[3.0], [2.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(swapped_clips.vs, np.array([[1.5], [1.0]], dtype=np.float32), rtol=0.0, atol=1e-6)

    np.testing.assert_allclose(
        low_trace_clip.pg_advantages,
        np.array([[3.0], [2.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(high_trace_clip.pg_advantages, low_trace_clip.pg_advantages, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        swapped_clips.pg_advantages,
        np.array([[1.0], [1.0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )

    assert low_trace_clip.vs[0, 0] == pytest.approx(2.0)
    assert high_trace_clip.vs[0, 0] == pytest.approx(3.0)
    assert low_trace_clip.pg_advantages[0, 0] == pytest.approx(high_trace_clip.pg_advantages[0, 0])
    assert low_trace_clip.pg_advantages[0, 0] == pytest.approx(3.0)
    assert swapped_clips.pg_advantages[0, 0] == pytest.approx(1.0)
    assert not np.allclose(low_trace_clip.vs, swapped_clips.vs)
    assert not np.allclose(low_trace_clip.pg_advantages, swapped_clips.pg_advantages)


def test_compute_vtrace_targets_requires_t_plus_one_values() -> None:
    with pytest.raises(ValueError, match="one extra bootstrap step"):
        compute_vtrace_targets(
            rewards=np.zeros((2, 3), dtype=np.float32),
            values=np.zeros((2, 3), dtype=np.float32),
            discounts=np.zeros((2, 3), dtype=np.float32),
            behavior_logp=np.zeros((2, 3), dtype=np.float32),
            target_logp=np.zeros((2, 3), dtype=np.float32),
        )
