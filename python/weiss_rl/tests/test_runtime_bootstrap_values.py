from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.runtime.components.bootstrap_values import (
    BootstrapArraySpec,
    actor_perspective_discounts,
    concat_bootstrap_array,
    gae_advantages,
    impala_bootstrap_fields,
    reset_before_step,
    runtime_bootstrap_fields,
    runtime_discounts,
    runtime_done_flags,
)


def test_concat_bootstrap_array_coerces_dtype_and_preserves_unroll_order() -> None:
    unroll_a = SimpleNamespace(bootstrap_actor=np.asarray([1.0], dtype=np.float64))
    unroll_b = SimpleNamespace(bootstrap_actor=np.asarray([0.0, 1.0], dtype=np.float64))

    values = concat_bootstrap_array([unroll_a, unroll_b], BootstrapArraySpec("bootstrap_actor", np.dtype(np.int64)))

    assert values.dtype == np.int64
    assert values.tolist() == [1, 0, 1]


def test_impala_bootstrap_fields_preserve_batch_major_hidden_state_order() -> None:
    unroll_a = SimpleNamespace(
        bootstrap_value=np.asarray([0.25], dtype=np.float32),
        bootstrap_obs=np.asarray([[1.0, 2.0]], dtype=np.float32),
        bootstrap_actor=np.asarray([1], dtype=np.int64),
        final_hidden_state=np.asarray([[[10.0, 11.0]]], dtype=np.float32),
    )
    unroll_b = SimpleNamespace(
        bootstrap_value=np.asarray([0.5, 0.75], dtype=np.float32),
        bootstrap_obs=np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        bootstrap_actor=np.asarray([0, 1], dtype=np.int64),
        final_hidden_state=np.asarray([[[20.0, 21.0]], [[30.0, 31.0]]], dtype=np.float32),
    )

    fields = impala_bootstrap_fields([unroll_a, unroll_b])

    assert fields.value.tolist() == pytest.approx([0.25, 0.5, 0.75])
    assert np.allclose(fields.obs, np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
    assert fields.actor.tolist() == [1, 0, 1]
    assert np.allclose(
        fields.final_hidden_state,
        np.asarray([[[10.0, 11.0]], [[20.0, 21.0]], [[30.0, 31.0]]], dtype=np.float32),
    )


def test_runtime_bootstrap_fields_skip_impala_only_observations_and_hidden_state() -> None:
    unroll = SimpleNamespace(
        bootstrap_value=np.asarray([0.25], dtype=np.float32),
        bootstrap_actor=np.asarray([1], dtype=np.int64),
        bootstrap_obs=np.asarray([[99.0]], dtype=np.float32),
        final_hidden_state=np.asarray([[[88.0]]], dtype=np.float32),
    )

    fields = runtime_bootstrap_fields([unroll])

    assert fields.value.tolist() == pytest.approx([0.25])
    assert fields.actor.tolist() == [1]
    assert not hasattr(fields, "obs")
    assert not hasattr(fields, "final_hidden_state")


def test_done_reset_and_discounts_keep_actor_perspective_bootstrap_semantics() -> None:
    terminated = np.asarray([[False], [False], [False]], dtype=np.bool_)
    truncated = np.asarray([[False], [True], [False]], dtype=np.bool_)
    done = runtime_done_flags(terminated=terminated, truncated=truncated)

    discounts = runtime_discounts(
        done=done,
        to_play_seat=np.asarray([[0], [1], [1]], dtype=np.int64),
        bootstrap_actor=np.asarray([0], dtype=np.int64),
        gamma=0.99,
    )

    assert done[:, 0].tolist() == [False, True, False]
    assert reset_before_step(done)[:, 0].tolist() == [False, False, True]
    assert discounts[:, 0].tolist() == pytest.approx([-0.99, 0.0, -0.99])


def test_gae_advantages_still_use_behavior_bootstrap_value() -> None:
    advantages = gae_advantages(
        rewards=np.asarray([[0.0]], dtype=np.float32),
        values=np.asarray([[0.0]], dtype=np.float32),
        bootstrap_value=np.asarray([0.25], dtype=np.float32),
        discounts=np.asarray([[1.0]], dtype=np.float32),
        gae_lambda=1.0,
    )

    assert advantages[:, 0].tolist() == pytest.approx([0.25])


def test_actor_perspective_discounts_reject_live_invalid_bootstrap_actor() -> None:
    with pytest.raises(ValueError, match="continuation actor"):
        actor_perspective_discounts(
            done=np.asarray([[False]], dtype=np.bool_),
            to_play_seat=np.asarray([[0]], dtype=np.int64),
            bootstrap_actor=np.asarray([-1], dtype=np.int64),
            gamma=0.99,
        )
