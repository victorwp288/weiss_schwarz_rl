from __future__ import annotations

import pytest
from weiss_rl.training.batches import collect_training_batch

from .training_batches_test_support import RuntimeRecorder, rewards_config, training_config


def test_collect_training_batch_dispatches_impala_with_vtrace_arguments() -> None:
    runtime = RuntimeRecorder()

    batch = collect_training_batch(
        runtime=runtime,
        algorithm="impala_vtrace_structured_v1",
        training_config=training_config(),
        rewards_config=rewards_config(),
    )

    assert batch == "impala_batch"
    assert runtime.calls == [
        (
            "impala",
            {
                "gamma": 0.99,
                "truncation_reward": -0.5,
                "truncation_bootstrap_value": True,
                "vtrace_rho_bar": 1.25,
                "vtrace_c_bar": 0.75,
            },
        )
    ]


def test_collect_training_batch_dispatches_ppo_with_gae_arguments() -> None:
    runtime = RuntimeRecorder()

    batch = collect_training_batch(
        runtime=runtime,
        algorithm="ppo_lite_masked_v1",
        training_config=training_config(),
        rewards_config=rewards_config(),
    )

    assert batch == "ppo_batch"
    assert runtime.calls == [
        (
            "ppo",
            {
                "gamma": 0.99,
                "gae_lambda": 0.92,
                "truncation_reward": -0.5,
                "truncation_bootstrap_value": True,
            },
        )
    ]


def test_collect_training_batch_rejects_unknown_algorithm() -> None:
    with pytest.raises(RuntimeError, match="Unsupported training.algorithm: unknown"):
        collect_training_batch(
            runtime=RuntimeRecorder(),
            algorithm="unknown",
            training_config=training_config(),
            rewards_config=rewards_config(),
        )
