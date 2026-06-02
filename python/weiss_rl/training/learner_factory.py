from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from torch import nn

from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner
from weiss_rl.training.algorithm_families import training_algorithm_family
from weiss_rl.training.learner_kwargs import (
    common_training_learner_kwargs,
    impala_training_learner_kwargs,
    ppo_training_learner_kwargs,
)


def build_training_learner(
    *,
    algorithm: str,
    model: Any,
    compiled_model: nn.Module | None,
    training_config: Any,
    training_paths: Any,
    pass_action_id: int,
    checkpoint_interval_updates: int,
    impala_learner_cls: Callable[..., Any] = ImpalaLearner,
    ppo_lite_learner_cls: Callable[..., Any] = PpoLiteLearner,
) -> ImpalaLearner | PpoLiteLearner:
    """Construct the training learner for the configured algorithm family."""

    common_kwargs = common_training_learner_kwargs(
        model=model,
        compiled_model=compiled_model,
        training_config=training_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=checkpoint_interval_updates,
    )
    family = training_algorithm_family(algorithm)
    if family == "impala":
        return cast(
            ImpalaLearner,
            impala_learner_cls(
                **common_kwargs,
                **impala_training_learner_kwargs(training_config),
            ),
        )
    if family == "ppo":
        return cast(
            PpoLiteLearner,
            ppo_lite_learner_cls(
                **common_kwargs,
                **ppo_training_learner_kwargs(training_config),
            ),
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")
