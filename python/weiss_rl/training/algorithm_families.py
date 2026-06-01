from __future__ import annotations

from typing import Literal

TrainingAlgorithmFamily = Literal["impala", "ppo"]

IMPALA_ALGORITHMS = frozenset(
    {
        "impala_vtrace_gru",
        "impala_vtrace_ff",
        "structured_v2",
        "impala_vtrace_structured_v1",
    }
)
PPO_ALGORITHMS = frozenset({"ppo_lite_masked_v1"})
STRUCTURED_VTRACE_ALGORITHMS = frozenset({"structured_v2", "impala_vtrace_structured_v1"})


def training_algorithm_family(algorithm: str) -> TrainingAlgorithmFamily:
    if algorithm in IMPALA_ALGORITHMS:
        return "impala"
    if algorithm in PPO_ALGORITHMS:
        return "ppo"
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


__all__ = [
    "IMPALA_ALGORITHMS",
    "PPO_ALGORITHMS",
    "STRUCTURED_VTRACE_ALGORITHMS",
    "TrainingAlgorithmFamily",
    "training_algorithm_family",
]
