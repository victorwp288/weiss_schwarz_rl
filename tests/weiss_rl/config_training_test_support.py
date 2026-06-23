from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def copy_section(body: dict[str, object], key: str) -> dict[str, object]:
    return dict(cast(Mapping[str, object], body[key]))


def training_body() -> dict[str, object]:
    return {
        "algorithm": "impala_vtrace_gru",
        "rollout": {"unroll_length": 4, "batch_unrolls_per_update": 2},
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "grad_norm_clip": 0.5,
            "value_loss_coef": 0.25,
        },
        "exploration": {
            "entropy_coef": 0.01,
            "entropy_anneal_to": 0.001,
            "entropy_anneal_steps_updates": 100,
        },
        "precision": {
            "mixed_precision": False,
            "compile_learner": False,
            "masking_math_float32": True,
        },
        "checkpointing": {
            "checkpoint_interval_updates": 10,
            "snapshot_interval_updates": 20,
            "actor_reload_interval_updates": 5,
        },
        "vtrace": {"rho_bar": 1.0, "c_bar": 1.0},
    }
