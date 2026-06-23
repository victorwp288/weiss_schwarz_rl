"""Runnable training module with the legacy hook surface kept contained."""

from __future__ import annotations

# ruff: noqa: F401
import sys
from collections.abc import MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint.cli import run_train_main
from weiss_rl.training.train_entrypoint.compat import (
    install_train_entrypoint_aliases,
    install_train_entrypoint_compat_exports,
)
from weiss_rl.training.train_entrypoint.wrappers import install_train_entrypoint_wrappers

_PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"
_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT = "baseline_checkpoint.pt"
_SCRIPT_COMPAT_API: Any | None = None


def _entrypoint_api() -> Any:
    return _SCRIPT_COMPAT_API if _SCRIPT_COMPAT_API is not None else sys.modules[__name__]


def _install_script_compatibility(namespace: MutableMapping[str, Any]) -> None:
    install_train_entrypoint_compat_exports(namespace)
    install_train_entrypoint_aliases(
        namespace,
        checkpoint_guard_helpers=namespace["_checkpoint_guard_helpers"],
    )
    install_train_entrypoint_wrappers(
        namespace,
        entrypoint_api=_entrypoint_api,
        periodic_dev_eval_runner_base=namespace["PeriodicDevEvalRunner"],
        promotion_gate_runner_base=namespace["PromotionGateRunner"],
        random_legal_policy_id=_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,
    )


_install_script_compatibility(globals())
_model_guidance_payload = globals()["model_guidance_payload"]


def main() -> None:
    run_train_main(_entrypoint_api())


if __name__ == "__main__":
    main()
