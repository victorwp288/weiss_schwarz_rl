from __future__ import annotations

# ruff: noqa: F401
# This path-based script intentionally re-exports helper names used by tests and
# by the compatibility-oriented training entrypoint hook modules.
import sys
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
install_train_entrypoint_compat_exports(globals())
install_train_entrypoint_aliases(globals(), checkpoint_guard_helpers=globals()["_checkpoint_guard_helpers"])


def _entrypoint_api() -> Any:
    return _SCRIPT_COMPAT_API if _SCRIPT_COMPAT_API is not None else sys.modules[__name__]


_model_guidance_payload = globals()["model_guidance_payload"]


install_train_entrypoint_wrappers(
    globals(),
    entrypoint_api=_entrypoint_api,
    periodic_dev_eval_runner_base=globals()["PeriodicDevEvalRunner"],
    promotion_gate_runner_base=globals()["PromotionGateRunner"],
    random_legal_policy_id=_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,
)


def main() -> None:
    run_train_main(_entrypoint_api())


if __name__ == "__main__":
    main()
