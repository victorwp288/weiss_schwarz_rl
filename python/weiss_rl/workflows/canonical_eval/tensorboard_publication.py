from __future__ import annotations

import sys
from typing import Any

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState
from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs


def begin_canonical_eval_tensorboard_logging(*, run_state: CanonicalEvalRunState, dependencies: Any) -> None:
    tensorboard_logger = run_state.tensorboard_logger
    if not tensorboard_logger.enabled:
        unavailable_reason = dependencies.tensorboard_unavailable_reason_fn()
        print(
            "TensorBoard logging is disabled for eval: "
            + ("SummaryWriter unavailable" if unavailable_reason is None else unavailable_reason),
            file=sys.stderr,
        )
        return
    tensorboard_logger.log_text("eval/run/manifest", run_state.manifest)


def publish_canonical_eval_tensorboard_summaries(
    *,
    layout: Any,
    tensorboard_logger: Any,
    final_eval_payload: dict[str, Any],
    supplemental: CanonicalEvalSupplementalOutputs,
) -> None:
    if not tensorboard_logger.enabled:
        return
    tensorboard_logger.log_final_eval_summary(final_eval_payload, step=0)
    if supplemental.metagame_payload is not None:
        tensorboard_logger.log_metagame_summary(
            supplemental.metagame_payload,
            metagame_dir=layout.metagame_dir,
            step=0,
        )
    if supplemental.readiness_payload is not None:
        tensorboard_logger.log_paper_readiness(supplemental.readiness_payload, step=0)


__all__ = [
    "begin_canonical_eval_tensorboard_logging",
    "publish_canonical_eval_tensorboard_summaries",
]
