"""Runtime orchestration for publishing checkpoint snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from weiss_rl.training.checkpointing.publish import publish_checkpoint_snapshot


@dataclass(frozen=True, slots=True)
class CheckpointPublishRunResult:
    result: dict[str, Any]


def run_checkpoint_publish(args: argparse.Namespace) -> CheckpointPublishRunResult:
    result = publish_checkpoint_snapshot(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint_path,
        policy_id=args.policy_id,
        pin=bool(args.pin),
        replace=bool(args.replace),
    )
    return CheckpointPublishRunResult(result=result)


__all__ = ["CheckpointPublishRunResult", "run_checkpoint_publish"]
