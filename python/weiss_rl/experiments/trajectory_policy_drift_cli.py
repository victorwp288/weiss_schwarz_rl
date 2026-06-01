"""Parser construction for trajectory policy-drift diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_trajectory_policy_drift_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score identical replay trajectory rows with multiple policies and report drift"
    )
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="Policy spec as LABEL|RUN_DIR|CHECKPOINT_RELPATH. Repeat for direct/update checkpoints.",
    )
    parser.add_argument("--reference-label", default=None, help="Policy label used as the drift reference")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=25)
    return parser
