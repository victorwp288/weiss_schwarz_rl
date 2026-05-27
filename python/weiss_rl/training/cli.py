from __future__ import annotations

import argparse
from pathlib import Path


def build_train_parser() -> argparse.ArgumentParser:
    """Build the path-compatible canonical training parser.

    Defaults, choices, destination names, and hidden compatibility aliases are
    part of the public script contract. Keep changes here paired with
    entrypoint tests and refactor-log notes.
    """

    parser = argparse.ArgumentParser(description="Canonical single-node thesis training entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Expected spec hash or spec bundle SHA-256")
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Stage the built-in public-safe toy catalog/policy bundle instead of probing weiss_sim.",
    )
    parser.add_argument(
        "--config-hash",
        type=str,
        default="",
        help="Expected config_hash256 for contract validation",
    )
    parser.add_argument("--run-label", type=str, default="", help="Optional run directory label override")
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--override",
        "--config-override",
        dest="config_override",
        action="append",
        default=None,
        help="Deterministic config override in KEY=JSON_VALUE form, e.g. training.optimizer.learning_rate=0.0001",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Env count for the single-node training run")
    parser.add_argument("--unroll-length", type=int, default=None, help="Tiny rollout length for the smoke run")
    parser.add_argument("--max-updates", type=int, default=1, help="Number of learner updates to run")
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default=None,
        choices=("train_ordered", "train_async_fast"),
        help="Queue runtime mode: deterministic ordered collection or throughput-oriented async-fast collection",
    )
    parser.add_argument(
        "--profile-timers",
        action="store_true",
        help="Enable cheap runtime/learner timers and record_function ranges without emitting a torch profiler trace",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Emit a torch profiler trace under profiling/torch_profiler/trace.json",
    )
    parser.add_argument("--profile", type=str, default=None, help="Optional simulator profile override")
    parser.add_argument("--device", type=str, default="", help="Optional learner device override")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument(
        "--checkpoint-interval-updates",
        type=int,
        default=None,
        help="Optional checkpoint cadence override for the single-node training run",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--dev-eval-summaries-json",
        type=Path,
        default=None,
        help="Optional dev-eval summaries JSON used to resolve the deterministic final policy set in the manifest",
    )
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help="Completed baseline_noleague run directory used to import the canonical B1 baseline anchor",
    )
    parser.add_argument(
        "--seed-snapshot-run-dir",
        type=Path,
        default=None,
        help="Optional completed run directory whose snapshot registry should be imported into the current training league before update 1",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize model weights/guidance for a fresh run without resuming counters",
    )
    parser.add_argument(
        "--init-schedule-offset-updates",
        type=int,
        default=None,
        help=(
            "Optional guidance-schedule offset for --init-from-checkpoint. "
            "By default fresh warm-starts carry the source checkpoint's cumulative schedule time."
        ),
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume training in-place inside an existing run directory",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Checkpoint path or alias (`latest`/`best`/`observed_best`) to restore before continuing training",
    )
    return parser
