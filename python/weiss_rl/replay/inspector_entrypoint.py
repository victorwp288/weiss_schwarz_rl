from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.replay.inspector import inspect_replay_bundle
from weiss_rl.replay.inspector_report import format_replay_inspection_report, write_replay_inspection_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two policies on the recorded states from a deterministic replay bundle"
    )
    parser.add_argument("--bundle", type=Path, required=True, help="Replay bundle zip to inspect")
    parser.add_argument(
        "--stack-config",
        type=Path,
        required=True,
        help="Stack config used to reconstruct the policy model architecture",
    )
    parser.add_argument(
        "--policy-a",
        type=str,
        required=True,
        help="Policy A identifier or weights path",
    )
    parser.add_argument(
        "--policy-b",
        type=str,
        required=True,
        help="Policy B identifier or weights path",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run dir used to resolve snapshot-registry policy IDs and relative weights paths",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help=(
            "Optional snapshot registry used to resolve policy IDs "
            "(default: <run-dir>/training/snapshots/registry.json)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-difference replay steps to report (default: 10)",
    )
    parser.add_argument(
        "--top-actions",
        type=int,
        default=5,
        help="Number of highest-delta legal actions to include per reported step (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured report JSON to stdout instead of the human-readable text report",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to persist the structured JSON report",
    )
    parser.add_argument(
        "--accept-snapshot-config-hash",
        action="append",
        default=[],
        help=(
            "Additional snapshot config_hash256 to accept when intentionally comparing imported snapshots "
            "from a different training stack. Repeatable."
        ),
    )
    parser.add_argument(
        "--opponent-context-policy-id",
        type=str,
        default=None,
        help=(
            "Optional opponent policy id used to supply opponent_context_index when scoring model policies. "
            "Use this for opponent-conditioned adapter diagnostics."
        ),
    )
    parser.add_argument(
        "--require-opponent-context-index",
        action="store_true",
        help="Fail if a model policy cannot map --opponent-context-policy-id to a nonzero context index.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if args.top_actions <= 0:
        parser.error("--top-actions must be >= 1")

    report = inspect_replay_bundle(
        bundle_path=args.bundle,
        stack=args.stack_config,
        policy_a=args.policy_a,
        policy_b=args.policy_b,
        run_dir=args.run_dir,
        snapshot_registry_path=args.snapshot_registry_json,
        top_k=args.top_k,
        top_actions=args.top_actions,
        accepted_snapshot_config_hashes=args.accept_snapshot_config_hash,
        opponent_context_policy_id=args.opponent_context_policy_id,
        require_opponent_context_index=args.require_opponent_context_index,
    )
    if args.report_json is not None:
        write_replay_inspection_report(args.report_json, report)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_replay_inspection_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
