from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weiss_rl.diagnostics.artifact_hygiene import default_artifact_roots, format_findings, run_artifact_hygiene_scan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Artifact hygiene gate for tracked files and generated artifacts")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for tracked-file scanning (default: current working directory)",
    )
    parser.add_argument(
        "--artifact-root",
        dest="artifact_roots",
        action="append",
        type=Path,
        default=None,
        help="Artifact root to scan recursively. Repeat to add multiple roots. Defaults to runs/ when present.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    artifact_roots = (
        tuple(args.artifact_roots) if args.artifact_roots is not None else default_artifact_roots(repo_root)
    )
    summary = run_artifact_hygiene_scan(repo_root=repo_root, artifact_roots=artifact_roots)
    if summary.findings:
        print("Artifact hygiene scan failed:", file=sys.stderr)
        print(format_findings(summary.findings), file=sys.stderr)
        return 1
    print(
        "Artifact hygiene scan passed. "
        f"Tracked files scanned: {summary.repo_file_count}. "
        f"Artifact files scanned: {summary.artifact_file_count}. "
        f"Replay bundles scanned: {summary.replay_bundle_count}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
