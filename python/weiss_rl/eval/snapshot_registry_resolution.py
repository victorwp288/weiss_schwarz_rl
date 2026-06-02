"""Resolve source run directories for snapshot-registry backed eval."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from weiss_rl.eval.policies.set import LEGACY_NO_LEAGUE_POLICY_ID, NO_LEAGUE_POLICY_ID
from weiss_rl.league.registry import SnapshotMeta, SnapshotRegistry


@dataclass(slots=True)
class SnapshotRegistrySource:
    path: Path
    registry: SnapshotRegistry
    snapshots_by_policy_id: dict[str, SnapshotMeta]

    @classmethod
    def load(cls, path: Path) -> SnapshotRegistrySource:
        registry = SnapshotRegistry.load(path)
        return cls(
            path=Path(path),
            registry=registry,
            snapshots_by_policy_id={snapshot.policy_id: snapshot for snapshot in registry.snapshots},
        )

    def snapshot_for_policy_id(self, policy_id: str) -> SnapshotMeta | None:
        return snapshot_by_policy_id_or_imported_seed_suffix(
            snapshots_by_policy_id=self.snapshots_by_policy_id,
            policy_id=policy_id,
            registry_path=self.path,
        )

    def resolve_run_dir(self, *, run_dir: Path, policy_ids: list[str]) -> Path:
        return resolve_snapshot_registry_run_dir(
            run_dir=run_dir,
            registry_path=self.path,
            registry=self.registry,
            policy_ids=policy_ids,
        )


def resolve_snapshot_registry_run_dir(
    *,
    run_dir: Path,
    registry_path: Path,
    registry: SnapshotRegistry,
    policy_ids: list[str],
) -> Path:
    resolved_run_dir = Path(run_dir).resolve()
    resolved_registry_path = Path(registry_path).resolve()
    resolution_snapshots = snapshot_registry_resolution_snapshots(registry=registry, policy_ids=policy_ids)
    canonical_candidate = canonical_snapshot_registry_run_dir(resolved_registry_path)
    canonical_search_root: Path | None = None
    if canonical_candidate is not None:
        if not resolution_snapshots or run_dir_contains_registry_snapshots(canonical_candidate, resolution_snapshots):
            return canonical_candidate
        canonical_search_root = canonical_candidate.parent
    if not resolution_snapshots:
        return resolved_run_dir

    candidate_run_dirs: list[Path] = []
    for candidate in [resolved_run_dir, resolved_registry_path.parent, *resolved_registry_path.parents]:
        if run_dir_matches_registry_snapshots(candidate, resolution_snapshots):
            candidate_run_dirs.append(candidate.resolve())
    search_roots = [resolved_run_dir.parent, resolved_registry_path.parent]
    if canonical_search_root is not None:
        search_roots.append(canonical_search_root)
        canonical_common_search_root = common_search_root([resolved_run_dir, canonical_search_root])
        if canonical_common_search_root is not None and is_recursive_registry_search_root(canonical_common_search_root):
            search_roots.append(canonical_common_search_root)
    shared_search_root = common_search_root([resolved_run_dir, resolved_registry_path])
    if (
        shared_search_root is not None
        and is_recursive_registry_search_root(shared_search_root)
        and should_include_common_search_root(search_roots=search_roots, common_search_root=shared_search_root)
    ):
        search_roots.append(shared_search_root)
    for search_root in unique_paths(
        [search_root for search_root in search_roots if is_recursive_registry_search_root(search_root)]
    ):
        candidate_run_dirs.extend(
            discover_snapshot_registry_run_dirs(
                search_root=search_root,
                snapshots=resolution_snapshots,
            )
        )

    unique_candidates = unique_paths(candidate_run_dirs)
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    if len(unique_candidates) > 1:
        if any(candidate == resolved_run_dir for candidate in unique_candidates):
            return resolved_run_dir
        matches = ", ".join(candidate.as_posix() for candidate in unique_candidates)
        raise RuntimeError(
            "Could not uniquely resolve the source run for snapshot registry "
            f"{resolved_registry_path}; matching run directories: {matches}"
        )
    raise FileNotFoundError(
        "Could not resolve the source run for snapshot registry "
        f"{resolved_registry_path}. Provide the canonical <run>/training/snapshots/registry.json path "
        "or place the copied registry next to matching snapshot artifacts."
    )


def canonical_snapshot_registry_run_dir(registry_path: Path) -> Path | None:
    resolved_registry_path = Path(registry_path).resolve()
    if resolved_registry_path.parts[-3:] != ("training", "snapshots", "registry.json"):
        return None
    return resolved_registry_path.parent.parent.parent


def snapshot_registry_resolution_snapshots(
    *,
    registry: SnapshotRegistry,
    policy_ids: list[str],
) -> list[SnapshotMeta]:
    snapshots_by_policy_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    requested_snapshots: list[SnapshotMeta] = []
    seen_policy_ids: set[str] = set()

    def _append_snapshot(policy_id: str) -> None:
        snapshot = snapshot_by_policy_id_or_imported_seed_suffix(
            snapshots_by_policy_id=snapshots_by_policy_id,
            policy_id=policy_id,
            registry_path=None,
        )
        if snapshot is None or snapshot.policy_id in seen_policy_ids:
            return
        requested_snapshots.append(snapshot)
        seen_policy_ids.add(snapshot.policy_id)

    for policy_id in policy_ids:
        _append_snapshot(policy_id)
    if NO_LEAGUE_POLICY_ID in policy_ids:
        _append_snapshot(LEGACY_NO_LEAGUE_POLICY_ID)
    if requested_snapshots:
        return requested_snapshots
    return spaced_snapshot_resolution_samples(registry.snapshots)


def snapshot_by_policy_id_or_imported_seed_suffix(
    *,
    snapshots_by_policy_id: Mapping[str, SnapshotMeta],
    policy_id: str,
    registry_path: Path | None,
) -> SnapshotMeta | None:
    snapshot = snapshots_by_policy_id.get(policy_id)
    if snapshot is not None:
        return snapshot
    normalized = str(policy_id).strip()
    if not normalized.startswith("seed_"):
        return None
    suffix = f"_{normalized}"
    matches = [
        candidate
        for candidate_id, candidate in snapshots_by_policy_id.items()
        if str(candidate_id).startswith("seed_") and str(candidate_id).endswith(suffix)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    registry_hint = "" if registry_path is None else f" in snapshot registry {registry_path}"
    match_ids = ", ".join(sorted(snapshot.policy_id for snapshot in matches))
    raise RuntimeError(f"Ambiguous imported seed policy suffix {policy_id!r}{registry_hint}; matches: {match_ids}")


def spaced_snapshot_resolution_samples(snapshots: list[SnapshotMeta]) -> list[SnapshotMeta]:
    if len(snapshots) <= 3:
        return list(snapshots)
    middle_index = len(snapshots) // 2
    return [
        snapshots[0],
        snapshots[middle_index],
        snapshots[-1],
    ]


def discover_snapshot_registry_run_dirs(
    *,
    search_root: Path,
    snapshots: list[SnapshotMeta],
) -> list[Path]:
    resolved_search_root = Path(search_root).resolve()
    if not resolved_search_root.is_dir() or not snapshots:
        return []
    first_snapshot = snapshots[0]
    pattern = f"**/training/snapshots/{first_snapshot.policy_id}/weights.pt"
    candidates: list[Path] = []
    for weights_path in resolved_search_root.glob(pattern):
        candidate_run_dir = weights_path.parent.parent.parent.parent
        if run_dir_matches_registry_snapshots(candidate_run_dir, snapshots):
            candidates.append(candidate_run_dir.resolve())
    return unique_paths(candidates)


def run_dir_matches_registry_snapshots(
    candidate_run_dir: Path,
    snapshots: list[SnapshotMeta],
) -> bool:
    resolved_candidate = Path(candidate_run_dir).resolve()
    for snapshot in snapshots:
        weights_path = resolved_candidate / snapshot.path
        if not weights_path.is_file():
            return False
        expected_sha256 = str(snapshot.weights_sha256).strip().lower()
        if expected_sha256 and sha256_file(weights_path) != expected_sha256:
            return False
    return True


def run_dir_contains_registry_snapshots(
    candidate_run_dir: Path,
    snapshots: list[SnapshotMeta],
) -> bool:
    resolved_candidate = Path(candidate_run_dir).resolve()
    for snapshot in snapshots:
        if not (resolved_candidate / snapshot.path).is_file():
            return False
    return True


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = resolved.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def common_search_root(paths: list[Path]) -> Path | None:
    resolved_paths = [Path(path).resolve() for path in paths]
    if not resolved_paths:
        return None
    common_parts = list(resolved_paths[0].parts)
    for path in resolved_paths[1:]:
        shared_prefix_len = 0
        for left, right in zip(common_parts, path.parts, strict=False):
            if left != right:
                break
            shared_prefix_len += 1
        common_parts = common_parts[:shared_prefix_len]
        if not common_parts:
            return None
    return Path(*common_parts)


def is_recursive_registry_search_root(path: Path) -> bool:
    resolved_path = Path(path).resolve()
    anchor = resolved_path.anchor
    if anchor and resolved_path == Path(anchor):
        return False
    return True


def should_include_common_search_root(*, search_roots: list[Path], common_search_root: Path) -> bool:
    resolved_common_search_root = Path(common_search_root).resolve()
    return all(Path(search_root).resolve().parent == resolved_common_search_root for search_root in search_roots)


__all__ = [
    "SnapshotRegistrySource",
    "canonical_snapshot_registry_run_dir",
    "common_search_root",
    "discover_snapshot_registry_run_dirs",
    "is_recursive_registry_search_root",
    "resolve_snapshot_registry_run_dir",
    "run_dir_contains_registry_snapshots",
    "run_dir_matches_registry_snapshots",
    "sha256_file",
    "should_include_common_search_root",
    "snapshot_by_policy_id_or_imported_seed_suffix",
    "snapshot_registry_resolution_snapshots",
    "spaced_snapshot_resolution_samples",
    "unique_paths",
]
