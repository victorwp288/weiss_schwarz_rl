"""Snapshot registry and metadata."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

_REGISTRY_SCHEMA_VERSION = 1
REGISTRY_FILENAME = "registry.json"
SNAPSHOT_WEIGHTS_FILENAME = "weights.pt"
SNAPSHOT_METADATA_FILENAME = "policy_meta.json"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_dumps(obj: Any) -> str:
    # Stable output for diffs and reproducibility.
    return json.dumps(obj, indent=2, sort_keys=True, separators=(",", ": "))


def snapshot_weights_relpath(policy_id: str) -> str:
    normalized_policy_id = str(policy_id).strip()
    if not normalized_policy_id:
        raise ValueError("policy_id must be non-empty")
    return PurePosixPath("training", "snapshots", normalized_policy_id, SNAPSHOT_WEIGHTS_FILENAME).as_posix()


def snapshot_metadata_relpath(policy_id: str) -> str:
    normalized_policy_id = str(policy_id).strip()
    if not normalized_policy_id:
        raise ValueError("policy_id must be non-empty")
    return PurePosixPath("training", "snapshots", normalized_policy_id, SNAPSHOT_METADATA_FILENAME).as_posix()


def _normalize_snapshot_artifact_path(path: str) -> str:
    normalized_path = str(path).strip()
    if not normalized_path:
        raise ValueError("path must be non-empty")

    pure_path = PurePosixPath(normalized_path)
    parts = pure_path.parts
    if pure_path.is_absolute() or len(parts) != 4 or parts[:2] != ("training", "snapshots"):
        raise ValueError(
            "path must be a run-relative snapshot weights artifact under training/snapshots/<policy_id>/weights.pt"
        )
    if parts[-1] != SNAPSHOT_WEIGHTS_FILENAME:
        raise ValueError(
            "path must be a run-relative snapshot weights artifact under training/snapshots/<policy_id>/weights.pt"
        )
    return pure_path.as_posix()


def _normalize_window_size(value: int, *, field_name: str) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return normalized


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    policy_id: str
    update: int
    weights_sha256: str
    path: str  # run-relative posix path, e.g. "training/snapshots/policy_000123/weights.pt"
    created_utc: str = field(default_factory=_now_utc_iso)

    def sort_key(self) -> tuple[int, str]:
        # Stable ordering: primary by update, then policy_id.
        return (int(self.update), str(self.policy_id))


@dataclass(frozen=True, slots=True)
class SnapshotReferenceNormalization:
    refs: list[str]
    dropped_refs: list[str]
    trimmed_refs: list[str]


@dataclass(frozen=True, slots=True)
class ChampionDemotion:
    removed_refs: list[str]
    remaining_refs: list[str]


def _move_ref_to_end(snapshot_ids: list[str], snapshot_id: str) -> list[str]:
    return [existing for existing in snapshot_ids if existing != snapshot_id] + [snapshot_id]


def normalize_snapshot_references(
    snapshot_ids: Iterable[str],
    *,
    existing_snapshot_ids: set[str],
    limit: int | None = None,
) -> SnapshotReferenceNormalization:
    normalized: list[str] = []
    dropped_refs: list[str] = []
    for raw_snapshot_id in snapshot_ids:
        snapshot_id = str(raw_snapshot_id).strip()
        if not snapshot_id or snapshot_id not in existing_snapshot_ids:
            if snapshot_id:
                dropped_refs.append(snapshot_id)
            continue
        normalized = _move_ref_to_end(normalized, snapshot_id)

    if limit is None:
        return SnapshotReferenceNormalization(
            refs=normalized,
            dropped_refs=dropped_refs,
            trimmed_refs=[],
        )
    if limit <= 0:
        return SnapshotReferenceNormalization(
            refs=[],
            dropped_refs=dropped_refs,
            trimmed_refs=normalized,
        )

    refs = normalized[-limit:]
    return SnapshotReferenceNormalization(
        refs=refs,
        dropped_refs=dropped_refs,
        trimmed_refs=normalized[: max(0, len(normalized) - len(refs))],
    )


def champion_demotion_newer_than(
    champion_refs: Iterable[str],
    *,
    updates_by_policy: Mapping[str, int],
    update: int,
) -> ChampionDemotion:
    update_i = int(update)
    removed_refs = [
        snapshot_id for snapshot_id in champion_refs if int(updates_by_policy.get(snapshot_id, -1)) > update_i
    ]
    if not removed_refs:
        return ChampionDemotion(removed_refs=[], remaining_refs=list(champion_refs))
    removed_set = set(removed_refs)
    return ChampionDemotion(
        removed_refs=removed_refs,
        remaining_refs=[snapshot_id for snapshot_id in champion_refs if snapshot_id not in removed_set],
    )


def champion_demotion_stale_by_age(
    champion_refs: Iterable[str],
    *,
    updates_by_policy: Mapping[str, int],
    current_update: int,
    max_age_updates: int,
) -> ChampionDemotion:
    current_update_i = int(current_update)
    max_age_updates_i = int(max_age_updates)
    removed_refs = [
        snapshot_id
        for snapshot_id in champion_refs
        if (current_update_i - int(updates_by_policy.get(snapshot_id, current_update_i))) > max_age_updates_i
    ]
    if not removed_refs:
        return ChampionDemotion(removed_refs=[], remaining_refs=list(champion_refs))
    removed_set = set(removed_refs)
    return ChampionDemotion(
        removed_refs=removed_refs,
        remaining_refs=[snapshot_id for snapshot_id in champion_refs if snapshot_id not in removed_set],
    )


@dataclass(slots=True)
class SnapshotRegistry:
    """Durable snapshot registry with stable ordering and champion tracking."""

    recent_size: int = 24
    champion_size: int = 4
    snapshots: list[SnapshotMeta] = field(default_factory=list)
    champion_snapshots: list[str] = field(default_factory=list)
    pinned_snapshots: list[str] = field(default_factory=list)

    def latest(self, n: int = 1) -> list[SnapshotMeta]:
        n = int(n)
        if n <= 0:
            return []
        self.normalize()
        return self.snapshots[-n:]

    def latest_n(self, n: int = 1) -> list[SnapshotMeta]:
        return self.latest(n)

    def latest_ids(self, n: int = 1) -> list[str]:
        return [snapshot.policy_id for snapshot in self.latest(n)]

    def latest_champions(
        self,
        n: int = 1,
        *,
        current_update: int | None = None,
        max_age_updates: int | None = None,
    ) -> list[str]:
        n = int(n)
        if n <= 0:
            return []
        self.normalize()
        champion_ids = list(self.champion_snapshots)
        if current_update is not None and max_age_updates is not None and int(max_age_updates) > 0:
            updates_by_policy = self._updates_by_policy()
            current_update_i = int(current_update)
            max_age_updates_i = int(max_age_updates)
            champion_ids = [
                snapshot_id
                for snapshot_id in champion_ids
                if (current_update_i - int(updates_by_policy.get(snapshot_id, current_update_i))) <= max_age_updates_i
            ]
        return champion_ids[-n:]

    def has_snapshot(self, snapshot_id: str) -> bool:
        normalized_snapshot_id = str(snapshot_id).strip()
        if not normalized_snapshot_id:
            return False
        return any(snapshot.policy_id == normalized_snapshot_id for snapshot in self.snapshots)

    def add_champion(self, snapshot_id: str) -> None:
        normalized_snapshot_id = self._require_existing_snapshot_id(snapshot_id)
        self.champion_snapshots = _move_ref_to_end(self.champion_snapshots, normalized_snapshot_id)
        self.normalize()

    def pin_snapshot(self, snapshot_id: str) -> None:
        normalized_snapshot_id = self._require_existing_snapshot_id(snapshot_id)
        self.pinned_snapshots = _move_ref_to_end(self.pinned_snapshots, normalized_snapshot_id)
        self.normalize()

    def add(self, snapshot_id: str, *, is_champion: bool = False) -> None:
        self.add_champion(snapshot_id) if is_champion else None

    def remove_champion(self, snapshot_id: str) -> bool:
        normalized_snapshot_id = str(snapshot_id).strip()
        if not normalized_snapshot_id:
            return False
        original = list(self.champion_snapshots)
        self.champion_snapshots = [
            snapshot for snapshot in self.champion_snapshots if snapshot != normalized_snapshot_id
        ]
        self.normalize()
        return self.champion_snapshots != original

    def demote_champions_newer_than(self, update: int) -> list[str]:
        self.normalize()
        demotion = champion_demotion_newer_than(
            self.champion_snapshots,
            updates_by_policy=self._updates_by_policy(),
            update=int(update),
        )
        if not demotion.removed_refs:
            return []
        self.champion_snapshots = demotion.remaining_refs
        self.normalize()
        return demotion.removed_refs

    def demote_stale_champions(self, *, current_update: int, max_age_updates: int) -> list[str]:
        max_age_updates_i = int(max_age_updates)
        if max_age_updates_i <= 0:
            return []
        self.normalize()
        demotion = champion_demotion_stale_by_age(
            self.champion_snapshots,
            updates_by_policy=self._updates_by_policy(),
            current_update=int(current_update),
            max_age_updates=max_age_updates_i,
        )
        if not demotion.removed_refs:
            return []
        self.champion_snapshots = demotion.remaining_refs
        self.normalize()
        return demotion.removed_refs

    def add_snapshot(
        self,
        *,
        policy_id: str,
        update: int,
        weights_sha256: str,
        path: str,
        created_utc: str | None = None,
    ) -> None:
        update_i = int(update)
        if update_i < 0:
            raise ValueError("update must be >= 0")
        normalized_policy_id = str(policy_id).strip()
        if not normalized_policy_id:
            raise ValueError("policy_id must be non-empty")

        meta = SnapshotMeta(
            policy_id=normalized_policy_id,
            update=update_i,
            weights_sha256=str(weights_sha256),
            path=_normalize_snapshot_artifact_path(path),
            created_utc=created_utc or _now_utc_iso(),
        )

        for index, existing in enumerate(self.snapshots):
            if existing.policy_id == meta.policy_id:
                self.snapshots[index] = meta
                self.normalize()
                return

        self.snapshots.append(meta)
        self.normalize()

    def retained_snapshot_ids(self) -> set[str]:
        self.normalize()
        retained_ids = set(self.latest_ids(self.recent_size))
        retained_ids.update(self.latest_champions(self.champion_size))
        retained_ids.update(self.pinned_snapshots)
        return retained_ids

    def prune(self) -> list[SnapshotMeta]:
        self.normalize()
        retained_ids = self.retained_snapshot_ids()
        retained_snapshots: list[SnapshotMeta] = []
        pruned_snapshots: list[SnapshotMeta] = []
        for snapshot in self.snapshots:
            if snapshot.policy_id in retained_ids:
                retained_snapshots.append(snapshot)
            else:
                pruned_snapshots.append(snapshot)

        self.snapshots = retained_snapshots
        retained_policy_ids = {snapshot.policy_id for snapshot in self.snapshots}
        self.champion_snapshots = [
            snapshot_id for snapshot_id in self.champion_snapshots if snapshot_id in retained_policy_ids
        ]
        self.pinned_snapshots = [
            snapshot_id for snapshot_id in self.pinned_snapshots if snapshot_id in retained_policy_ids
        ]
        return pruned_snapshots

    def normalize(self) -> None:
        self.recent_size = _normalize_window_size(self.recent_size, field_name="recent_size")
        self.champion_size = _normalize_window_size(self.champion_size, field_name="champion_size")
        self.snapshots = self._normalized_snapshots()
        existing_snapshot_ids = {snapshot.policy_id for snapshot in self.snapshots}
        self.champion_snapshots = self._normalized_refs(
            self.champion_snapshots,
            existing_snapshot_ids=existing_snapshot_ids,
            limit=self.champion_size,
        )
        self.pinned_snapshots = self._normalized_refs(
            self.pinned_snapshots,
            existing_snapshot_ids=existing_snapshot_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        self.normalize()
        return {
            "schema_version": _REGISTRY_SCHEMA_VERSION,
            "recent_size": int(self.recent_size),
            "champion_size": int(self.champion_size),
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
            "champion_snapshots": list(self.champion_snapshots),
            "pinned_snapshots": list(self.pinned_snapshots),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_stable_json_dumps(self.to_dict()) + "\n", encoding="utf-8")

    def _normalized_snapshots(self) -> list[SnapshotMeta]:
        deduped: dict[str, SnapshotMeta] = {}
        for snapshot in self.snapshots:
            policy_id = str(snapshot.policy_id).strip()
            if not policy_id:
                raise ValueError("registry snapshot missing policy_id")
            update = int(snapshot.update)
            if update < 0:
                raise ValueError(f"registry snapshot {policy_id} has update < 0")
            deduped[policy_id] = SnapshotMeta(
                policy_id=policy_id,
                update=update,
                weights_sha256=str(snapshot.weights_sha256),
                path=_normalize_snapshot_artifact_path(snapshot.path),
                created_utc=str(snapshot.created_utc or _now_utc_iso()),
            )
        return sorted(deduped.values(), key=lambda snapshot: snapshot.sort_key())

    def _normalized_refs(
        self,
        snapshot_ids: Iterable[str],
        *,
        existing_snapshot_ids: set[str],
        limit: int | None = None,
    ) -> list[str]:
        return normalize_snapshot_references(
            snapshot_ids,
            existing_snapshot_ids=existing_snapshot_ids,
            limit=limit,
        ).refs

    def _require_existing_snapshot_id(self, snapshot_id: str) -> str:
        normalized_snapshot_id = str(snapshot_id).strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id must be non-empty")
        if not self.has_snapshot(normalized_snapshot_id):
            raise ValueError(f"snapshot_id must reference an existing snapshot: {normalized_snapshot_id!r}")
        return normalized_snapshot_id

    def _updates_by_policy(self) -> dict[str, int]:
        return {snapshot.policy_id: int(snapshot.update) for snapshot in self.snapshots}

    @classmethod
    def load(cls, path: Path) -> SnapshotRegistry:
        if not path.exists():
            return cls()

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("registry.json must be a JSON object")

        if isinstance(raw.get("snapshots"), list) and raw.get("schema_version") is None:
            snapshot_ids = raw.get("snapshots", [])
            champion_snapshot_ids = raw.get("champion_snapshots", [])
            if snapshot_ids and all(isinstance(item, str) for item in snapshot_ids):
                registry = cls(
                    snapshots=[
                        SnapshotMeta(
                            policy_id=str(snapshot_id).strip(),
                            update=int(index),
                            weights_sha256="",
                            path=snapshot_weights_relpath(str(snapshot_id).strip()),
                            created_utc=_now_utc_iso(),
                        )
                        for index, snapshot_id in enumerate(snapshot_ids)
                        if str(snapshot_id).strip()
                    ],
                    champion_snapshots=[str(snapshot_id).strip() for snapshot_id in champion_snapshot_ids],
                )
                registry.normalize()
                return registry

        schema_version = int(raw.get("schema_version", 0))
        if schema_version != _REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported registry schema_version={schema_version}")

        recent_size = int(raw.get("recent_size", 24))
        champion_size = int(raw.get("champion_size", 4))

        snapshots_raw = raw.get("snapshots", [])
        if not isinstance(snapshots_raw, list):
            raise ValueError("registry.snapshots must be a list")

        snapshots: list[SnapshotMeta] = []
        for item in snapshots_raw:
            if not isinstance(item, dict):
                raise ValueError("registry.snapshots entries must be objects")

            policy_id = str(item.get("policy_id", "")).strip()
            if not policy_id:
                raise ValueError("registry snapshot missing policy_id")

            update = int(item.get("update", 0))
            if update < 0:
                raise ValueError(f"registry snapshot {policy_id} has update < 0")

            path_value = str(item.get("path", "")).strip()
            if not path_value:
                raise ValueError(f"registry snapshot {policy_id} missing non-empty path")

            snapshots.append(
                SnapshotMeta(
                    policy_id=policy_id,
                    update=update,
                    weights_sha256=str(item.get("weights_sha256", "")),
                    path=_normalize_snapshot_artifact_path(path_value),
                    created_utc=str(item.get("created_utc", _now_utc_iso())),
                )
            )

        champion_snapshots_raw = raw.get("champion_snapshots", [])
        if not isinstance(champion_snapshots_raw, list):
            raise ValueError("registry.champion_snapshots must be a list")
        pinned_snapshots_raw = raw.get("pinned_snapshots", [])
        if not isinstance(pinned_snapshots_raw, list):
            raise ValueError("registry.pinned_snapshots must be a list")

        registry = cls(
            recent_size=recent_size,
            champion_size=champion_size,
            snapshots=snapshots,
            champion_snapshots=[str(item).strip() for item in champion_snapshots_raw],
            pinned_snapshots=[str(item).strip() for item in pinned_snapshots_raw],
        )
        registry.normalize()
        return registry
