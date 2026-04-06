"""Snapshot registry and metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import json

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


@dataclass(slots=True)
class SnapshotRegistry:
    """Durable snapshot registry with stable ordering."""

    # Retention policy defaults from master plan §12.2.
    recent_size: int = 24
    champion_size: int = 4

    # In v1 we store a single list; caller can interpret subsets (recent/champion) as needed.
    snapshots: list[SnapshotMeta] = field(default_factory=list)

    # ------------------------
    # Query helpers
    # ------------------------
    def latest(self, n: int = 1) -> list[SnapshotMeta]:
        n = int(n)
        if n <= 0:
            return []
        ordered = self._sorted()
        return ordered[-n:]

    def latest_n(self, n: int = 1) -> list[SnapshotMeta]:
        # Alias for callers that prefer "latest_n".
        return self.latest(n)

    def latest_ids(self, n: int = 1) -> list[str]:
        return [s.policy_id for s in self.latest(n)]

    def _sorted(self) -> list[SnapshotMeta]:
        return sorted(self.snapshots, key=lambda s: s.sort_key())

    # ------------------------
    # Mutation helpers
    # ------------------------
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
        if not str(policy_id).strip():
            raise ValueError("policy_id must be non-empty")

        normalized_path = _normalize_snapshot_artifact_path(path)

        meta = SnapshotMeta(
            policy_id=str(policy_id),
            update=update_i,
            weights_sha256=str(weights_sha256),
            path=normalized_path,
            created_utc=created_utc or _now_utc_iso(),
        )

        # Replace existing entry with same policy_id (policy_id is the stable handle).
        for i, existing in enumerate(self.snapshots):
            if existing.policy_id == meta.policy_id:
                self.snapshots[i] = meta
                self.snapshots = self._sorted()
                return

        self.snapshots.append(meta)
        self.snapshots = self._sorted()

    # ------------------------
    # Persistence
    # ------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _REGISTRY_SCHEMA_VERSION,
            "recent_size": int(self.recent_size),
            "champion_size": int(self.champion_size),
            "snapshots": [asdict(s) for s in self._sorted()],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_stable_json_dumps(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SnapshotRegistry":
        if not path.exists():
            return cls()

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("registry.json must be a JSON object")

        # Backward compat: old registry was {"snapshots": ["id1","id2"]} without schema_version.
        if isinstance(raw.get("snapshots"), list) and raw.get("schema_version") is None:
            snaps = raw.get("snapshots", [])
            if snaps and all(isinstance(x, str) for x in snaps):
                reg = cls()
                reg.snapshots = [
                    SnapshotMeta(
                        policy_id=str(sid),
                        update=int(i),
                        weights_sha256="",
                        path="unknown",
                        created_utc=_now_utc_iso(),
                    )
                    for i, sid in enumerate(snaps)
                ]
                reg.snapshots = reg._sorted()
                return reg

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
                    path=path_value,
                    created_utc=str(item.get("created_utc", _now_utc_iso())),
                )
            )

        reg = cls(recent_size=recent_size, champion_size=champion_size, snapshots=snapshots)
        reg.snapshots = reg._sorted()
        return reg