"""Local run and policy catalog helpers for the human-play setup screen."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from weiss_rl.config import load_stack_config
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)

_AUTO_POLICY_ID = "main_league_selected"


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_dir: str
    name: str
    label: str
    modified_unix: float
    policy_count: int
    default_policy_id: str
    has_config: bool
    has_registry: bool
    config_loadable: bool
    load_error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "name": self.name,
            "label": self.label,
            "modified_unix": self.modified_unix,
            "policy_count": int(self.policy_count),
            "default_policy_id": self.default_policy_id,
            "has_config": bool(self.has_config),
            "has_registry": bool(self.has_registry),
            "config_loadable": bool(self.config_loadable),
            "load_error": self.load_error,
        }


@dataclass(frozen=True, slots=True)
class PolicySummary:
    policy_id: str
    label: str
    kind: str
    update: int | None = None
    path: str | None = None
    selected_by_default: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "label": self.label,
            "kind": self.kind,
            "update": self.update,
            "path": self.path,
            "selected_by_default": bool(self.selected_by_default),
        }


def default_repo_root() -> Path:
    override = os.environ.get("WEISS_HUMAN_PLAY_REPO_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def list_candidate_runs(*, repo_root: Path | None = None, limit: int = 80) -> list[RunSummary]:
    root = default_repo_root() if repo_root is None else Path(repo_root).resolve()
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return []
    candidates: list[RunSummary] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        has_config = (run_dir / "config_canonical.json").is_file()
        registry_path = _registry_path(run_dir)
        if not has_config and not registry_path.is_file():
            continue
        load_error = _config_load_error(run_dir) if has_config else "missing config_canonical.json"
        policies = list_policies_for_run(run_dir)
        candidates.append(
            RunSummary(
                run_dir=str(run_dir.resolve()),
                name=run_dir.name,
                label=_label_from_run_name(run_dir.name),
                modified_unix=run_dir.stat().st_mtime,
                policy_count=sum(1 for policy in policies if policy.kind == "snapshot"),
                default_policy_id=_AUTO_POLICY_ID,
                has_config=has_config,
                has_registry=registry_path.is_file(),
                config_loadable=load_error is None,
                load_error=load_error,
            )
        )
    return sorted(
        candidates,
        key=lambda item: (item.config_loadable, item.has_registry, item.modified_unix),
        reverse=True,
    )[: max(1, int(limit))]


def list_policies_for_run(run_dir: Path) -> list[PolicySummary]:
    registry = _load_registry(run_dir)
    snapshot_rows = registry.get("snapshots", [])
    snapshots = snapshot_rows if isinstance(snapshot_rows, list) else []
    policies = [
        PolicySummary(
            policy_id=_AUTO_POLICY_ID,
            label="Auto-select strongest main model",
            kind="alias",
            selected_by_default=True,
        ),
        PolicySummary(policy_id=RANDOM_LEGAL_POLICY_ID, label="B0 RandomLegal", kind="baseline"),
        PolicySummary(policy_id=NO_LEAGUE_POLICY_ID, label="B1 NoLeague baseline", kind="baseline"),
        PolicySummary(policy_id=HEURISTIC_PUBLIC_POLICY_ID, label="B2 HeuristicPublic", kind="heuristic"),
        PolicySummary(policy_id=HEURISTIC_PUBLIC_AGGRO_POLICY_ID, label="B3 HeuristicPublicAggro", kind="heuristic"),
        PolicySummary(
            policy_id=HEURISTIC_PUBLIC_CONTROL_POLICY_ID, label="B4 HeuristicPublicControl", kind="heuristic"
        ),
    ]
    seen = {policy.policy_id for policy in policies}
    for row in sorted(
        (row for row in snapshots if isinstance(row, dict)),
        key=lambda item: (int(item.get("update", 0)), str(item.get("policy_id", ""))),
        reverse=True,
    ):
        policy_id = str(row.get("policy_id", "")).strip()
        if not policy_id or policy_id in seen:
            continue
        seen.add(policy_id)
        policies.append(
            PolicySummary(
                policy_id=policy_id,
                label=_label_from_policy_id(policy_id, update=_optional_int(row.get("update"))),
                kind="snapshot",
                update=_optional_int(row.get("update")),
                path=_optional_str(row.get("path")),
            )
        )
    return policies


def _registry_path(run_dir: Path) -> Path:
    return Path(run_dir) / "training" / "snapshots" / "registry.json"


def _config_load_error(run_dir: Path) -> str | None:
    try:
        load_stack_config(Path(run_dir) / "config_canonical.json")
    except Exception as exc:  # pragma: no cover - exact legacy config errors vary by local run archive.
        return str(exc)
    return None


def _load_registry(run_dir: Path) -> dict[str, Any]:
    path = _registry_path(run_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _label_from_run_name(name: str) -> str:
    text = str(name).strip().replace("_", " ")
    if not text:
        return "Unnamed run"
    return " ".join(word.upper() if word.lower() in {"b0", "b1", "b2", "b3", "b4"} else word for word in text.split())


def _label_from_policy_id(policy_id: str, *, update: int | None) -> str:
    if policy_id.startswith("policy_"):
        suffix = policy_id.removeprefix("policy_").lstrip("0") or "0"
        return f"Policy {suffix}" if update is None else f"Policy {suffix} (update {update})"
    if policy_id.startswith("train_u"):
        return policy_id.replace("_", " ")
    return policy_id


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None
