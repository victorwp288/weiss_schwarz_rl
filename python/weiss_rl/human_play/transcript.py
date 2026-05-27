"""Artifact writing for human play/study sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    decision_index: int
    actor_seat: int
    actor_kind: str
    action_id: int
    action_label: str
    legal_action_ids: tuple[int, ...]
    decision_id: int | None
    decision_kind: str | None
    view_hash64: str | None
    legal_fingerprint64: str | None
    elapsed_ms: float | None = None
    model_ranked_actions: tuple[dict[str, Any], ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "decision_index": int(self.decision_index),
            "actor_seat": int(self.actor_seat),
            "actor_kind": self.actor_kind,
            "action_id": int(self.action_id),
            "action_label": self.action_label,
            "legal_action_ids": [int(action_id) for action_id in self.legal_action_ids],
            "decision_id": self.decision_id,
            "decision_kind": self.decision_kind,
            "view_hash64": self.view_hash64,
            "legal_fingerprint64": self.legal_fingerprint64,
            "elapsed_ms": self.elapsed_ms,
            "model_ranked_actions": [dict(item) for item in self.model_ranked_actions],
        }


class HumanPlayTranscript:
    """Append-only on-disk transcript for one human-play session."""

    def __init__(self, session_dir: Path, *, manifest: dict[str, Any]) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.session_dir / "decisions.jsonl"
        self.events_path = self.session_dir / "events.jsonl"
        self.manifest_path = self.session_dir / "manifest.json"
        self.postgame_path = self.session_dir / "postgame_report.md"
        payload = dict(manifest)
        payload.setdefault("schema_version", "human_play_manifest_v1")
        payload.setdefault("created_at_unix", time.time())
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def append_decision(self, record: DecisionRecord) -> None:
        with self.decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")

    def append_event(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("time_unix", time.time())
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_postgame_report(self, summary: dict[str, Any]) -> None:
        lines = [
            "# Human Play Session",
            "",
            f"- status: `{summary.get('status', 'unknown')}`",
            f"- terminal: `{summary.get('terminal', False)}`",
            f"- decisions: `{summary.get('decision_count', 0)}`",
            f"- winner_seat: `{summary.get('winner_seat', None)}`",
            f"- termination_reason: `{summary.get('termination_reason', None)}`",
            "",
            "Artifacts:",
            "",
            f"- manifest: `{self.manifest_path.name}`",
            f"- decisions: `{self.decisions_path.name}`",
            f"- events: `{self.events_path.name}`",
        ]
        self.postgame_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
