"""Console reporting helpers for B1 candidate selection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def select_b1_candidate_output_lines(summary: Mapping[str, Any], *, output_json: Path | None) -> list[str]:
    if output_json is not None:
        return [str(output_json)]

    lines = [json.dumps({key: summary[key] for key in ("candidate_count", "warnings")}, indent=2, sort_keys=True)]
    selected = summary.get("selected")
    if isinstance(selected, Mapping):
        lines.append(
            "selected "
            f"run={selected['run_name']} snapshot={selected['snapshot_policy_id']} "
            f"update={selected['update_count']} score={selected['selection_score']:.6f} "
            f"required_min={selected['required_anchor_min']:.6f} eligible={selected['eligible']}"
        )
        command = selected.get("confirmation_command")
        if isinstance(command, list):
            lines.append("confirm_command " + command_text([str(part) for part in command]))
    published = summary.get("published_baseline_alias")
    if isinstance(published, Mapping):
        lines.append(
            "published_baseline_alias "
            f"policy_id={published['policy_id']} source={published['alias_for_policy_id']} "
            f"update={published['update']}"
        )
    published_selected = summary.get("published_selected_alias")
    if isinstance(published_selected, Mapping):
        lines.append(
            "published_selected_alias "
            f"policy_id={published_selected['policy_id']} source={published_selected['alias_for_policy_id']} "
            f"update={published_selected['update']}"
        )
    return lines
