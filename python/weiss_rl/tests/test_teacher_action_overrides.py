from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.teacher_action_overrides import (
    TeacherActionOverrideExportConfig,
    build_teacher_action_overrides_from_inspections,
    write_teacher_action_overrides_jsonl,
)
from weiss_rl.replay.trajectory_bc import load_teacher_action_overrides_jsonl


def test_teacher_action_overrides_from_inspections_filters_mismatches(tmp_path: Path) -> None:
    bundle_path = tmp_path / "replay_abc_pair000_swap0.zip"
    inspection_json = tmp_path / "inspection.json"
    inspection_json.write_text(
        json.dumps(
            {
                "bundle_path": bundle_path.as_posix(),
                "top_differences": [
                    {
                        "actor": 0,
                        "policy_a_matches_policy_b_top_action": False,
                        "policy_a_top_action": {"action": 1, "family": "clock_from_hand"},
                        "policy_b_top_action": {"action": 2, "family": "clock_from_hand"},
                        "step_index": 3,
                        "total_variation": 0.75,
                    },
                    {
                        "actor": 0,
                        "policy_a_matches_policy_b_top_action": True,
                        "policy_b_top_action": {"action": 4, "family": "pass"},
                        "step_index": 4,
                        "total_variation": 0.10,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rows, summary = build_teacher_action_overrides_from_inspections(
        TeacherActionOverrideExportConfig(
            inspection_jsons=(inspection_json,),
            min_total_variation=0.5,
        )
    )

    assert summary["row_count"] == 1
    assert summary["bundle_count"] == 1
    assert rows[0]["bundle_name"] == bundle_path.name
    assert rows[0]["step_index"] == 3
    assert rows[0]["teacher_action"] == 2
    assert rows[0]["policy_b_top_action"]["family"] == "clock_from_hand"

    output_jsonl = tmp_path / "overrides.jsonl"
    write_teacher_action_overrides_jsonl(output_jsonl, rows)
    overrides = load_teacher_action_overrides_jsonl(output_jsonl)
    assert overrides[(bundle_path.name, 3)] == 2
