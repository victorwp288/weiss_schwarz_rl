from __future__ import annotations

import json
from types import SimpleNamespace

from weiss_rl.training.dev_eval import (
    persist_periodic_dev_eval_summary,
    update_stall_monitor,
)


def test_persist_periodic_dev_eval_summary_uses_policy_id_as_key(tmp_path) -> None:
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")

    persist_periodic_dev_eval_summary(
        training_paths=paths,
        payload={
            "policy_id": "policy_000010",
            "aggregate_score": 0.625,
            "anchor_scores": {"B0 RandomLegal": 0.75},
            "update_count": 10,
            "policy_version": 3,
        },
    )
    persist_periodic_dev_eval_summary(
        training_paths=paths,
        payload={"policy_id": "", "aggregate_score": 1.0},
    )

    payload = json.loads((paths.logs_dir / "periodic_dev_eval_summaries.json").read_text(encoding="utf-8"))
    assert payload == {
        "policy_000010": {
            "aggregate_score": 0.625,
            "anchor_scores": {"B0 RandomLegal": 0.75},
            "update_count": 10,
            "policy_version": 3,
        }
    }


def test_update_stall_monitor_tracks_consecutive_stall_risk(tmp_path) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(
                    enabled=True,
                    truncation_rate_threshold=0.25,
                    consecutive_evals=2,
                )
            )
        )
    )
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    summary = {
        "anchors": {
            "B0 RandomLegal": {
                "summary": {
                    "games": 4,
                    "truncations": 1,
                    "no_progress_timeouts": 0,
                    "natural_timeouts": 0,
                }
            },
            "B2 HeuristicPublic": {
                "summary": {
                    "games": 4,
                    "truncations": 0,
                    "no_progress_timeouts": 2,
                    "natural_timeouts": 1,
                }
            },
        }
    }

    first = update_stall_monitor(stack=stack, training_paths=paths, update_count=20, summary_payload=summary)
    second = update_stall_monitor(stack=stack, training_paths=paths, update_count=40, summary_payload=summary)

    assert first is not None
    assert first["stall_risk"] is False
    assert first["consecutive_trigger_count"] == 1
    assert first["worst_anchor"] == "B2 HeuristicPublic"
    assert first["stall_indicator_kind"] == "no_progress_timeout"
    assert second is not None
    assert second["stall_risk"] is True
    assert second["consecutive_trigger_count"] == 2
    persisted = json.loads((paths.logs_dir / "stall_monitor.json").read_text(encoding="utf-8"))
    assert persisted == second
