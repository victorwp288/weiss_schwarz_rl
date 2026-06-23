from __future__ import annotations

from .entrypoints_eval_selection_test_support import (
    EXPECTED_DETERMINISTIC_POLICY_IDS,
    prepare_policy_selection_run,
)
from .entrypoints_test_support import Path


def test_eval_entrypoint_prefers_run_local_policy_selection_over_manifest_fallback(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    stack, layout = prepare_policy_selection_run(tmp_path, "eval_policy_selection")
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_stale_only"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == EXPECTED_DETERMINISTIC_POLICY_IDS
    assert details["mode"] == "deterministic_v1"
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"
