from __future__ import annotations

from .entrypoints_eval_selection_test_support import (
    EXPECTED_DETERMINISTIC_POLICY_IDS,
    prepare_policy_selection_run,
)
from .entrypoints_test_support import ArtifactLayout, Path, json


def test_eval_entrypoint_honors_completed_manifest_policy_selection(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    stack, layout = prepare_policy_selection_run(tmp_path, "eval_policy_selection_locked")
    layout.run_summary_path.write_text(
        json.dumps({"kind": "run_summary_v1", "canonical_eval_completed": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_locked"],
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

    assert policy_ids == ["B0 RandomLegal", "policy_locked"]
    assert details["mode"] == "deterministic_v1"
    assert details["status"] == "resolved"
    assert details["policy_count"] == 2
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_entrypoint_ignores_incomplete_manifest_selection_from_canonical_eval_pipeline(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    stack, layout = prepare_policy_selection_run(tmp_path, "eval_policy_selection_incomplete")
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_stale_only"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
            "resolved_by": "canonical_eval_pipeline_v1",
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


def test_eval_entrypoint_ignores_completed_explicit_cli_manifest_selection(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    stack, layout = prepare_policy_selection_run(tmp_path, "eval_policy_selection_explicit_cli")
    layout.final_eval_summary_json().parent.mkdir(parents=True, exist_ok=True)
    layout.final_eval_summary_json().write_text("{}\n", encoding="utf-8")
    manifest = {
        "policy_set_selection": ["policy_custom_only"],
        "policy_set_selection_details": {
            "mode": "explicit_cli",
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


def test_eval_manifest_persistence_records_explicit_cli_policy_selection(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    run_dir = tmp_path / "runs" / "eval_manifest_persistence"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_set_selection": ["policy_original"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }
    layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    eval_script._persist_policy_selection_in_manifest(
        layout=layout,
        manifest=dict(manifest),
        policy_ids=["policy_explicit"],
        selection_details={"mode": "explicit_cli", "policy_count": 1},
    )

    persisted = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert persisted["policy_set_selection"] == ["policy_explicit"]
    assert persisted["policy_set_selection_details"] == {
        "mode": "explicit_cli",
        "policy_count": 1,
        "resolved_by": "canonical_eval_pipeline_v1",
        "status": "resolved",
    }
