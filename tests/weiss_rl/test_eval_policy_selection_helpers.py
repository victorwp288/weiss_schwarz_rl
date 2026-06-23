from __future__ import annotations

from .entrypoints_test_support import (
    ArtifactLayout,
    Path,
    json,
    pytest,
)


def test_eval_policy_selection_results_build_explicit_cli_details() -> None:
    from weiss_rl.workflows.eval_support.eval_policy_selection_results import _explicit_policy_selection

    assert _explicit_policy_selection([" B0 RandomLegal ", "", "policy_000100"]) == (
        ["B0 RandomLegal", "policy_000100"],
        {"mode": "explicit_cli", "policy_count": 2},
    )
    assert _explicit_policy_selection(["", "   "]) is None


def test_eval_policy_selection_results_build_manifest_fallback_details() -> None:
    from weiss_rl.workflows.eval_support.eval_policy_selection_results import _manifest_policy_selection_fallback

    assert _manifest_policy_selection_fallback({"policy_set_selection": [" B0 RandomLegal ", "", 123]}) == (
        ["B0 RandomLegal", "123"],
        {"mode": "manifest_policy_set_selection_fallback", "policy_count": 2},
    )
    assert _manifest_policy_selection_fallback({"policy_set_selection": "not-a-list"}) is None
    assert _manifest_policy_selection_fallback({}) is None


def test_eval_policy_final_set_resolution_uses_available_source_paths(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _resolve_available_policy_source_paths

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    default_registry = layout.training_snapshots_dir / "registry.json"
    default_registry.write_text("{}\n", encoding="utf-8")
    periodic_dev_eval = layout.training_logs_dir / "periodic_dev_eval_summaries.json"
    periodic_dev_eval.write_text("{}\n", encoding="utf-8")
    manifest_registry = tmp_path / "manifest" / "registry.json"
    manifest_registry.parent.mkdir(parents=True, exist_ok=True)
    manifest_registry.write_text("{}\n", encoding="utf-8")
    explicit_dev_eval = tmp_path / "explicit" / "dev_eval.json"
    explicit_dev_eval.parent.mkdir(parents=True, exist_ok=True)
    explicit_dev_eval.write_text("{}\n", encoding="utf-8")

    resolved_registry, resolved_dev_eval = _resolve_available_policy_source_paths(
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=explicit_dev_eval,
        manifest_snapshot_registry=manifest_registry,
        manifest_dev_eval=None,
    )

    assert resolved_registry == manifest_registry
    assert resolved_dev_eval == explicit_dev_eval

    fallback_registry, fallback_dev_eval = _resolve_available_policy_source_paths(
        layout=layout,
        snapshot_registry_path=tmp_path / "missing" / "registry.json",
        dev_eval_summaries_path=tmp_path / "missing" / "dev_eval.json",
        manifest_snapshot_registry=None,
        manifest_dev_eval=None,
    )

    assert fallback_registry is None
    assert fallback_dev_eval == periodic_dev_eval


def test_eval_policy_final_set_resolution_builds_deterministic_selection_details(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _resolve_deterministic_final_policy_set

    observed: dict[str, object] = {}
    registry_path = tmp_path / "registry.json"
    dev_eval_path = tmp_path / "dev_eval.json"

    def fake_resolve_final_policy_set(**kwargs: object) -> list[str]:
        observed["resolve"] = kwargs
        return ["B0 RandomLegal", "policy_000100"]

    resolved = _resolve_deterministic_final_policy_set(
        evaluation=SimpleNamespace(final_policy_set_selection={"folding": "seat_swap_mean"}, final_policy_set_size=2),
        resolved_snapshot_registry=registry_path,
        resolved_dev_eval=dev_eval_path,
        resolve_final_policy_set_fn=fake_resolve_final_policy_set,
    )

    assert resolved == (
        ["B0 RandomLegal", "policy_000100"],
        {
            "mode": "deterministic_v1",
            "policy_count": 2,
            "snapshot_registry_path": registry_path.as_posix(),
            "dev_eval_summaries_path": dev_eval_path.as_posix(),
            "final_policy_set_size": 2,
        },
    )
    assert observed["resolve"] == {
        "snapshot_registry_path": registry_path,
        "dev_eval_summaries_path": dev_eval_path,
        "config": {"folding": "seat_swap_mean"},
        "final_policy_set_size": 2,
    }
    assert (
        _resolve_deterministic_final_policy_set(
            evaluation=SimpleNamespace(final_policy_set_selection={}, final_policy_set_size=2),
            resolved_snapshot_registry=None,
            resolved_dev_eval=dev_eval_path,
        )
        is None
    )


def test_eval_policy_final_set_resolution_reports_missing_inputs(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _raise_missing_final_policy_inputs

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")

    with pytest.raises(FileNotFoundError, match="requires a snapshot registry") as registry_exc:
        _raise_missing_final_policy_inputs(
            layout=layout,
            resolved_snapshot_registry=None,
            resolved_dev_eval=tmp_path / "dev_eval.json",
            snapshot_registry_path=tmp_path / "explicit" / "registry.json",
            manifest_snapshot_registry=tmp_path / "manifest" / "registry.json",
            dev_eval_summaries_path=None,
            manifest_dev_eval=None,
        )
    assert str(tmp_path / "explicit" / "registry.json") in str(registry_exc.value)

    with pytest.raises(FileNotFoundError, match="requires dev-eval summaries") as dev_eval_exc:
        _raise_missing_final_policy_inputs(
            layout=layout,
            resolved_snapshot_registry=tmp_path / "registry.json",
            resolved_dev_eval=None,
            snapshot_registry_path=None,
            manifest_snapshot_registry=None,
            dev_eval_summaries_path=tmp_path / "explicit" / "dev_eval.json",
            manifest_dev_eval=tmp_path / "manifest" / "dev_eval.json",
        )
    message = str(dev_eval_exc.value)
    assert (tmp_path / "explicit" / "dev_eval.json").as_posix() in message
    assert (tmp_path / "manifest" / "dev_eval.json").as_posix() in message
    assert "training/logs/dev_eval_summaries.json" in message
    assert "training/logs/periodic_dev_eval_summaries.json" in message


def test_eval_policy_manifest_selection_resolves_source_paths_from_manifest(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_manifest_selection import _resolve_selection_inputs_from_manifest

    absolute_dev_eval = tmp_path / "external" / "dev_eval.json"
    snapshot_registry, dev_eval = _resolve_selection_inputs_from_manifest(
        stack_root=tmp_path / "stack",
        manifest={
            "policy_set_selection_details": {
                "source_paths": {
                    "snapshot_registry_json": "runs/main/training/snapshots/registry.json",
                    "dev_eval_summaries_json": absolute_dev_eval.as_posix(),
                }
            }
        },
    )

    assert snapshot_registry == tmp_path / "stack" / "runs" / "main" / "training" / "snapshots" / "registry.json"
    assert dev_eval == absolute_dev_eval


def test_eval_policy_manifest_selection_requires_completed_artifacts(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_manifest_selection import _authoritative_manifest_policy_selection

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_set_selection": [" B0 RandomLegal ", "policy_000100"],
        "policy_set_selection_details": {"mode": "deterministic_v1", "status": "resolved"},
    }

    assert (
        _authoritative_manifest_policy_selection(
            manifest=manifest,
            layout=layout,
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
        )
        is None
    )

    layout.run_summary_path.write_text(
        json.dumps({"canonical_eval_completed": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resolved = _authoritative_manifest_policy_selection(
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert resolved == (
        ["B0 RandomLegal", "policy_000100"],
        {"mode": "deterministic_v1", "status": "resolved", "policy_count": 2},
    )
    assert (
        _authoritative_manifest_policy_selection(
            manifest=manifest,
            layout=layout,
            snapshot_registry_path=tmp_path / "registry.json",
            dev_eval_summaries_path=None,
        )
        is None
    )
