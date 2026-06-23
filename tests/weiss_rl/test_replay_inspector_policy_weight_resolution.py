from __future__ import annotations

from pathlib import Path

from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.replay.inspection_policy_loading import resolve_policy_weights_path


def test_resolve_policy_weights_path_prefers_run_dir_for_relative_specs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    cwd_dir = tmp_path / "cwd"
    run_dir.mkdir()
    cwd_dir.mkdir()

    relative_spec = Path("training/snapshots/policy_a/weights.pt")
    cwd_weights_path = cwd_dir / relative_spec
    cwd_weights_path.parent.mkdir(parents=True)
    cwd_weights_path.write_bytes(b"cwd")

    run_dir_weights_path = run_dir / relative_spec
    run_dir_weights_path.parent.mkdir(parents=True)
    run_dir_weights_path.write_bytes(b"run-dir")

    monkeypatch.chdir(cwd_dir)

    resolved_path, label = resolve_policy_weights_path(
        spec=relative_spec.as_posix(),
        run_dir=run_dir,
        registry=None,
    )

    assert label == relative_spec.as_posix()
    assert resolved_path == run_dir_weights_path.resolve()


def test_resolve_policy_weights_path_accepts_imported_seed_wrapped_suffix_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    actual_policy_id = "seed_outer_seed_inner_policy_000002"
    requested_policy_id = "seed_inner_policy_000002"

    weights_path = run_dir / "training" / "snapshots" / actual_policy_id / "weights.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"weights")

    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id=actual_policy_id,
        update=2,
        weights_sha256="sha",
        path=weights_path.relative_to(run_dir).as_posix(),
    )

    resolved_path, label = resolve_policy_weights_path(
        spec=requested_policy_id,
        run_dir=run_dir,
        registry=registry,
    )

    assert label == actual_policy_id
    assert resolved_path == weights_path.resolve()
