from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.config import load_stack_config
from weiss_rl.eval.simulator_runner import resolve_eval_policies

from ._config_paths import canonical_stack_config_path
from .heuristic_public_test_support import _heuristic_spec_bundle, _write_eval_snapshot, _write_snapshot_registry


def test_resolve_eval_policies_requires_full_requested_snapshot_set_for_copied_registry(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_specs = [
        ("policy_000100", 100),
        ("policy_000200", 200),
        ("policy_000300", 300),
        ("policy_000400", 400),
        ("policy_000500", 500),
    ]
    snapshots: list[tuple[str, int, Path]] = []
    for policy_id, update in snapshot_specs:
        snapshots.append(
            (
                policy_id,
                update,
                _write_eval_snapshot(
                    stack=stack,
                    run_dir=source_run_dir,
                    policy_id=policy_id,
                    update=update,
                ),
            )
        )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=snapshots,
    )

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    for policy_id in ("policy_000100", "policy_000300", "policy_000500"):
        source_weights = next(
            weights_path for snapshot_id, _update, weights_path in snapshots if snapshot_id == policy_id
        )
        copied_weights = tmp_path / "cache" / "training" / "snapshots" / policy_id / "weights.pt"
        copied_weights.parent.mkdir(parents=True, exist_ok=True)
        copied_weights.write_bytes(source_weights.read_bytes())
    consumer_run_dir = tmp_path / "runs" / "consumer_run_full_request"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[policy_id for policy_id, _update in snapshot_specs],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert {resolved[policy_id].source_run_dir for policy_id, _update in snapshot_specs} == {
        source_run_dir.resolve().as_posix()
    }
