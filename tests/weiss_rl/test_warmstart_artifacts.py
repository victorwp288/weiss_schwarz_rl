from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import weiss_rl.training.warmstarts.warmstart_artifacts as warmstart_artifacts
from weiss_rl.artifacts import ArtifactLayout


def test_warmstart_artifacts_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "paired_outcome_preference_warmstart_support")
    assert not hasattr(training, "warmstart_artifacts")
    assert not hasattr(training, "warmstart_replay_support")


def test_warmstart_run_contract_artifacts_write_manifest_config_and_optional_spec(
    tmp_path,
    monkeypatch,
) -> None:
    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.ensure_directories()
    source_run_dir = tmp_path / "source"
    source_run_dir.mkdir()
    (source_run_dir / "spec_bundle.json").write_text('{"spec": true}\n', encoding="utf-8")
    stack = SimpleNamespace(name="stack")

    monkeypatch.setattr(warmstart_artifacts, "compute_config_hash256", lambda _stack: "c0ffee")
    monkeypatch.setattr(warmstart_artifacts, "canonical_config_dict", lambda _stack: {"config": {"ok": True}})

    warmstart_artifacts.write_warmstart_run_contract_artifacts(
        output_layout=layout,
        stack=stack,
        source_run_dir=source_run_dir,
        spec_hash="ab" * 32,
        manifest_format="unit_manifest_v1",
        run_kind="unit_warmstart",
    )

    assert layout.config_hash_path.read_text(encoding="utf-8") == "c0ffee\n"
    assert json.loads(layout.config_json_path.read_text(encoding="utf-8")) == {"config": {"ok": True}}
    assert layout.spec_hash_path.read_text(encoding="utf-8") == f"{'ab' * 32}\n"
    assert json.loads(layout.spec_bundle_path.read_text(encoding="utf-8")) == {"spec": True}
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "unit_manifest_v1"
    assert manifest["config_hash256"] == "c0ffee"
    assert manifest["spec_hash256"] == "ab" * 32
    expected_run_id = hashlib.sha256(
        json.dumps(
            {
                "kind": "unit_warmstart",
                "run_dir": layout.run_dir.resolve().as_posix(),
                "config_hash256": "c0ffee",
                "spec_hash256": "ab" * 32,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["run_id256"] == expected_run_id


def test_publish_warmstart_snapshot_writes_weights_metadata_and_registry(tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"warmstart weights")
    expected_sha = hashlib.sha256(b"warmstart weights").hexdigest()

    payload = warmstart_artifacts.publish_warmstart_snapshot(
        output_run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        update_count=42,
        policy_id="unit_latest",
        metadata_format="unit_snapshot_meta_v1",
    )

    weights_path = run_dir / "training" / "snapshots" / "unit_latest" / "weights.pt"
    metadata_path = run_dir / "training" / "snapshots" / "unit_latest" / "policy_meta.json"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    assert weights_path.read_bytes() == b"warmstart weights"
    assert payload == {
        "policy_id": "unit_latest",
        "weights_path": weights_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "registry_path": registry_path.as_posix(),
        "weights_sha256": expected_sha,
    }
    assert warmstart_artifacts.sha256_file(weights_path) == expected_sha
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "format": "unit_snapshot_meta_v1",
        "policy_id": "unit_latest",
        "source_checkpoint_path": checkpoint_path.resolve().as_posix(),
        "update": 42,
        "weights_sha256": expected_sha,
    }
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["pinned_snapshots"] == ["unit_latest"]
    assert registry["snapshots"][0]["policy_id"] == "unit_latest"
    assert registry["snapshots"][0]["path"] == "training/snapshots/unit_latest/weights.pt"
    assert registry["snapshots"][0]["weights_sha256"] == expected_sha


def test_warmstart_artifact_factories_specialize_contract_and_snapshot_helpers(tmp_path, monkeypatch) -> None:
    layout = ArtifactLayout.from_run_dir(tmp_path / "factory_run")
    layout.ensure_directories()
    stack = SimpleNamespace(name="factory-stack")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"factory weights")

    monkeypatch.setattr(warmstart_artifacts, "compute_config_hash256", lambda _stack: "factory-config")
    monkeypatch.setattr(warmstart_artifacts, "canonical_config_dict", lambda _stack: {"factory": True})

    write_contract = warmstart_artifacts.warmstart_run_contract_writer(
        manifest_format="factory_manifest_v1",
        run_kind="factory_warmstart",
    )
    publish_snapshot = warmstart_artifacts.warmstart_snapshot_publisher(
        policy_id="factory_latest",
        metadata_format="factory_snapshot_v1",
    )

    write_contract(output_layout=layout, stack=stack, source_run_dir=None, spec_hash="cd" * 32)
    payload = publish_snapshot(output_run_dir=layout.run_dir, checkpoint_path=checkpoint_path, update_count=7)

    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "factory_manifest_v1"
    assert (
        manifest["run_id256"]
        == hashlib.sha256(
            json.dumps(
                {
                    "kind": "factory_warmstart",
                    "run_dir": layout.run_dir.resolve().as_posix(),
                    "config_hash256": "factory-config",
                    "spec_hash256": "cd" * 32,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    metadata = json.loads(Path(payload["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["format"] == "factory_snapshot_v1"
    assert metadata["policy_id"] == "factory_latest"
    assert metadata["update"] == 7
