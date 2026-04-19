from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.config import canonical_config_dict, compute_config_hash256, load_stack_config
from weiss_rl.manifest import RunManifest, build_seed_file_manifest, default_run_dir_name, write_run_artifacts
from weiss_rl.tests._config_paths import canonical_stack_config_path


def test_write_run_artifacts_creates_manifest_scaffold(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    manifest = RunManifest(
        run_id256="ab" * 32,
        run_id64="0123456789abcdef",
        start_nonce=7,
        git_commit="deadbeef" * 5,
        git_dirty=False,
        spec_hash256="cd" * 32,
        config_hash256=compute_config_hash256(stack),
        simulator={"version": "0.7.0"},
        spec_bundle={"spec_hash": 123},
        config_canonical=canonical_config_dict(stack),
        seed_files=build_seed_file_manifest(stack.seed_sets, root=stack.root),
        hardware={"platform": "test"},
        evaluation_pinning={"eval_sampling_algorithm": "pinned_cdf_pcg_v1"},
        policy_set_selection=["B0 RandomLegal", "B1 NoLeague baseline"],
    )

    artifacts = write_run_artifacts(tmp_path, manifest)

    assert artifacts.run_dir_name == default_run_dir_name("0123456789abcdef")
    assert artifacts.run_dir == tmp_path / "run_0123456789abcdef"
    assert artifacts.manifest_path.exists()
    assert artifacts.spec_bundle_path.exists()
    assert artifacts.spec_hash_path.read_text(encoding="utf-8") == f"{manifest.spec_hash256}\n"
    assert artifacts.config_hash_path.read_text(encoding="utf-8") == f"{manifest.config_hash256}\n"
    assert artifacts.config_json_path.exists()
    assert json.loads(artifacts.spec_bundle_path.read_text(encoding="utf-8")) == {"spec_hash": 123}
    assert json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))["simulator"]["version"] == "0.7.0"
    assert (artifacts.run_dir / "checkpoints").is_dir()
    assert (artifacts.run_dir / "eval").is_dir()
    assert (artifacts.run_dir / "figures").is_dir()
    assert (artifacts.run_dir / "logs").is_dir()
    assert (artifacts.run_dir / "replays").is_dir()


def test_write_run_artifacts_uses_explicit_run_label(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id256="ab" * 32,
        run_id64="0123456789abcdef",
        start_nonce=7,
        git_commit="deadbeef" * 5,
        git_dirty=False,
    )

    artifacts = write_run_artifacts(tmp_path, manifest, run_label="smoke_local")

    assert artifacts.run_dir_name == "smoke_local"
    assert artifacts.run_dir == tmp_path / "smoke_local"
