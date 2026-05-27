from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.artifacts.manifest import default_run_dir_name
from weiss_rl.artifacts.reproducibility import compute_run_id64, compute_run_id256
from weiss_rl.training.run_identity import new_run_identity, resume_run_identity


def test_new_run_identity_computes_run_ids_and_default_directory_name() -> None:
    spec_hash = "ab" * 32
    config_hash = "cd" * 32
    git_commit = "ef" * 20
    start_nonce = 123

    identity = new_run_identity(
        spec_hash256=spec_hash,
        config_hash256=config_hash,
        git_commit=git_commit,
        start_nonce=start_nonce,
        run_label="",
    )

    expected_run_id64 = f"{compute_run_id64(spec_hash, config_hash, git_commit, start_nonce):016x}"
    assert identity.run_id256 == compute_run_id256(spec_hash, config_hash, git_commit, start_nonce)
    assert identity.run_id64 == expected_run_id64
    assert identity.run_dir_name == default_run_dir_name(expected_run_id64)


def test_new_run_identity_uses_explicit_run_label() -> None:
    identity = new_run_identity(
        spec_hash256="ab" * 32,
        config_hash256="cd" * 32,
        git_commit="",
        start_nonce=123,
        run_label="named_run",
    )

    assert identity.run_dir_name == "named_run"


def test_resume_run_identity_loads_manifest_ids_and_validates_hashes(tmp_path: Path) -> None:
    manifest = {
        "run_id256": " AB ",
        "run_id64": " CD ",
        "spec_hash256": "aa" * 32,
        "config_hash256": "bb" * 32,
    }

    identity = resume_run_identity(
        manifest,
        manifest_path=tmp_path / "manifest.json",
        run_dir_name="existing",
        expected_spec_hash256="aa" * 32,
        expected_config_hash256="bb" * 32,
    )

    assert identity.run_id256 == "ab"
    assert identity.run_id64 == "cd"
    assert identity.run_dir_name == "existing"


def test_resume_run_identity_reports_spec_and_config_hash_mismatches(tmp_path: Path) -> None:
    manifest = {
        "run_id256": "ab",
        "run_id64": "cd",
        "spec_hash256": "aa" * 32,
        "config_hash256": "bb" * 32,
    }

    with pytest.raises(RuntimeError, match="resume run spec hash mismatch"):
        resume_run_identity(
            manifest,
            manifest_path=tmp_path / "manifest.json",
            run_dir_name="existing",
            expected_spec_hash256="cc" * 32,
            expected_config_hash256="bb" * 32,
        )
    with pytest.raises(RuntimeError, match="resume run config hash mismatch"):
        resume_run_identity(
            manifest,
            manifest_path=tmp_path / "manifest.json",
            run_dir_name="existing",
            expected_spec_hash256="aa" * 32,
            expected_config_hash256="cc" * 32,
        )
