from __future__ import annotations

import argparse

import pytest

from weiss_rl.training.inputs import (
    expected_sha256,
    normalize_sha256,
    require_matching_hash,
    require_positive_int,
    resolve_run_label,
    spec_mismatch_policy,
)


def test_sha256_helpers_normalize_validate_and_match() -> None:
    digest = "A" * 64

    assert normalize_sha256(f" {digest} ") == "a" * 64
    assert normalize_sha256("not-a-digest") == ""
    assert expected_sha256("", flag_name="--config-hash") == ""
    assert expected_sha256(digest, flag_name="--config-hash") == "a" * 64

    with pytest.raises(ValueError, match="--config-hash must be a 64-character"):
        expected_sha256("xyz", flag_name="--config-hash")

    require_matching_hash(flag_name="--config-hash", expected="", actual="observed")
    require_matching_hash(flag_name="--config-hash", expected="abc", actual="abc")
    with pytest.raises(RuntimeError, match="--config-hash mismatch: expected abc, observed def"):
        require_matching_hash(flag_name="--config-hash", expected="abc", actual="def")


def test_run_label_resolution_preserves_legacy_alias_warning(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser(prog="train.py")

    assert resolve_run_label(parser, " run_a ", "") == "run_a"
    assert resolve_run_label(parser, "", " legacy ") == "legacy"
    captured = capsys.readouterr()
    assert "Warning: --run-id is deprecated; use --run-label instead." in captured.err

    assert resolve_run_label(parser, "same", "same") == "same"
    with pytest.raises(SystemExit):
        resolve_run_label(parser, "new", "old")


def test_positive_int_and_spec_mismatch_policy_helpers() -> None:
    assert require_positive_int("--num-envs", 3) == 3
    with pytest.raises(ValueError, match="--num-envs must be >= 1, got 0"):
        require_positive_int("--num-envs", 0)

    assert spec_mismatch_policy(object()) == "hard_fail"
