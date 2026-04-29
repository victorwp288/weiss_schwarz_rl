from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.config import load_stack_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _temp_repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "python").mkdir()
    return tmp_path


def test_load_stack_config_accepts_fail_fast_policies() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "thesis_locked.yaml")

    assert stack.config.reproducibility is not None
    assert stack.config.evaluation is not None
    assert stack.config.reproducibility.spec_bundle.require_export_spec_bundle is True
    assert stack.config.reproducibility.spec_bundle.persist_in_manifest is True
    assert stack.config.reproducibility.spec_bundle.fail_on_spec_mismatch is True
    assert stack.config.reproducibility.legal_fingerprint.replay_eval_mismatch_policy == "hard_fail"
    assert stack.config.evaluation.legal_fingerprint_checks.mismatch_policy == "hard_fail"


@pytest.mark.parametrize(
    ("needle", "replacement", "expected_message"),
    [
        (
            "replay_eval_mismatch_policy: hard_fail",
            "replay_eval_mismatch_policy: warn",
            "reproducibility.legal_fingerprint.replay_eval_mismatch_policy",
        ),
        (
            "  legal_fingerprint_checks:\n    enabled: true\n    version: legal_fingerprint_v1\n    require_strictly_increasing_legal_ids: true\n    mismatch_policy: hard_fail",
            "  legal_fingerprint_checks:\n    enabled: true\n    version: legal_fingerprint_v1\n    require_strictly_increasing_legal_ids: true\n    mismatch_policy: ignore",
            "evaluation.legal_fingerprint_checks.mismatch_policy",
        ),
        (
            "fail_on_spec_mismatch: true",
            "fail_on_spec_mismatch: false",
            "reproducibility.spec_bundle.fail_on_spec_mismatch",
        ),
    ],
)
def test_load_stack_config_rejects_non_fail_fast_component_policies(
    tmp_path: Path,
    needle: str,
    replacement: str,
    expected_message: str,
) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    stack_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    stack_path.write_text(
        (repo_root / "configs/thesis_locked.yaml").read_text(encoding="utf-8").replace(needle, replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_message):
        load_stack_config(stack_path)


def test_load_stack_config_rejects_string_boolean_flags(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fake_repo = _temp_repo(tmp_path)
    stack_path = fake_repo / "configs" / "typed_thesis_locked.yaml"
    text = (repo_root / "configs/thesis_locked.yaml").read_text(encoding="utf-8")
    text = text.replace("require_export_spec_bundle: true", 'require_export_spec_bundle: "true"', 1)
    text = text.replace("fail_on_spec_mismatch: true", 'fail_on_spec_mismatch: "false"', 1)
    stack_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="must be a boolean"):
        load_stack_config(stack_path)
