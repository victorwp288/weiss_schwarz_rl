from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.config import load_stack_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_policy_stack(
    tmp_path: Path,
    *,
    replay_eval_policy: str = "hard_fail",
    eval_policy: str = "hard_fail",
    fail_on_spec_mismatch: bool = True,
) -> Path:
    repo_root = _repo_root()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    reproducibility_text = (repo_root / "configs" / "reproducibility_locked.yaml").read_text(encoding="utf-8")
    evaluation_text = (repo_root / "configs" / "evaluation_locked.yaml").read_text(encoding="utf-8")

    reproducibility_text = reproducibility_text.replace(
        "replay_eval_mismatch_policy: hard_fail",
        f"replay_eval_mismatch_policy: {replay_eval_policy}",
    )
    reproducibility_text = reproducibility_text.replace(
        "fail_on_spec_mismatch: true",
        f"fail_on_spec_mismatch: {str(fail_on_spec_mismatch).lower()}",
    )
    evaluation_text = evaluation_text.replace(
        "mismatch_policy: hard_fail",
        f"mismatch_policy: {eval_policy}",
    )

    (config_dir / "rl_stack_locked.yaml").write_text(
        "\n".join(
            (
                "rl_stack_locked:",
                "  schema_version: 1",
                "  components:",
                "    reproducibility: configs/reproducibility_locked.yaml",
                "    evaluation: configs/evaluation_locked.yaml",
                "  seed_sets: {}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (config_dir / "reproducibility_locked.yaml").write_text(reproducibility_text, encoding="utf-8")
    (config_dir / "evaluation_locked.yaml").write_text(evaluation_text, encoding="utf-8")
    return config_dir / "rl_stack_locked.yaml"


def test_load_stack_config_accepts_fail_fast_component_policies(tmp_path: Path) -> None:
    stack = load_stack_config(_write_policy_stack(tmp_path))

    assert stack.config.reproducibility is not None
    assert stack.config.evaluation is not None
    assert stack.config.reproducibility.spec_bundle.require_export_spec_bundle is True
    assert stack.config.reproducibility.spec_bundle.persist_in_manifest is True
    assert stack.config.reproducibility.spec_bundle.fail_on_spec_mismatch is True
    assert stack.config.reproducibility.legal_fingerprint.replay_eval_mismatch_policy == "hard_fail"
    assert stack.config.evaluation.legal_fingerprint_checks.mismatch_policy == "hard_fail"


@pytest.mark.parametrize(
    ("replay_eval_policy", "eval_policy", "fail_on_spec_mismatch", "expected_message"),
    [
        ("warn", "hard_fail", True, "reproducibility.legal_fingerprint.replay_eval_mismatch_policy"),
        ("hard_fail", "ignore", True, "evaluation.legal_fingerprint_checks.mismatch_policy"),
        ("hard_fail", "hard_fail", False, "reproducibility.spec_bundle.fail_on_spec_mismatch"),
    ],
)
def test_load_stack_config_rejects_non_fail_fast_component_policies(
    tmp_path: Path,
    replay_eval_policy: str,
    eval_policy: str,
    fail_on_spec_mismatch: bool,
    expected_message: str,
) -> None:
    stack_path = _write_policy_stack(
        tmp_path,
        replay_eval_policy=replay_eval_policy,
        eval_policy=eval_policy,
        fail_on_spec_mismatch=fail_on_spec_mismatch,
    )

    with pytest.raises(ValueError, match=expected_message):
        load_stack_config(stack_path)


def test_load_stack_config_rejects_string_boolean_flags(tmp_path: Path) -> None:
    repo_root = _repo_root()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    reproducibility_text = (repo_root / "configs" / "reproducibility_locked.yaml").read_text(encoding="utf-8")
    reproducibility_text = reproducibility_text.replace("require_export_spec_bundle: true", 'require_export_spec_bundle: "true"')
    reproducibility_text = reproducibility_text.replace("fail_on_spec_mismatch: true", 'fail_on_spec_mismatch: "false"')

    (config_dir / "rl_stack_locked.yaml").write_text(
        "\n".join(
            (
                "rl_stack_locked:",
                "  schema_version: 1",
                "  components:",
                "    reproducibility: configs/reproducibility_locked.yaml",
                "  seed_sets: {}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (config_dir / "reproducibility_locked.yaml").write_text(reproducibility_text, encoding="utf-8")

    with pytest.raises(ValueError, match="must be a boolean"):
        load_stack_config(config_dir / "rl_stack_locked.yaml")
