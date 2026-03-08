from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.config import load_stack_config


def _write_stack(
    tmp_path: Path,
    *,
    replay_eval_policy: str = "hard_fail",
    eval_policy: str = "hard_fail",
    fail_on_spec_mismatch: bool = True,
) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    (config_dir / "rl_stack_locked.yaml").write_text(
        """
rl_stack_locked:
  components:
    reproducibility: configs/reproducibility_locked.yaml
    evaluation: configs/evaluation_locked.yaml
  seed_sets: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "reproducibility_locked.yaml").write_text(
        f"""
reproducibility:
  spec_bundle:
    fail_on_spec_mismatch: {str(fail_on_spec_mismatch).lower()}
  legal_fingerprint:
    replay_eval_mismatch_policy: {replay_eval_policy}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "evaluation_locked.yaml").write_text(
        f"""
evaluation:
  legal_fingerprint_checks:
    mismatch_policy: {eval_policy}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_dir / "rl_stack_locked.yaml"


def test_load_stack_config_accepts_fail_fast_component_policies(tmp_path: Path) -> None:
    stack_path = _write_stack(tmp_path)

    config = load_stack_config(stack_path)

    assert config.spec_mismatch_policy == "hard_fail"


@pytest.mark.parametrize(
    ("replay_eval_policy", "eval_policy", "fail_on_spec_mismatch", "expected_message"),
    [
        ("warn", "hard_fail", True, "replay_eval_mismatch_policy"),
        ("hard_fail", "ignore", True, "mismatch_policy"),
        ("hard_fail", "hard_fail", False, "fail_on_spec_mismatch"),
    ],
)
def test_load_stack_config_rejects_non_fail_fast_component_policies(
    tmp_path: Path,
    replay_eval_policy: str,
    eval_policy: str,
    fail_on_spec_mismatch: bool,
    expected_message: str,
) -> None:
    stack_path = _write_stack(
        tmp_path,
        replay_eval_policy=replay_eval_policy,
        eval_policy=eval_policy,
        fail_on_spec_mismatch=fail_on_spec_mismatch,
    )

    with pytest.raises(ValueError, match=expected_message):
        load_stack_config(stack_path)
