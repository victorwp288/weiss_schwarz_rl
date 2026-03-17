from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_config_example_module():
    module_path = Path(__file__).resolve().parents[3] / "examples" / "config_example.py"
    spec = importlib.util.spec_from_file_location("test_config_example_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_example_config_accepts_random_legal_policy(tmp_path: Path) -> None:
    module = _load_config_example_module()

    module.repo_root = lambda: tmp_path

    stack_path = tmp_path / "configs" / "stack.yaml"
    env_path = tmp_path / "configs" / "environment.yaml"
    eval_path = tmp_path / "configs" / "evaluation.yaml"
    loop_path = tmp_path / "configs" / "minimal_loop.yaml"
    stack_path.parent.mkdir(parents=True)

    stack_path.write_text(
        "rl_stack_locked:\n"
        "  components:\n"
        "    environment: configs/environment.yaml\n"
        "    evaluation: configs/evaluation.yaml\n",
        encoding="utf-8",
    )
    env_path.write_text(
        "environment:\n"
        "  max_decisions: 2000\n"
        "  max_ticks: 100000\n"
        "  observation_visibility: public\n",
        encoding="utf-8",
    )
    eval_path.write_text(
        "evaluation:\n"
        "  eval_sampling_algorithm: pinned_cpu_cdf\n",
        encoding="utf-8",
    )
    loop_path.write_text(
        "minimal_loop:\n"
        "  action_policy: random_legal\n",
        encoding="utf-8",
    )

    config = module.load_example_config(stack_config_path=stack_path, loop_config_path=loop_path)

    assert config.action_policy == "random_legal"
