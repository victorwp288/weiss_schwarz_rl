from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run these examples") from exc


@dataclass(slots=True)
class ExampleRunConfig:
    """Runtime config used by the loop example.

    Values come from:
    1) `configs/rl_stack_locked.yaml` (component wiring)
    2) component files (`environment_locked.yaml`, `evaluation_locked.yaml`)
    3) `configs/minimal_loop.yaml` (loop-specific overrides)
    """

    mode: str = "fast"
    num_envs: int = 2
    num_steps: int = 300
    seed: int = 7
    max_decisions: int = 2000
    max_ticks: int = 100_000
    observation_visibility: str = "public"
    action_policy: str = "uniform_legal"
    log_every: int = 30
    error_policy: str = "replace"
    auto_reset_done: bool = True
    auto_reset_engine_errors: bool = True


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _as_int(source: dict[str, Any], key: str, fallback: int) -> int:
    value = source.get(key, fallback)
    return int(value)


def _as_bool(source: dict[str, Any], key: str, fallback: bool) -> bool:
    value = source.get(key, fallback)
    return bool(value)


def _as_str(source: dict[str, Any], key: str, fallback: str) -> str:
    value = source.get(key, fallback)
    return str(value).strip()


def load_example_config(
    *,
    stack_config_path: Path | None = None,
    loop_config_path: Path | None = None,
) -> ExampleRunConfig:
    """Resolve example config from the split locked config stack."""

    root = repo_root()
    stack_path = stack_config_path or (root / "configs" / "rl_stack_locked.yaml")
    loop_path = loop_config_path or (root / "configs" / "minimal_loop.yaml")

    stack_doc = _load_yaml_file(stack_path)
    stack = stack_doc.get("rl_stack_locked", stack_doc)
    if not isinstance(stack, dict):
        raise ValueError(f"Expected mapping in {stack_path} at key 'rl_stack_locked'")

    components = stack.get("components", {})
    if not isinstance(components, dict):
        raise ValueError(f"Expected 'components' mapping in {stack_path}")

    env_component = components.get("environment")
    eval_component = components.get("evaluation")
    if not isinstance(env_component, str) or not isinstance(eval_component, str):
        raise ValueError("Stack config must define string component paths for environment/evaluation")

    env_doc = _load_yaml_file(root / env_component)
    eval_doc = _load_yaml_file(root / eval_component)
    loop_doc = _load_yaml_file(loop_path)

    env_cfg = env_doc.get("environment", env_doc)
    eval_cfg = eval_doc.get("evaluation", eval_doc)
    loop_cfg = loop_doc.get("minimal_loop", loop_doc)

    if not isinstance(env_cfg, dict) or not isinstance(eval_cfg, dict) or not isinstance(loop_cfg, dict):
        raise ValueError("Expected environment/evaluation/minimal_loop sections to be mappings")

    visibility = _as_str(
        loop_cfg,
        "observation_visibility",
        _as_str(env_cfg, "observation_visibility", _as_str(env_cfg, "visibility", "public")),
    ).lower()

    max_decisions_default = _as_int(
        env_cfg,
        "max_decisions",
        _as_int(env_cfg, "max_decisions_per_episode", 2000),
    )

    action_policy = _as_str(loop_cfg, "action_policy", "uniform_legal").lower()
    if action_policy in {"uniform", "random"}:
        action_policy = "uniform_legal"

    config = ExampleRunConfig(
        mode=_as_str(loop_cfg, "mode", "fast").lower(),
        num_envs=_as_int(loop_cfg, "num_envs", 2),
        num_steps=_as_int(loop_cfg, "num_steps", 300),
        seed=_as_int(loop_cfg, "seed", 7),
        max_decisions=_as_int(loop_cfg, "max_decisions", max_decisions_default),
        max_ticks=_as_int(loop_cfg, "max_ticks", _as_int(env_cfg, "max_ticks", 100_000)),
        observation_visibility=visibility,
        action_policy=action_policy,
        log_every=max(1, _as_int(loop_cfg, "log_every", 30)),
        error_policy=_as_str(loop_cfg, "error_policy", "replace").lower(),
        auto_reset_done=_as_bool(loop_cfg, "auto_reset_done", True),
        auto_reset_engine_errors=_as_bool(loop_cfg, "auto_reset_engine_errors", True),
    )

    if config.mode not in {"fast", "inspect"}:
        raise ValueError(f"Unsupported mode: {config.mode}. Expected 'fast' or 'inspect'.")
    if config.action_policy not in {"first_legal", "uniform_legal"}:
        raise ValueError(
            f"Unsupported action_policy: {config.action_policy}. "
            "Expected 'first_legal' or 'uniform_legal'."
        )

    # Surface one evaluator setting as a sanity check that we read the new stack.
    if "eval_sampling_algorithm" not in eval_cfg:
        raise ValueError("evaluation config is missing required key: eval_sampling_algorithm")

    return config
