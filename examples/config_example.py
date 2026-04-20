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
    1) `configs/presets/structured_acceptance_standard.yaml` by default
       or a legacy typed preset if you pass one explicitly
    2) parent preset files via `extends`
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _load_preset_document(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = path.resolve()
    if seen is None:
        seen = set()
    if resolved_path in seen:
        raise ValueError(f"Preset inheritance cycle detected at {resolved_path}")
    seen.add(resolved_path)

    doc = _load_yaml_file(resolved_path)
    parent_name = doc.get("extends")
    if parent_name is None:
        return doc
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ValueError(f"Expected string extends target in {resolved_path}")

    parent_path = (resolved_path.parent / parent_name).resolve()
    parent_doc = _load_preset_document(parent_path, seen=seen)
    child_doc = dict(doc)
    child_doc.pop("extends", None)
    return _deep_merge(parent_doc, child_doc)


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
    preset_config_path: Path | None = None,
    loop_config_path: Path | None = None,
) -> ExampleRunConfig:
    """Resolve example config from the grouped preset workflow."""

    root = repo_root()
    preset_path = preset_config_path or (root / "configs" / "presets" / "structured_acceptance_standard.yaml")
    loop_path = loop_config_path or (root / "configs" / "minimal_loop.yaml")

    preset_doc = _load_preset_document(preset_path)
    env_cfg = preset_doc.get("environment", {})
    eval_cfg = preset_doc.get("evaluation", {})
    loop_doc = _load_yaml_file(loop_path)
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
    if config.action_policy not in {"first_legal", "uniform_legal", "random_legal"}:
        raise ValueError(
            "Unsupported action_policy: "
            f"{config.action_policy}. Expected 'first_legal', 'uniform_legal', or 'random_legal'."
        )

    # Surface one evaluator setting as a sanity check that we read the grouped preset.
    if "eval_sampling_algorithm" not in eval_cfg:
        raise ValueError("evaluation config is missing required key: eval_sampling_algorithm")

    return config
