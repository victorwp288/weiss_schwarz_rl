from __future__ import annotations

from pathlib import Path

_PRESET_PATHS = {
    "standard": Path("configs/presets/structured_acceptance_standard.yaml"),
    "standard-auto-gpu": Path("configs/presets/structured_acceptance_standard_auto_gpu.yaml"),
    "standard-thesis-eval": Path("configs/presets/structured_acceptance_standard_thesis_eval.yaml"),
    "standard-multideck": Path("configs/presets/structured_acceptance_standard_multideck.yaml"),
    "ablate-no-gru": Path("configs/thesis/ablations/no_gru.yaml"),
    "ablate-ppo-lite": Path("configs/thesis/ablations/ppo_lite.yaml"),
    "ablate-terminal-only": Path("configs/thesis/ablations/terminal_only_reward.yaml"),
}
_DEFAULT_EVAL_PRESET = "standard-thesis-eval"
_DEFAULT_EVAL_PRESET_OVERRIDES = {
    "standard-multideck": "standard-multideck",
}


def _resolve_cli_path(*, repo_root: Path, path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _resolve_stack_config(*, repo_root: Path, stack_config: Path | None, preset: str) -> Path:
    if stack_config is not None:
        return _resolve_cli_path(repo_root=repo_root, path=stack_config)
    return (repo_root / _PRESET_PATHS[preset]).resolve()


def _resolve_eval_stack_config(
    *,
    repo_root: Path,
    eval_stack_config: Path | None,
    train_stack_config: Path | None,
    eval_preset: str,
) -> Path:
    if eval_stack_config is not None:
        return _resolve_cli_path(repo_root=repo_root, path=eval_stack_config)
    if train_stack_config is not None and not eval_preset:
        return _resolve_cli_path(repo_root=repo_root, path=train_stack_config)
    return (repo_root / _PRESET_PATHS[eval_preset]).resolve()


def _default_eval_preset_for_preset(preset: str) -> str:
    return _DEFAULT_EVAL_PRESET_OVERRIDES.get(preset, _DEFAULT_EVAL_PRESET)
