"""Public thesis workflow command surface."""

from __future__ import annotations

from importlib import import_module

__all__ = ["main"]

_LEGACY_WORKFLOW_PREFIXES = {
    "eval_canonical_": "canonical_eval",
    "eval_entrypoint_": "eval_entrypoint_support",
    "evaluation_": "evaluation_workflow",
    "thesis_wrapper_": "thesis_wrapper_support",
    "training_": "training_workflow",
}


def __getattr__(name: str):
    for legacy_prefix, package_name in _LEGACY_WORKFLOW_PREFIXES.items():
        if not name.startswith(legacy_prefix):
            continue
        suffix = name.removeprefix(legacy_prefix)
        module_name = f"{__name__}.{package_name}.{suffix}"
        try:
            module = import_module(f".{package_name}.{suffix}", __name__)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                break
            raise
        globals()[name] = module
        return module
    for package_name in (
        "artifact_contract",
        "compare_runs",
        "eval_support",
        "figures",
        "verification",
    ):
        module_name = f"{__name__}.{package_name}.{name}"
        try:
            module = import_module(f".{package_name}.{name}", __name__)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    from weiss_rl.workflows.runner import main as runner_main

    runner_main()
