from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from types import ModuleType
from typing import Any


def install_package_entrypoint_exports(target_globals: MutableMapping[str, object], package_module: ModuleType) -> None:
    for name in dir(package_module):
        if name.startswith("__") and name.endswith("__"):
            continue
        target_globals[name] = getattr(package_module, name)


def bind_package_script_api(
    package_module: ModuleType,
    script_module: ModuleType,
    *,
    api_attr_name: str = "_SCRIPT_COMPAT_API",
) -> None:
    setattr(package_module, api_attr_name, script_module)


def run_package_main_with_script_overrides(
    package_module: ModuleType,
    script_globals: MutableMapping[str, object],
    override_names: Iterable[str],
) -> Any:
    originals: dict[str, object] = {}
    for name in override_names:
        originals[name] = getattr(package_module, name)
        setattr(package_module, name, script_globals[name])
    try:
        return package_module.main()
    finally:
        for name, original in originals.items():
            setattr(package_module, name, original)


def build_package_override_main(
    package_module: ModuleType,
    script_globals: MutableMapping[str, object],
    override_names: Iterable[str],
) -> Any:
    override_names = tuple(override_names)

    def main() -> Any:
        return run_package_main_with_script_overrides(package_module, script_globals, override_names)

    return main


def install_package_override_entrypoint_facade(
    target_globals: MutableMapping[str, object],
    package_module: ModuleType,
    override_names: Iterable[str],
) -> None:
    install_package_entrypoint_exports(target_globals, package_module)
    target_globals["main"] = build_package_override_main(package_module, target_globals, override_names)


__all__ = [
    "bind_package_script_api",
    "build_package_override_main",
    "install_package_entrypoint_exports",
    "install_package_override_entrypoint_facade",
    "run_package_main_with_script_overrides",
]
