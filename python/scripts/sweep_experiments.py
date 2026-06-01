#!/usr/bin/env python3
"""Compatibility shim for the package-owned sweep entrypoint."""

from __future__ import annotations

from weiss_rl.experiments import sweep_experiments_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_override_entrypoint_facade

install_package_override_entrypoint_facade(
    globals(),
    _impl,
    ("resolve_devices", "build_sweep_launch_plan", "execute_launch_plan"),
)


if __name__ == "__main__":
    globals()["main"]()
