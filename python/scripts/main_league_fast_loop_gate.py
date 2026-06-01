#!/usr/bin/env python3
"""Compatibility shim for the package-owned main-league fast-loop gate CLI."""

from __future__ import annotations

from weiss_rl.experiments import main_league_fast_loop_gate_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    _impl.main()
