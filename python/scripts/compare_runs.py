#!/usr/bin/env python3
"""Compatibility shim for the package-owned comparison entrypoint."""

from __future__ import annotations

from weiss_rl.workflows import compare_runs_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_override_entrypoint_facade

install_package_override_entrypoint_facade(globals(), _impl, ("render_benchmark_figures",))


if __name__ == "__main__":
    globals()["main"]()
