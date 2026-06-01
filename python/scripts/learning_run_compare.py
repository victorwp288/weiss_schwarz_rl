#!/usr/bin/env python3
"""Compatibility shim for the package-owned learning run comparison CLI."""

from __future__ import annotations

from weiss_rl.experiments import learning_run_compare_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    _impl.main()
