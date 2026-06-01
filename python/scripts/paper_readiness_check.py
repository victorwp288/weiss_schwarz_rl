#!/usr/bin/env python3
"""Compatibility shim for the package-owned paper-readiness check CLI."""

from __future__ import annotations

from weiss_rl.eval import paper_readiness_check_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    _impl.main()
