"""Compatibility shim for the package-owned learning progress diagnostic CLI."""

from __future__ import annotations

from weiss_rl.diagnostics import learning_progress as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    _impl.main()
