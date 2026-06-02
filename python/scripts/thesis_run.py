"""Compatibility shim for the package-owned thesis wrapper CLI."""

from __future__ import annotations

from weiss_rl.workflows import thesis_wrapper as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)
__all__ = list(_impl.__all__)


if __name__ == "__main__":
    _impl.main()
