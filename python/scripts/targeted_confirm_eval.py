"""Compatibility shim for the package-owned targeted confirmation eval CLI."""

from __future__ import annotations

from weiss_rl.eval import targeted_confirm_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    _impl.main()
