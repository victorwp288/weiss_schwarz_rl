"""Compatibility shim for the package-owned guarded league bootstrap CLI."""

from __future__ import annotations

from weiss_rl.experiments import guarded_league_bootstrap_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    _impl.main()
