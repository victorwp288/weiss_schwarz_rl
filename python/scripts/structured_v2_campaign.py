#!/usr/bin/env python3
"""Compatibility shim for the package-owned structured acceptance campaign CLI."""

from __future__ import annotations

from weiss_rl.experiments import structured_v2_campaign_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
