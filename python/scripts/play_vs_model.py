#!/usr/bin/env python3
"""Compatibility shim for the package-owned play-vs-model CLI."""

from __future__ import annotations

from weiss_rl.human_play import play_vs_model_entrypoint as _impl
from weiss_rl.workflows.script_compat import install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)


if __name__ == "__main__":
    _impl.main()
