#!/usr/bin/env python3
"""Compatibility shim for the package-owned training entrypoint."""

from __future__ import annotations

import sys

from weiss_rl.training import train_entrypoint as _impl
from weiss_rl.training.train_entrypoint_dev_eval_wrappers import install_dev_eval_wrappers
from weiss_rl.training.train_entrypoint_snapshot_wrappers import install_snapshot_wrappers
from weiss_rl.workflows.script_compat import bind_package_script_api, install_package_entrypoint_exports

install_package_entrypoint_exports(globals(), _impl)
bind_package_script_api(_impl, sys.modules[__name__])
install_snapshot_wrappers(globals(), entrypoint_api=lambda: sys.modules[__name__])
install_dev_eval_wrappers(globals(), entrypoint_api=lambda: sys.modules[__name__])


if __name__ == "__main__":
    _impl.main()
