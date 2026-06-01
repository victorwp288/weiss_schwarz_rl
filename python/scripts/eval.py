"""Compatibility shim for the package-owned evaluation CLI."""

from __future__ import annotations

import sys

from weiss_rl.workflows import eval_entrypoint as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
