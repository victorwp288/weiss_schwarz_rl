#!/usr/bin/env python3
"""Compatibility shim for the package-owned repo hygiene check."""

from __future__ import annotations

from weiss_rl.diagnostics import repo_hygiene_check_entrypoint as _impl

main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
