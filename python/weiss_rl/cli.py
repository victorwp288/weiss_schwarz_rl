from __future__ import annotations

from weiss_rl.workflows import runner as _runner

for _name in _runner.__all__:
    globals()[_name] = getattr(_runner, _name)

__all__ = list(_runner.__all__)


if __name__ == "__main__":
    _runner.main()
