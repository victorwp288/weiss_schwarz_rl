from __future__ import annotations

from pathlib import Path

from weiss_rl.eval.policies.resolution import (
    _is_recursive_registry_search_root as policy_resolution_is_recursive_registry_search_root,
)
from weiss_rl.eval.policies.resolution import (
    _should_include_common_search_root as policy_resolution_should_include_common_search_root,
)


def test_recursive_registry_search_root_rejects_filesystem_anchor() -> None:
    anchor_root = Path(Path.cwd().anchor)

    assert policy_resolution_is_recursive_registry_search_root(anchor_root) is False
    assert policy_resolution_is_recursive_registry_search_root(anchor_root / "workspace") is True


def test_common_search_root_is_only_used_for_sibling_search_trees(tmp_path: Path) -> None:
    sibling_common_root = tmp_path / "staging"
    sibling_search_roots = [
        sibling_common_root / "runs",
        sibling_common_root / "cache",
    ]
    broad_common_root = tmp_path / "home"
    broad_search_roots = [
        broad_common_root / "Desktop" / "repo" / "runs",
        broad_common_root / "Downloads",
    ]

    assert (
        policy_resolution_should_include_common_search_root(
            search_roots=sibling_search_roots,
            common_search_root=sibling_common_root,
        )
        is True
    )
    assert (
        policy_resolution_should_include_common_search_root(
            search_roots=broad_search_roots,
            common_search_root=broad_common_root,
        )
        is False
    )
