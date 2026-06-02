from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[3]

PUBLIC_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "configs" / "README.md",
    REPO_ROOT / "configs" / "thesis" / "ablations" / "README.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "getting_started.md",
    REPO_ROOT / "docs" / "evaluation.md",
    REPO_ROOT / "docs" / "artifact_contract.md",
    REPO_ROOT / "docs" / "artifacts.md",
    REPO_ROOT / "docs" / "architecture.md",
    REPO_ROOT / "docs" / "configuration.md",
    REPO_ROOT / "docs" / "reproducibility.md",
    REPO_ROOT / "docs" / "simulator_compatibility.md",
    REPO_ROOT / "docs" / "testing.md",
    REPO_ROOT / "docs" / "troubleshooting.md",
    REPO_ROOT / "docs" / "documentation_maintenance.md",
    REPO_ROOT / "docs" / "thesis_workflow.md",
    REPO_ROOT / "runs" / "README.md",
    REPO_ROOT / "tests" / "README.md",
)

CANONICAL_THESIS_ABLATIONS = frozenset(
    {
        "configs/thesis/ablations/no_gru.yaml",
        "configs/thesis/ablations/ppo_lite.yaml",
        "configs/thesis/ablations/terminal_only_reward.yaml",
    }
)

ABLATION_CONFIG_RE = re.compile(r"configs/thesis/ablations/[A-Za-z0-9_.-]+\.yaml")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def test_public_docs_only_advertise_canonical_thesis_ablations() -> None:
    unexpected: list[str] = []

    for doc_path in PUBLIC_DOCS:
        text = doc_path.read_text(encoding="utf-8")
        for match in ABLATION_CONFIG_RE.findall(text):
            if match not in CANONICAL_THESIS_ABLATIONS:
                unexpected.append(f"{doc_path.relative_to(REPO_ROOT)}: {match}")

    assert unexpected == []


def test_public_docs_local_markdown_links_resolve() -> None:
    missing: list[str] = []

    for doc_path in PUBLIC_DOCS:
        text = doc_path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            target = raw_target.strip()
            if not target or target.startswith("#") or "://" in target or target.startswith("mailto:"):
                continue
            path_part = unquote(target.split("#", 1)[0])
            if not path_part:
                continue
            resolved = (doc_path.parent / path_part).resolve()
            if not resolved.exists():
                missing.append(f"{doc_path.relative_to(REPO_ROOT)}: {target}")

    assert missing == []
