from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

PUBLIC_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "configs" / "README.md",
    REPO_ROOT / "configs" / "thesis" / "ablations" / "README.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "experiments.md",
    REPO_ROOT / "docs" / "standard_recipe.md",
    REPO_ROOT / "docs" / "thesis_workflow.md",
    REPO_ROOT / "python" / "scripts" / "README.md",
)

CANONICAL_THESIS_ABLATIONS = frozenset(
    {
        "configs/thesis/ablations/no_gru.yaml",
        "configs/thesis/ablations/ppo_lite.yaml",
        "configs/thesis/ablations/terminal_only_reward.yaml",
    }
)

ABLATION_CONFIG_RE = re.compile(r"configs/thesis/ablations/[A-Za-z0-9_.-]+\.yaml")


def test_public_docs_only_advertise_canonical_thesis_ablations() -> None:
    unexpected: list[str] = []

    for doc_path in PUBLIC_DOCS:
        text = doc_path.read_text(encoding="utf-8")
        for match in ABLATION_CONFIG_RE.findall(text):
            if match not in CANONICAL_THESIS_ABLATIONS:
                unexpected.append(f"{doc_path.relative_to(REPO_ROOT)}: {match}")

    assert unexpected == []
