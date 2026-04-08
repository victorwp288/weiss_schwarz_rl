"""Paper figure generation scaffold."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast


def render_placeholder_figure(out_path: Path) -> None:
    """Write a simple placeholder artifact until plotting is implemented."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("placeholder_figure\n", encoding="utf-8")


def render_public_demo_figures(*, final_eval_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Render clearly-labeled demo-only figure placeholders from public toy eval artifacts."""
    summary_path = final_eval_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing final_eval summary.json: {summary_path}")

    payload = cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))
    metadata = cast(dict[str, Any], payload.get("metadata", {}))
    if not bool(metadata.get("demo_only", False)):
        raise ValueError("public-demo figure rendering requires final_eval metadata.demo_only=true")

    out_dir.mkdir(parents=True, exist_ok=True)
    policy_ids = [str(policy_id) for policy_id in payload.get("policy_ids", [])]
    mean_matrix = cast(dict[str, Any], cast(dict[str, Any], payload.get("matrices", {})).get("mean", {}))
    mean_values = cast(list[list[float]], mean_matrix.get("values", []))

    placeholder_path = out_dir / "toy_demo_placeholder.txt"
    placeholder_path.write_text(
        "\n".join(
            (
                "toy_public_demo_placeholder_figure",
                str(metadata.get("warning", "demo-only artifact")),
                f"source_final_eval_dir={final_eval_dir.as_posix()}",
                f"policy_count={len(policy_ids)}",
                "",
            )
        ),
        encoding="utf-8",
    )

    matrix_csv_path = out_dir / "toy_demo_mean_matrix.csv"
    with matrix_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id, row in zip(policy_ids, mean_values, strict=True):
            writer.writerow([focal_policy_id, *row])

    manifest_path = out_dir / "toy_demo_manifest.json"
    manifest_payload = {
        "kind": "toy_public_demo_figures_v1",
        "demo_only": True,
        "public_safe": True,
        "warning": metadata.get("warning", "demo-only artifact"),
        "source_final_eval_dir": final_eval_dir.as_posix(),
        "source_summary_path": summary_path.as_posix(),
        "policy_ids": policy_ids,
        "artifacts": {
            "placeholder": placeholder_path.name,
            "mean_matrix_csv": matrix_csv_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "placeholder": placeholder_path,
        "mean_matrix_csv": matrix_csv_path,
        "manifest": manifest_path,
    }
