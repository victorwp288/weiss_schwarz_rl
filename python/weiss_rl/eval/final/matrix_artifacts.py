"""Matrix artifact writers for final evaluation outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout


def write_final_eval_matrix_artifacts(
    *,
    output_dir: Path,
    matrices: Mapping[str, Mapping[str, Any]],
    layout: ArtifactLayout | None,
) -> None:
    matrices_dir = layout.final_eval_matrices_dir if layout is not None else output_dir / "matrices"
    for field, matrix_payload in matrices.items():
        json_path = matrices_dir / f"{field}.json"
        csv_path = matrices_dir / f"{field}.csv"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(matrix_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_matrix_csv(csv_path, matrix_payload)
        if layout is not None:
            legacy_name = legacy_payoff_matrix_name(field)
            if legacy_name is not None:
                write_matrix_csv(layout.final_eval_payoff_matrix_csv(legacy_name), matrix_payload)


def write_matrix_csv(path: Path, matrix_payload: Mapping[str, Any]) -> None:
    policy_ids = cast(list[str], matrix_payload["policy_ids"])
    values = cast(list[list[Any]], matrix_payload["values"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id, row in zip(policy_ids, values, strict=True):
            writer.writerow([focal_policy_id, *row])


def legacy_payoff_matrix_name(field: str) -> str | None:
    if field == "mean":
        return "p_mean"
    return None


__all__ = [
    "legacy_payoff_matrix_name",
    "write_final_eval_matrix_artifacts",
    "write_matrix_csv",
]
