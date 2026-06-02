"""Console reporting for checkpoint interpolation."""

from __future__ import annotations

from pathlib import Path


def checkpoint_interpolation_output_line(
    *,
    checkpoint_path: Path,
    summary_path: Path,
    second_weight: float,
) -> str:
    return (
        f"Interpolated checkpoint written to {checkpoint_path} with second_weight={float(second_weight):.3f}; "
        f"summary written to {summary_path}"
    )


__all__ = ["checkpoint_interpolation_output_line"]
