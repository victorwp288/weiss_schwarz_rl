from __future__ import annotations

from typing import Any


def build_canonical_metagame_output(*, layout: Any, study_config: Any, dependencies: Any) -> dict[str, Any]:
    return dependencies.build_sensitivity_report_fn(
        final_eval_dir=layout.final_eval_dir,
        out_dir=layout.metagame_dir,
        metagame_config=study_config.metagame,
        sensitivity_config=study_config.sensitivity,
    )


__all__ = ["build_canonical_metagame_output"]
