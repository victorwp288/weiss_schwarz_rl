"""Console reporting for checkpoint snapshot publication."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def checkpoint_publish_output_text(result: Mapping[str, Any]) -> str:
    return json.dumps(dict(result), indent=2, sort_keys=True)


__all__ = ["checkpoint_publish_output_text"]
