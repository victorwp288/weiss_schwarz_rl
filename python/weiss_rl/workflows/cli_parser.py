"""Compatibility facade for the package workflow parser.

The canonical parser lives in `weiss_rl.workflows.parsers` and is assembled from
training/evaluation/controller parser modules. This module keeps the old import
path alive without carrying a second copy of the full command tree.
"""

from __future__ import annotations

from weiss_rl.workflows.parsers import _parse_args as parse_workflow_args
from weiss_rl.workflows.parsers import build_parser as build_workflow_parser

__all__ = ["build_workflow_parser", "parse_workflow_args"]
