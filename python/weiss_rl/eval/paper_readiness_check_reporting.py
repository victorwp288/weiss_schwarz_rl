"""Console reporting helpers for the paper-readiness check CLI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def format_alarm(name: str, check: Any) -> str:
    detail = str(name)
    if isinstance(check, Mapping):
        message = check.get("message")
        reason = check.get("reason")
        if isinstance(message, str) and message.strip():
            return f"{detail} ({message.strip()})"
        if isinstance(reason, str) and reason.strip():
            return f"{detail} ({reason.strip()})"
    return detail


def format_alarm_detail(payload: Mapping[str, Any], alarm: str) -> str:
    checks = payload.get("checks")
    if isinstance(checks, Mapping) and alarm in checks:
        return format_alarm(alarm, checks.get(alarm))
    section = payload.get(alarm)
    if isinstance(section, Mapping):
        message = section.get("message")
        if isinstance(message, str) and message.strip():
            return f"{alarm} ({message.strip()})"
        reason = section.get("reason")
        if isinstance(reason, str) and reason.strip():
            return f"{alarm} ({reason.strip()})"
    guardrails = payload.get("final_eval_guardrails")
    if alarm == "final_eval_guardrails" and isinstance(guardrails, Mapping):
        message = guardrails.get("message")
        if isinstance(message, str) and message.strip():
            return f"{alarm} ({message.strip()})"
        reason = guardrails.get("reason")
        if isinstance(reason, str) and reason.strip():
            return f"{alarm} ({reason.strip()})"
    return alarm


def format_failure_message(payload: Mapping[str, Any]) -> str:
    raw_alarms = payload.get("alarms", ())
    alarms = [str(alarm) for alarm in raw_alarms] if isinstance(raw_alarms, (list, tuple)) else []
    alarm_details = ", ".join(format_alarm_detail(payload, alarm) for alarm in alarms)
    return f"Paper readiness checks failed: {alarm_details}"
