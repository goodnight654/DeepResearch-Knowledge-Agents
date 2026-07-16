"""Small helpers for parsing structured LLM responses safely."""

from __future__ import annotations

import json
from typing import Any


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    """Parse a JSON object from plain text, a fenced block, or surrounding prose."""
    text = str(raw or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        first_newline = text.find("\n")
        closing_fence = text.rfind("```")
        if first_newline >= 0 and closing_fence > first_newline:
            text = text[first_newline + 1 : closing_fence].strip()

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Convert untrusted model output to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number
