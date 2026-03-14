"""Helpers for robust JSON extraction from model outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    text = strip_code_fences(raw)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {"result": payload}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {"result": payload}
        except json.JSONDecodeError:
            return None

