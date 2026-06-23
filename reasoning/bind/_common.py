"""Shared parsing primitives + cross-cutting regexes for the bind layer."""
from __future__ import annotations

import json as _json  # noqa: F401  (used by _try_json)
import re

from reasoning.bind.entities import _NUMBER  # noqa: F401  (single source; re-exported)
from reasoning.engine import one_shot  # noqa: F401  (re-exported: finance default arg)

__all__ = ["_parse_list", "_parse_int", "_try_json", "_NUMBER", "one_shot"]

# --- shared deterministic parsing primitives (moved from patterns.py) ----------------
def _parse_list(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s*(.+?)\s*$", line)
        if m:
            items.append(m.group(1))
    return items


def _parse_int(text: str, default: int) -> int:
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else default


def _try_json(text: str):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return _json.loads(t), None
    except Exception as e:
        m = re.search(r"(\{.*\}|\[.*\])", t, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(1)), None
            except Exception as e2:
                return None, str(e2)
        return None, str(e)
