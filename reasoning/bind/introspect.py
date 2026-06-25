"""bind layer — self-introspection meta-queries ("what can you do?", "what can I ask about
geometry?"). Conservative: only fires on an explicit capability question with no digits (real
domain queries carry numbers), so it never steals a computational query."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_VERB = re.compile(
    r"what can (?:you|i|the system|this|it) (?:do|ask|answer)|"
    r"how (?:do|can) i (?:query|ask|use)|"
    r"what (?:kinds?|types?) of (?:queries|questions|problems)|"
    r"(?:list|show)(?: me)?(?: the| your)? (?:domains|topics|capabilities)|"
    r"what (?:domains|topics|capabilities)|your capabilities|what are you capable", re.IGNORECASE)

_ABOUT = re.compile(r"\b(?:about|with|regarding|on|for|query|querying|use)\s+(?:the\s+)?"
                    r"([a-zA-Z][a-zA-Z _\-]{1,38}?)(?:\s+domain)?\s*[?.!]*$", re.IGNORECASE)


def bind_introspect(query: str) -> tuple[bool, dict | None, str, int]:
    def ok(op, **data):
        return True, {"op": op, **data}, "code", 0

    if re.search(r"\d", query):            # meta-questions aren't numeric → leave computation alone
        return False, None, "code", 0
    if not _VERB.search(query):
        return False, None, "code", 0
    m = _ABOUT.search(query)
    if m:
        topic = m[1].strip().lower()
        if topic and topic not in ("you", "me", "it", "this", "the system"):
            return ok("about", topic=topic)
    return ok("overview")
