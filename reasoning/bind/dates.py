"""bind layer — dates domains (see reasoning.bind)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403


# --- date families ---------------------------------------------------------------------
# date discovery: year-first numeric (-, /, .) and named-month (either order, with year).
# All unambiguous; locale-ambiguous numeric month-first (MM/DD/YYYY) is intentionally absent.
_MONTH_WORD = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_ISO_DATE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
_DATE_PATTERNS = [
    _ISO_DATE,
    re.compile(rf"\b\d{{1,2}}\s+{_MONTH_WORD}\.?,?\s+\d{{4}}\b", re.IGNORECASE),   # D Month YYYY
    re.compile(rf"\b{_MONTH_WORD}\.?\s+\d{{1,2}},?\s+\d{{4}}\b", re.IGNORECASE),   # Month D, YYYY
]


def _find_dates(query: str) -> list[str]:
    """Every date expression in the query, in order, de-overlapped (numeric or named-month)."""
    spans = sorted((m.start(), m.end(), m.group())
                   for pat in _DATE_PATTERNS for m in pat.finditer(query))
    out, last_end = [], -1
    for start, end, text in spans:
        if start >= last_end:  # skip overlapping matches from a second pattern
            out.append(text)
            last_end = end
    return out


def bind_days_between(query: str) -> tuple[bool, dict | None, str, int]:
    """Coded-only bind: two dates → {d1, d2}. No model fallback (yet)."""
    dates = _find_dates(query)
    if len(dates) == 2:
        return True, {"d1": dates[0], "d2": dates[1]}, "code", 0
    return False, None, "code", 0


# difference unit → (skill, whether to round, unit label)
_DIFF_UNITS = [
    ("week", "date.weeks_between", True, "weeks"),
    ("month", "date.months_between", False, "months"),
    ("year", "date.years_between", False, "years"),
    ("day", "date.days_between", False, "days"),
]


def bind_date_difference(query: str) -> tuple[bool, dict | None, str, int]:
    """Two dates + the unit asked for (days/weeks/months/years)."""
    dates = _find_dates(query)
    if len(dates) != 2:
        return False, None, "code", 0
    q = query.lower()
    skill, rounded, label = next(((s, r, lbl) for w, s, r, lbl in _DIFF_UNITS if w in q),
                                 ("date.days_between", False, "days"))
    return True, {"d1": dates[0], "d2": dates[1], "skill": skill,
                  "rounded": rounded, "label": label}, "code", 0


_AFTER = re.compile(r"(\d+)\s*(day|week)s?\s+after", re.IGNORECASE)


def bind_date_add(query: str) -> tuple[bool, dict | None, str, int]:
    """'<n> days/weeks after <date>' → {date, n, skill}."""
    m, dates = _AFTER.search(query), _find_dates(query)
    if m and dates:
        skill = "date.add_weeks" if m.group(2).lower() == "week" else "date.add_days"
        return True, {"date": dates[0], "n": int(m.group(1)), "skill": skill}, "code", 0
    return False, None, "code", 0


def bind_single_date(query: str) -> tuple[bool, dict | None, str, int]:
    """A single date → {date}. For weekday / day-of-year queries."""
    dates = _find_dates(query)
    if len(dates) == 1:
        return True, {"date": dates[0]}, "code", 0
    return False, None, "code", 0


_YEAR = re.compile(r"\b(\d{4})\b")


def bind_year(query: str) -> tuple[bool, dict | None, str, int]:
    """A 4-digit year → {year}. For leap-year queries."""
    m = _YEAR.search(query)
    if m:
        return True, {"year": int(m.group(1))}, "code", 0
    return False, None, "code", 0
