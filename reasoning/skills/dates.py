"""Date skills — stdlib datetime only, no heavy deps.

parse() accepts ISO (YYYY-MM-DD) plus a couple of common English forms ("March 3",
"July 15 2024"); like the bind layer it is conservative — it raises on anything it does
not confidently understand, so a caller can fall back to the model rather than guess.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from .base import register

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
_MONTHS.update({m[:3].lower(): i for m, i in list(_MONTHS.items())})


@register("date.parse", "parse a date string (ISO YYYY-MM-DD or 'Month D [Year]') to a date",
          [(("2024-07-15",), date(2024, 7, 15)), (("March 3 2024",), date(2024, 3, 3))])
def parse(s: str, default_year: int = 2024) -> date:
    s = s.strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?", s)
    if m and m[1].lower() in _MONTHS:
        return date(int(m[3]) if m[3] else default_year, _MONTHS[m[1].lower()], int(m[2]))
    raise ValueError(f"unrecognized date: {s!r}")


@register("date.days_between", "number of days between two dates (d2 - d1)",
          [((date(2024, 3, 3), date(2024, 7, 15)), 134),
           ((date(2024, 2, 28), date(2024, 3, 1)), 2)])  # 2024 is a leap year
def days_between(d1: date, d2: date) -> int:
    return (d2 - d1).days


@register("date.add_days", "the date n days after a date",
          [((date(2024, 1, 1), 30), date(2024, 1, 31))])
def add_days(d: date, n: int) -> date:
    return d + timedelta(days=n)


@register("date.weekday_name", "the weekday name of a date",
          [((date(2024, 7, 15),), "Monday")])
def weekday_name(d: date) -> str:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"][d.weekday()]
