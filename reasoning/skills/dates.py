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


@register("date.parse",
          "parse a date: year-first numeric (YYYY-MM-DD, /, .) or named-month "
          "('Month D [Year]' or 'D Month Year')",
          [(("2024-07-15",), date(2024, 7, 15)), (("2900/10/10",), date(2900, 10, 10)),
           (("March 3 2024",), date(2024, 3, 3)), (("10 Feb 2020",), date(2020, 2, 10))])
def parse(s: str, default_year: int = 2024) -> date:
    s = s.strip()

    def _build(y, mo, da):
        try:
            return date(y, mo, da)
        except ValueError as e:
            raise ValueError(f"invalid date {s!r}: {e}") from e

    # year-first numeric: separators -, /, . are unambiguous (month/day-first is not)
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return _build(int(m[1]), int(m[2]), int(m[3]))
    # "Month D[, Year]" — named month first (month word removes MM/DD ambiguity)
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?", s)
    if m and m[1].lower() in _MONTHS:
        return _build(int(m[3]) if m[3] else default_year, _MONTHS[m[1].lower()], int(m[2]))
    # "D Month Year" — day first, named month
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})", s)
    if m and m[2].lower() in _MONTHS:
        return _build(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))
    raise ValueError(f"unrecognized date: {s!r}")


@register("date.days_between", "number of days between two dates (order-independent)",
          [((date(2024, 3, 3), date(2024, 7, 15)), 134),
           ((date(2024, 7, 15), date(2024, 3, 3)), 134),   # reversed → same magnitude
           ((date(2024, 2, 28), date(2024, 3, 1)), 2)])  # 2024 is a leap year
def days_between(d1: date, d2: date) -> int:
    return abs((d2 - d1).days)


@register("date.add_days", "the date n days after a date",
          [((date(2024, 1, 1), 30), date(2024, 1, 31))])
def add_days(d: date, n: int) -> date:
    return d + timedelta(days=n)


@register("date.weekday_name", "the weekday name of a date",
          [((date(2024, 7, 15),), "Monday")])
def weekday_name(d: date) -> str:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"][d.weekday()]


@register("date.add_weeks", "the date n weeks after a date",
          [((date(2024, 6, 1), 12), date(2024, 8, 24))])
def add_weeks(d: date, n: int) -> date:
    return d + timedelta(weeks=n)


@register("date.weeks_between", "number of weeks between two dates, order-independent (fractional)",
          [((date(2024, 1, 1), date(2024, 12, 31)), 52.142857142857146)])
def weeks_between(d1: date, d2: date) -> float:
    return abs((d2 - d1).days) / 7


@register("date.months_between", "whole calendar months between two dates (order-independent)",
          [((date(2023, 2, 15), date(2024, 8, 15)), 18),
           ((date(2040, 2, 15), date(2029, 8, 15)), 126),  # reversed → magnitude
           ((date(2024, 1, 31), date(2024, 3, 1)), 1)])
def months_between(d1: date, d2: date) -> int:
    lo, hi = sorted((d1, d2))
    return (hi.year - lo.year) * 12 + (hi.month - lo.month) - (1 if hi.day < lo.day else 0)


@register("date.years_between", "whole years between two dates (order-independent)",
          [((date(2000, 5, 20), date(2024, 5, 20)), 24), ((date(2000, 5, 20), date(2024, 5, 19)), 23)])
def years_between(d1: date, d2: date) -> int:
    lo, hi = sorted((d1, d2))
    return hi.year - lo.year - ((hi.month, hi.day) < (lo.month, lo.day))


@register("date.is_leap_year", "whether a year is a leap year",
          [((2024,), True), ((1900,), False), ((2000,), True)])
def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


@register("date.day_of_year", "the ordinal day of the year for a date (Jan 1 = 1)",
          [((date(2024, 3, 1),), 61), ((date(2024, 1, 1),), 1)])
def day_of_year(d: date) -> int:
    return d.timetuple().tm_yday
