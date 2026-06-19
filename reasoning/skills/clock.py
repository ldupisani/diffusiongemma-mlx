"""Clock / time-of-day skills — add durations, 12h↔24h, minutes between, duration.

Times are parsed to minutes-since-midnight (handling "2:30 PM", "14:30", "00:00") and
formatted back; all operations are modular arithmetic on that integer.
"""

from __future__ import annotations

import re

from .base import register

_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", re.IGNORECASE)


def _parse(s: str) -> int:
    m = _TIME.search(s)
    if not m:
        raise ValueError(f"could not parse a time from {s!r}")
    h, mn, ap = int(m[1]), int(m[2]), (m[3] or "").upper()
    if ap == "AM":
        h = 0 if h == 12 else h
    elif ap == "PM":
        h = 12 if h == 12 else h + 12
    return (h * 60 + mn) % 1440


def _fmt12(total: int) -> str:
    total %= 1440
    h, mn = divmod(total, 60)
    ap = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{mn:02d} {ap}"


def _fmt24(total: int) -> str:
    total %= 1440
    return f"{total // 60:02d}:{total % 60:02d}"


@register("clock.add_minutes", "the clock time a number of minutes after a given time (12h)",
          [(("2:30 PM", 180), "5:30 PM"), (("11:30 PM", 45), "12:15 AM")])
def add_minutes(time: str, minutes: int) -> str:
    return _fmt12(_parse(time) + int(minutes))


@register("clock.to_24h", "convert a 12-hour time to 24-hour format",
          [(("3:45 PM",), "15:45"), (("6:00 PM",), "18:00")])
def to_24h(time: str) -> str:
    return _fmt24(_parse(time))


@register("clock.to_12h", "convert a 24-hour time to 12-hour format",
          [(("14:30",), "2:30 PM"), (("00:00",), "12:00 AM")])
def to_12h(time: str) -> str:
    return _fmt12(_parse(time))


@register("clock.minutes_between", "whole minutes between two times",
          [(("9:15 AM", "10:45 AM"), 90)])
def minutes_between(t1: str, t2: str) -> int:
    return abs(_parse(t2) - _parse(t1))


@register("clock.duration", "duration between two times as 'H hours and M minutes'",
          [(("8:00 AM", "5:30 PM"), "9 hours and 30 minutes")])
def duration(t1: str, t2: str) -> str:
    h, m = divmod(abs(_parse(t2) - _parse(t1)), 60)
    return f"{h} hours and {m} minutes"
