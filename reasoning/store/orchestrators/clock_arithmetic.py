"""Clock / time-of-day arithmetic — add durations, 12h↔24h, between, duration. Zero model."""

from reasoning.bind import bind_clock
from reasoning.compiled import dispatch

META = {
    "id": "clock_arithmetic",
    "query_type": "clock_arithmetic",
    "description": "time after/before a duration, 12h↔24h conversion, minutes between, or duration",
    "example_queries": [
        "What time is it 3 hours after 2:30 PM?",
        "What time is 90 minutes after 11:15 AM?",
        "Convert 14:30 to 12-hour format.",
        "Convert 3:45 PM to 24-hour format.",
        "How many minutes are between 9:15 AM and 10:45 AM?",
        "How long is it from 8:00 AM to 5:30 PM?",
    ],
    "keywords": ["time", "hours", "minutes", "after", "before", "12-hour", "24-hour", "between"],
    "model_free": True,
}


def precondition(query):
    return bind_clock(query)[0]


def run(query):
    # NB: no residue guard here — bind_clock TRANSFORMS its input ("3 hours" → minutes=180), so a
    # raw-number residue check would mismatch the stored value and false-decline. Transform-based
    # binds need a bespoke guard, not the generic one (see §3d caveat).
    return dispatch(bind_clock(query), parse_message="could not parse the time(s) or operation")
