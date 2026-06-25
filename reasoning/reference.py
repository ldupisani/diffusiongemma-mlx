"""Reference data access — bundled, constant, verifiable datasets (currently geography).

The curated seed (store/reference/seed.py) is built lazily into a SQLite database the first time
it's needed; that DB is auto-registered read-only into the dataset DuckDB engine (see
datasets._register), so reference tables (countries, population, cities) are queryable as handles
everywhere. This module adds direct lookups and an import path to extend/replace the seed with an
authoritative source. Reads a DB (I/O), so it's a layer of its own, not a pure skill.
"""

from __future__ import annotations

import csv
import io
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .store.reference import seed

DB = "geography"


def _dir() -> Path:
    """Where the built reference DB(s) live. $DIFG_REFERENCE_DIR overrides the bundled location
    (the built .sqlite is a generated artifact; seed.py is the version-controlled source)."""
    d = Path(os.environ.get("DIFG_REFERENCE_DIR",
                            Path(__file__).resolve().parent / "store" / "reference"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path(name: str = DB) -> Path:
    return _dir() / f"{name}.sqlite"


def _sqltype(v) -> str:
    return "INTEGER" if isinstance(v, int) and not isinstance(v, bool) else (
        "REAL" if isinstance(v, float) else "TEXT")


def build_db(path: Path) -> None:
    con = sqlite3.connect(path)
    for table, (cols, rows) in seed.TABLES.items():
        types = [_sqltype(rows[0][i]) for i in range(len(cols))]
        coldef = ", ".join(f'"{c}" {t}' for c, t in zip(cols, types))
        con.execute(f'CREATE TABLE "{table}" ({coldef})')
        con.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" * len(cols))})', rows)
    con.commit()
    con.close()


def ensure_db() -> Path:
    p = db_path()
    if not p.exists():
        build_db(p)
    con = sqlite3.connect(p)          # the retrieved-values log (model-sourced; kept separate)
    con.execute("CREATE TABLE IF NOT EXISTS retrieved "
                "(op TEXT, place TEXT, year INTEGER, value TEXT, source TEXT, retrieved_at TEXT)")
    con.commit()
    con.close()
    return p


def _con() -> sqlite3.Connection:
    return sqlite3.connect(ensure_db())


def _one(sql: str, params=()):
    con = _con()
    try:
        row = con.execute(sql, params).fetchone()
        return row[0] if row and len(row) == 1 else row
    finally:
        con.close()


# --- lookups ---------------------------------------------------------------

def country_population(place: str, year: int | None = None):
    if year is not None:
        return _one("SELECT population FROM population WHERE place=? AND year=?", (place, int(year)))
    return _one("SELECT population FROM population WHERE place=? ORDER BY year DESC LIMIT 1", (place,))


def city_population(name: str):
    return _one("SELECT population FROM cities WHERE name=?", (name,))


def coordinates(place: str):
    """(lat, lon) for a city by name, else for a country (its capital). None if unknown."""
    row = _one("SELECT lat, lon FROM cities WHERE name=?", (place,))
    if row:
        return row
    return _one("SELECT lat, lon FROM countries WHERE name=?", (place,))


def country_field(name: str, field: str):
    if field not in {"capital", "continent", "area_km2", "iso2"}:
        raise ValueError(f"unknown country field {field!r}")
    return _one(f'SELECT "{field}" FROM countries WHERE name=?', (name,))


def population(place: str, year: int | None = None):
    """Population of a country (by year) or, failing that, a city (single figure)."""
    v = country_population(place, year)
    return v if v is not None else city_population(place)


def place_names() -> dict:
    """lowercased name -> canonical name, for every country and city (for NL place matching)."""
    con = _con()
    try:
        names = {n.lower(): n for (n,) in con.execute("SELECT name FROM countries")}
        for (n,) in con.execute("SELECT name FROM cities"):
            names.setdefault(n.lower(), n)
        return names
    finally:
        con.close()


def tables(name: str = DB) -> list[str]:
    con = sqlite3.connect(ensure_db() if name == DB else db_path(name))
    try:
        return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    finally:
        con.close()


# --- model retrieval fallback (for data points NOT in the curated set) -----

def retrieval_enabled() -> bool:
    """Off by default (keeps the engine 0-model-call). Enable with DIFG_REFERENCE_RETRIEVAL=1."""
    return os.environ.get("DIFG_REFERENCE_RETRIEVAL", "").lower() in {"1", "true", "on", "claude", "sonnet"}


_PROMPTS = {
    "population": "What was the population of {place}{when}? Respond with ONLY the population as a "
                  "plain integer — digits only, no commas, words, or explanation.",
    "coordinates": "What are the latitude and longitude of {place}? Respond with ONLY "
                   "'<lat>, <lon>' in decimal degrees and nothing else.",
    "capital": "What is the capital city of {place}? Respond with ONLY the city name.",
    "continent": "Which continent is {place} on? Respond with ONLY the continent name.",
    "area": "What is the total area of {place} in square kilometres? Respond with ONLY the "
            "number — digits only, no commas or units.",
}


def _validate(op: str, raw: str) -> str | None:
    """Extract and range-check the value (the 'verifiable' step). None if it doesn't pass."""
    raw = raw.strip()
    if op in ("population", "area"):
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", raw)
        if not m:
            return None
        v = float(m.group().replace(",", ""))
        low = raw.lower()                                  # handle "3.0 million" style answers
        v *= 1e9 if "billion" in low else 1e6 if "million" in low else 1e3 if "thousand" in low else 1
        return str(int(v)) if v > 0 else None
    if op == "coordinates":
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
        if len(nums) < 2:
            return None
        lat, lon = float(nums[0]), float(nums[1])
        return f"{lat}, {lon}" if -90 <= lat <= 90 and -180 <= lon <= 180 else None
    line = raw.splitlines()[0].strip().strip(".")          # capital / continent: a short name
    return line if re.fullmatch(r"[A-Za-z .'\-]{2,40}", line) else None


def cached_value(op: str, place: str, year: int | None = None):
    """A value previously retrieved (and persisted) for this exact point, or None. Lets a repeat
    query reuse the model-sourced answer with 0 model calls."""
    where = "op=? AND place=? AND " + ("year IS NULL" if year is None else "year=?")
    params = (op, place) if year is None else (op, place, int(year))
    return _one(f"SELECT value FROM retrieved WHERE {where} ORDER BY retrieved_at DESC LIMIT 1",
                params)


def _record(op: str, place: str, year, value: str, source: str) -> None:
    con = _con()
    try:
        con.execute("INSERT INTO retrieved (op, place, year, value, source, retrieved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (op, place, None if year is None else int(year), value, source,
                     datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.commit()
    finally:
        con.close()


def retrieve(op: str, place: str, year: int | None = None, *, asker=None, model: str = "sonnet"):
    """Attempt to retrieve a missing data point via a model (Claude Sonnet by default), returning
    a validated value string or None. On success the value is PERSISTED to the `retrieved` log.
    `asker(prompt) -> str|None` is injectable for testing."""
    if op not in _PROMPTS:
        return None
    if asker is None:
        from . import teachers
        asker = lambda p: teachers.ask_claude(p, model=model)  # noqa: E731
    when = f" in {int(year)}" if year and op == "population" else ""
    raw = asker(_PROMPTS[op].format(place=place, when=when))
    val = _validate(op, raw) if raw else None
    if val is not None:
        _record(op, place, year, val, f"claude:{model}")
    return val


_SET_PROMPT = (
    "List the data for the request below as CSV: a header row, then one row per item. Use "
    "lowercase column names; always include 'name', and add population / latitude / longitude "
    "columns where they apply. Output ONLY the CSV — no prose, no markdown, no code fences.\n\n"
    "Request: {req}")


def _csv_rows(raw: str) -> list[list[str]]:
    text = re.sub(r"```[a-zA-Z]*", "", raw).strip()        # strip any code fences
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        return []
    ncol = len(rows[0])                                    # keep only rows matching the header width
    return [r for r in rows if len(r) == ncol]


def retrieve_set(request: str, save_as: str, *, asker=None, model: str = "sonnet"):
    """Batch retrieval: ask a model (Claude Sonnet) for a CSV table answering an open-ended or
    list request ('all cities in France', 'populations of France, Germany and Italy'), save it as
    a dataset handle, and return the number of data rows (None on failure). The saved dataset is
    the persisted, queryable, model-sourced artifact."""
    if asker is None:
        from . import teachers
        asker = lambda p: teachers.ask_claude(p, model=model)  # noqa: E731
    raw = asker(_SET_PROMPT.format(req=request))
    if not raw:
        return None
    rows = _csv_rows(raw)
    if len(rows) < 2 or not rows[0]:                       # need a header + ≥1 data row
        return None
    from . import datasets
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    (datasets.data_dir() / f"{save_as}.csv").write_text(buf.getvalue())
    return len(rows) - 1


def import_csv(src: str, table: str, name: str = DB) -> str:
    """Load a CSV into a reference table (the path to authoritative data). Replaces the table."""
    import csv
    src_p = Path(src).expanduser()
    with open(src_p, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    con = sqlite3.connect(ensure_db() if name == DB else db_path(name))
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(f'CREATE TABLE "{table}" ({", ".join(chr(34) + c + chr(34) for c in header)})')
    con.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" * len(header))})', rows)
    con.commit()
    con.close()
    return f"imported {len(rows)} rows into reference table '{table}'"
