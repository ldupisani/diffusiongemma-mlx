"""Reference data access — bundled, constant, verifiable datasets (currently geography).

The curated seed (store/reference/seed.py) is built lazily into a SQLite database the first time
it's needed; that DB is auto-registered read-only into the dataset DuckDB engine (see
datasets._register), so reference tables (countries, population, cities) are queryable as handles
everywhere. This module adds direct lookups and an import path to extend/replace the seed with an
authoritative source. Reads a DB (I/O), so it's a layer of its own, not a pure skill.
"""

from __future__ import annotations

import os
import sqlite3
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
