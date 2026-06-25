"""Dataset subsystem — larger-than-text data as inputs, scratch storage, and outputs.

Unlike the pure `skills/` library (no I/O, tiny values), this layer does file I/O and works over
datasets too large to inline. It is deliberately a SEPARATE subsystem so the skill-purity
invariant holds.

Model:
  * A dataset is referenced by a HANDLE (a name). Its file lives in the data dir
    ($DIFG_DATA_DIR, default <repo>/data) as <name>.<ext>.
  * DuckDB is the single engine: every supported format is registered as a DuckDB view named by
    its handle, so any query is plain SQL over `SELECT … FROM <handle>`.
  * Results stay small: a scalar, a compact text table (≤ MAX_INLINE_ROWS), or — when the output
    is too large — a NEW handle plus a preview (schema, row count, head). Never a data dump.
  * Everything is deterministic and model-free (0 model calls).

Supported: CSV/TSV, Parquet, JSON/JSONL, SQLite (.sqlite/.db), DuckDB (.duckdb), Arrow/Feather
(.arrow/.feather, via pyarrow), and NumPy arrays (.npy, via numpy — flattened to columns).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import duckdb

MAX_INLINE_ROWS = 20          # results larger than this are saved to a handle + previewed
PREVIEW_ROWS = 5

_FILE_FORMATS = {".csv": "csv", ".tsv": "csv", ".parquet": "parquet", ".pq": "parquet",
                 ".json": "json", ".jsonl": "json", ".ndjson": "json"}
_DB_FORMATS = {".sqlite": "sqlite", ".db": "sqlite", ".duckdb": "duckdb"}
# read via pyarrow/numpy (not DuckDB-native), then registered into the connection
_PY_FORMATS = {".arrow": "arrow", ".feather": "arrow", ".npy": "numpy"}
SUPPORTED = set(_FILE_FORMATS) | set(_DB_FORMATS) | set(_PY_FORMATS)


def data_dir() -> Path:
    d = Path(os.environ.get("DIFG_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _q(s) -> str:
    """Escape a string for a SQL single-quoted literal."""
    return str(s).replace("'", "''")


def _path(name: str) -> Path | None:
    """The file backing a handle, or None. Accepts a bare name or a name.ext."""
    d = data_dir()
    p = d / name
    if p.suffix.lower() in SUPPORTED and p.exists():
        return p
    for ext in SUPPORTED:
        if (d / f"{name}{ext}").exists():
            return d / f"{name}{ext}"
    return None


def list_datasets() -> list[str]:
    return sorted(p.stem for p in data_dir().iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED)


def _reference_tables() -> set[str]:
    try:
        import sqlite3

        from reasoning import reference
        con = sqlite3.connect(reference.ensure_db())
        try:
            return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()
    except (ImportError, OSError):
        return set()


def _exists(name: str) -> bool:
    """A handle exists if it's a file in the data dir or a bundled reference table."""
    return _path(name) is not None or name in _reference_tables()


def _read_sql(path: Path) -> str:
    """A DuckDB FROM-source expression for a file-format dataset."""
    fmt = _FILE_FORMATS[path.suffix.lower()]
    lit = _q(path)
    if fmt == "csv":
        return f"read_csv_auto('{lit}')"
    if fmt == "parquet":
        return f"read_parquet('{lit}')"
    return f"read_json_auto('{lit}')"


def _py_table(path: Path):
    """Load an Arrow/Feather or .npy file into a pyarrow Table (a .npy array becomes columns
    c0..cN, or a single 'value' column when 1-D)."""
    if _PY_FORMATS[path.suffix.lower()] == "arrow":
        import pyarrow.feather as feather
        return feather.read_table(str(path))
    import numpy as np
    import pyarrow as pa
    a = np.load(path, allow_pickle=False)
    if a.ndim == 1:
        return pa.table({"value": a})
    return pa.table({f"c{i}": a[:, i] for i in range(a.shape[1])})


def _register_reference(con: duckdb.DuckDBPyConnection) -> None:
    """Attach the bundled read-only reference DB (countries/population/cities) as views, so
    constant reference data is queryable as handles. User datasets of the same name shadow it
    (they are registered afterwards with CREATE OR REPLACE)."""
    try:
        from reasoning import reference
        con.execute(f"ATTACH IF NOT EXISTS '{_q(reference.ensure_db())}' AS _ref "
                    f"(TYPE sqlite, READ_ONLY)")
        tables = con.execute("SELECT table_name FROM information_schema.tables "
                             "WHERE table_catalog = '_ref'").fetchall()
        for (t,) in tables:
            con.execute(f'CREATE OR REPLACE VIEW "{t}" AS SELECT * FROM "_ref"."{t}"')
    except (duckdb.Error, ImportError, OSError):
        pass


def _register(con: duckdb.DuckDBPyConnection) -> None:
    """Register every dataset in the data dir as a view (or attached DB) named by its handle."""
    _register_reference(con)   # bundled reference first; user data dir can shadow it
    for path in data_dir().iterdir():
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        name = path.stem
        try:
            if ext in _FILE_FORMATS:
                con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM {_read_sql(path)}')
            elif ext in _PY_FORMATS:
                con.register(name, _py_table(path))
            elif ext in _DB_FORMATS:
                kind = _DB_FORMATS[ext]
                alias = f"_db_{name}"
                typ = " (TYPE sqlite)" if kind == "sqlite" else ""
                con.execute(f"ATTACH IF NOT EXISTS '{_q(path)}' AS \"{alias}\"{typ}")
                tables = con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_catalog = ?",
                    [alias]).fetchall()
                for (t,) in tables:
                    # first table is the handle itself; every table is also name__table
                    con.execute(f'CREATE OR REPLACE VIEW "{name}__{t}" AS '
                                f'SELECT * FROM "{alias}"."{t}"')
                if tables:
                    con.execute(f'CREATE OR REPLACE VIEW "{name}" AS '
                                f'SELECT * FROM "{alias}"."{tables[0][0]}"')
        except duckdb.Error:
            continue  # a malformed file shouldn't break the whole registry


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    _register(con)
    return con


def _fmt_scalar(v) -> str:
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else str(round(v, 6))
    return str(v)


def _fmt_table(cols, rows) -> str:
    head = " | ".join(cols)
    body = "\n".join(" | ".join(_fmt_scalar(c) for c in r) for r in rows)
    return f"{head}\n{body}" if body else head


def _arrow_table(con, sql: str):
    rel = con.sql(sql)
    return rel.to_arrow_table() if hasattr(rel, "to_arrow_table") else rel.fetch_arrow_table()


def _write(con, sql: str, name: str, fmt: str = "parquet") -> Path:
    if fmt == "arrow":                           # via pyarrow (DuckDB COPY has no Arrow format)
        import pyarrow.feather as feather
        out = data_dir() / f"{name}.arrow"
        feather.write_feather(_arrow_table(con, sql), str(out))
        return out
    if fmt == "npy":                             # numeric matrix only
        import numpy as np
        out = data_dir() / f"{name}.npy"
        np.save(out, np.asarray(_arrow_table(con, sql).to_pandas().to_numpy(), dtype=float))
        return out
    out = data_dir() / f"{name}.{fmt}"
    copy_fmt = {"parquet": "(FORMAT parquet)", "csv": "(FORMAT csv, HEADER)",
                "json": "(FORMAT json)"}[fmt]
    con.execute(f"COPY ({sql}) TO '{_q(out)}' {copy_fmt}")
    return out


def _preview(con, name: str, sql: str, n: int, cols) -> str:
    rows = con.sql(f"SELECT * FROM ({sql}) LIMIT {PREVIEW_ROWS}").fetchall()
    return (f"→ saved {n:,} rows × {len(cols)} cols to dataset '{name}'. "
            f"columns: {', '.join(cols)}\nhead:\n{_fmt_table(cols, rows)}")


# --- public operations -----------------------------------------------------

def import_file(src: str, name: str | None = None) -> str:
    src_p = Path(src).expanduser()
    if not src_p.exists():
        raise FileNotFoundError(src)
    ext = src_p.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"unsupported format {ext} (supported: {sorted(SUPPORTED)})")
    handle = name or src_p.stem
    dest = data_dir() / f"{handle}{ext}"
    shutil.copyfile(src_p, dest)
    return describe(handle)


def schema(name: str) -> list[tuple[str, str]]:
    """[(column, duckdb_type)] for a dataset/reference handle — for programmatic introspection."""
    if not _exists(name):
        raise FileNotFoundError(f"no dataset '{name}' (have: {list_datasets()})")
    con = _con()
    return [(c[0], c[1]) for c in con.execute(f'DESCRIBE SELECT * FROM "{name}"').fetchall()]


def describe(name: str) -> str:
    if not _exists(name):
        raise FileNotFoundError(f"no dataset '{name}' (have: {list_datasets()})")
    con = _con()
    schema = con.execute(f'DESCRIBE SELECT * FROM "{name}"').fetchall()
    n = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
    colspec = ", ".join(f"{c[0]} ({c[1]})" for c in schema)
    head = con.sql(f'SELECT * FROM "{name}" LIMIT {PREVIEW_ROWS}').fetchall()
    cols = [c[0] for c in schema]
    return (f"dataset '{name}' — {n:,} rows, {len(schema)} columns: {colspec}\n"
            f"head:\n{_fmt_table(cols, head)}")


def query(sql: str, *, save_as: str | None = None) -> str:
    """Run SQL over the registered datasets. Returns a scalar / small table inline, or — when the
    result is large (or save_as is given) — saves a new dataset and returns a preview."""
    con = _con()
    cols = con.sql(sql).columns
    n = con.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
    if save_as:
        _write(con, sql, save_as)
        return _preview(con, save_as, sql, n, cols)
    if n == 1 and len(cols) == 1:
        return _fmt_scalar(con.sql(sql).fetchone()[0])
    if n <= MAX_INLINE_ROWS:
        return _fmt_table(cols, con.sql(sql).fetchall())
    auto = f"result_{abs(hash(sql)) % 100000}"
    _write(con, sql, auto)
    return _preview(con, auto, sql, n, cols)


def aggregate(name: str, column: str, op: str, where: str | None = None) -> str:
    if not _exists(name):
        raise FileNotFoundError(f"no dataset '{name}'")
    op = {"average": "avg", "mean": "avg", "total": "sum", "maximum": "max",
          "minimum": "min", "count": "count", "stddev": "stddev"}.get(op.lower(), op.lower())
    clause = f" WHERE {where}" if where else ""
    return query(f'SELECT {op}("{column}") FROM "{name}"{clause}')


def count_rows(name: str, where: str | None = None) -> str:
    if not _exists(name):
        raise FileNotFoundError(f"no dataset '{name}'")
    clause = f" WHERE {where}" if where else ""
    return query(f'SELECT count(*) FROM "{name}"{clause}')


def convert(name: str, out_name: str, fmt: str) -> str:
    if not _exists(name):
        raise FileNotFoundError(f"no dataset '{name}'")
    con = _con()
    _write(con, f'SELECT * FROM "{name}"', out_name, fmt=fmt)
    return describe(out_name)


def save_rows(name: str, columns: list[str], rows) -> str:
    """Write Python rows (an iterable of tuples) to a dataset (parquet) and return its
    description. Used by the domain generators in datagen.py for stateful/recursive series."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = list(rows)
    cols = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
    pq.write_table(pa.table(cols), data_dir() / f"{name}.parquet")
    return describe(name)


def generate(n: int, save_as: str, select: str | None = None) -> str:
    """Generate a dataset of n rows. Default columns: id (1..n) and a uniform random value;
    pass a custom `select` (a SQL projection over `range(n) t(i)`) for other shapes."""
    proj = select or "i + 1 AS id, round(random(), 6) AS value"
    return query(f"SELECT {proj} FROM range({int(n)}) t(i)", save_as=save_as)
