"""bind layer — the dataset subsystem. Recognizes natural-language analytics over a named
dataset (a handle in $DIFG_DATA_DIR): describe, row count, column aggregate, or raw SQL.

Gated on an explicit dataset reference ("… in/of/from the dataset <name>", "in <name>", a raw
"SQL … on <name>"), so plain math/stats queries (which have no dataset) are never claimed.
Unlike other binds this returns {"op", ...} consumed by the dataset orchestrator's own run()
(the dataset layer does I/O, so it is not a pure skill behind `dispatch`)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_AGG = {"average": "average", "mean": "average", "sum": "sum", "total": "sum",
        "maximum": "max", "max": "max", "minimum": "min", "min": "min",
        "median": "median", "count": "count", "standard deviation": "stddev"}
# the dataset name after a referencing preposition
_IN = re.compile(r"(?:in|of|from|on)\s+(?:the\s+)?(?:dataset\s+|table\s+)?([A-Za-z_]\w*)", re.IGNORECASE)


def _dataset(query: str):
    matches = _IN.findall(query)
    if matches:
        return matches[-1]   # last match wins ("average of the price column in sales")
    m = re.search(r"(?:dataset|table)\s+([A-Za-z_]\w*)", query, re.IGNORECASE)
    return m[1] if m else None


def bind_dataset(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(op, **data):
        return True, {"op": op, **data}, "code", 0

    # raw SQL passthrough: "run the SQL: SELECT …" / "SQL query SELECT …"
    sqlm = re.search(r"\b(select\b.+)", query, re.IGNORECASE | re.DOTALL)
    if ("sql" in q or "query" in q) and sqlm:
        return ok("query", sql=sqlm[1].strip().rstrip(" .?;"))

    name = _dataset(query)
    if not name:
        return False, None, "code", 0

    if "describe" in q or "schema of" in q or "columns of" in q or "what columns" in q:
        return ok("describe", name=name)

    if "how many rows" in q or "row count" in q or "number of rows" in q or "count the rows" in q:
        where = _where(query)
        return ok("count", name=name, where=where)

    # column aggregate: "average of the price column in sales"
    agg = next((v for k, v in _AGG.items() if k in q), None)
    if agg:
        cm = re.search(r"(?:of|the)\s+(?:the\s+)?([A-Za-z_]\w*)\s+column", query, re.IGNORECASE) \
            or re.search(r"column\s+([A-Za-z_]\w*)", query, re.IGNORECASE) \
            or re.search(rf"{agg}\s+(?:of\s+)?(?:the\s+)?([A-Za-z_]\w*)", q)
        if cm and cm[1].lower() not in ("dataset", "table", "column"):
            return ok("aggregate", name=name, column=cm[1], agg=agg, where=_where(query))
    return False, None, "code", 0


def _where(query: str):
    m = re.search(r"where\s+(.+?)\s*[?.]?$", query, re.IGNORECASE)
    return m[1].strip() if m else None
