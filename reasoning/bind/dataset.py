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

    if "generate" in q or "synthesize" in q or "create a dataset" in q:
        g = _generate(query, q)
        if g:
            return ok("generate", **g)

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


def _generate(query: str, q: str):
    """Map a 'generate …' query to {generator, save_as, params}. Covers the common phrasings;
    full control is via the `data gen` CLI. Returns None if no generator keyword matches."""
    name_m = re.search(r"(?:save (?:it )?as|called|named)\s+([A-Za-z_]\w*)", q)
    n_m = re.search(r"\b(\d+)\b", q)              # first integer = count, where relevant
    n = n_m[1] if n_m else None
    num = lambda pat: (re.search(pat, q).group(1) if re.search(pat, q) else None)  # noqa: E731

    def g(gen, default_name, **params):
        params = {k: v for k, v in params.items() if v is not None}
        return {"gen": gen, "save_as": (name_m[1] if name_m else default_name), "params": params}

    if "prime" in q:
        return g("primes", "primes", n=n)
    if "fibonacci" in q:
        return g("fibonacci", "fibonacci", n=n)
    if "factorial" in q:
        return g("factorials", "factorials", n=n)
    if "partition" in q:
        return g("partitions", "partitions", n=n)
    if "normal" in q:
        return g("normal_sample", "normal_sample", n=n,
                 mean=num(r"mean\s+(-?\d+(?:\.\d+)?)"), std=num(r"(?:std|standard deviation)\s+(-?\d+(?:\.\d+)?)"))
    if "uniform" in q:
        return g("uniform_sample", "uniform_sample", n=n)
    if "coin" in q:
        return g("coin_flips", "coin_flips", n=n)
    if "dice" in q or "die" in q:
        return g("dice_rolls", "dice_rolls", n=n)
    if "logistic" in q:
        return g("logistic_trajectory", "logistic_trajectory",
                 r=num(r"r\s*=\s*(\d+(?:\.\d+)?)"), x0=num(r"x0\s*=\s*(\d+(?:\.\d+)?)"),
                 n=num(r"(\d+)\s+steps"))
    if "amortization" in q or "loan schedule" in q:
        return g("amortization", "amortization",
                 principal=num(r"\$?([\d,]+)") and num(r"\$?([\d,]+)").replace(",", ""),
                 annual_rate=num(r"(\d+(?:\.\d+)?)\s*%"), years=num(r"(\d+)\s*year"))
    if "compound" in q:
        return g("compound_growth", "compound_growth",
                 rate=num(r"(\d+(?:\.\d+)?)\s*%"), years=num(r"(\d+)\s*year"))
    if "regression" in q:
        return g("regression", "regression", n=n)
    if "classification" in q:
        return g("classification", "classification", n=n)
    if "eoq" in q:
        return g("eoq_curve", "eoq_curve", demand=num(r"demand\D+(\d+)"),
                 order_cost=num(r"order(?:ing)? cost\D+(\d+)"),
                 holding_cost=num(r"holding cost\D+(\d+)"))
    if "inventory simulation" in q or ("inventory" in q and "simulat" in q):
        return g("inventory_simulation", "inventory_simulation",
                 periods=num(r"(\d+)\s*(?:periods|weeks|months|days)"),
                 demand_rate=num(r"demand\D+(\d+)"), reorder_point=num(r"reorder point\D+(\d+)"),
                 order_qty=num(r"order(?:ing)? quantity\D+(\d+)"))
    return None
