"""Dataset analytics — describe / count / aggregate / SQL over a named dataset (a handle in
$DIFG_DATA_DIR). Delegates to reasoning.datasets (the I/O layer). Coded, 0 model calls."""
from reasoning import datasets as ds
from reasoning.bind import bind_dataset
from reasoning.compiled import explain
from reasoning.patterns import PatternResult


META = {
    "id": "dataset", "query_type": "dataset",
    "description": "analytics over a named dataset (CSV/Parquet/JSON/SQLite/DuckDB handle in the "
                   "data dir): describe it, count rows, aggregate a column, or run SQL",
    "example_queries": [
        "What is the average of the price column in sales?",
        "How many rows are in the orders dataset?",
        "Describe the dataset sales.",
        "Run the SQL SELECT status, count(*) FROM sales GROUP BY status.",
    ],
    "keywords": ["dataset", "column", "rows in", "describe the dataset", "sql"],
    "model_free": True,
}


def precondition(query):
    return bind_dataset(query)[0]


def run(query):
    ok, data, _, _ = bind_dataset(query)
    if not ok:
        return explain("could not parse the dataset query")
    op = data["op"]
    try:
        if op == "describe":
            ans = ds.describe(data["name"])
        elif op == "count":
            ans = ds.count_rows(data["name"], where=data.get("where"))
        elif op == "aggregate":
            ans = ds.aggregate(data["name"], data["column"], data["agg"], where=data.get("where"))
        elif op == "query":
            ans = ds.query(data["sql"])
        else:
            return explain(f"unknown dataset op {op!r}")
    except (FileNotFoundError, Exception) as e:  # missing handle / bad SQL → coded decline
        return explain(f"dataset op failed: {type(e).__name__}: {e}")
    return PatternResult(answer=ans,
                         trace=[{"step": f"dataset.{op}", "via": "code", "output": ans[:200]},
                                {"step": "summary", "model_calls": 0}])
