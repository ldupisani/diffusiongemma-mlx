"""Dataset subsystem — import/describe/query/aggregate/convert/generate over the supported
formats (CSV, JSON/JSONL, Parquet, SQLite, DuckDB). Uses a temp data dir; no model calls."""

import importlib
import os
import sqlite3
import tempfile
import unittest

duckdb = importlib.util.find_spec("duckdb")


@unittest.skipUnless(duckdb, "duckdb not installed")
class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DIFG_DATA_DIR"] = self.tmp.name
        from reasoning import datasets
        self.ds = datasets
        # a CSV dataset
        with open(os.path.join(self.tmp.name, "sales.csv"), "w") as f:
            f.write("id,price,status\n1,10,active\n2,20,active\n3,30,closed\n")
        # a JSONL dataset
        with open(os.path.join(self.tmp.name, "items.jsonl"), "w") as f:
            f.write('{"sku": "a", "qty": 5}\n{"sku": "b", "qty": 7}\n')
        # a SQLite dataset
        con = sqlite3.connect(os.path.join(self.tmp.name, "people.db"))
        con.execute("CREATE TABLE people (name TEXT, age INTEGER)")
        con.executemany("INSERT INTO people VALUES (?, ?)", [("Ada", 36), ("Alan", 41)])
        con.commit()
        con.close()

    def tearDown(self):
        os.environ.pop("DIFG_DATA_DIR", None)
        self.tmp.cleanup()

    def test_list_and_describe(self):
        self.assertEqual(set(self.ds.list_datasets()), {"sales", "items", "people"})
        d = self.ds.describe("sales")
        self.assertIn("3 rows", d)
        self.assertIn("price", d)

    def test_aggregate_and_count(self):
        self.assertEqual(self.ds.aggregate("sales", "price", "average"), "20")
        self.assertEqual(self.ds.aggregate("sales", "price", "sum"), "60")
        self.assertEqual(self.ds.count_rows("sales"), "3")
        self.assertEqual(self.ds.count_rows("sales", where="status = 'active'"), "2")

    def test_query_scalar_and_table(self):
        self.assertEqual(self.ds.query('SELECT max(price) FROM sales'), "30")
        out = self.ds.query('SELECT status, count(*) AS n FROM sales GROUP BY status ORDER BY status')
        self.assertIn("active", out)
        self.assertIn("closed", out)

    def test_jsonl_and_sqlite_sources(self):
        self.assertEqual(self.ds.aggregate("items", "qty", "sum"), "12")
        self.assertEqual(self.ds.count_rows("people"), "2")
        self.assertEqual(self.ds.query('SELECT max(age) FROM people'), "41")

    def test_large_result_is_saved_with_preview(self):
        out = self.ds.query("SELECT i AS n FROM range(100) t(i)")
        self.assertIn("saved 100 rows", out)
        self.assertIn("head:", out)
        self.assertIn("result_", "".join(self.ds.list_datasets()) + out)

    def test_generate_and_convert(self):
        msg = self.ds.generate(50, "gen")
        self.assertIn("saved 50 rows", msg)
        self.assertIn("gen", self.ds.list_datasets())
        self.assertEqual(self.ds.count_rows("gen"), "50")
        self.ds.convert("sales", "sales_pq", "parquet")
        self.assertIn("sales_pq", self.ds.list_datasets())
        self.assertEqual(self.ds.aggregate("sales_pq", "price", "average"), "20")

    def test_nl_routing_through_the_engine(self):
        from reasoning import router

        def route(q):
            m = router.recognize(q)
            return (m.orchestrator.id, m.orchestrator.run(q).answer.split("\n")[0]) if m else (None, None)

        self.assertEqual(route("What is the average of the price column in sales?"), ("dataset", "20"))
        self.assertEqual(route("How many rows are in the sales dataset?"), ("dataset", "3"))
        self.assertEqual(route("How many rows in sales where status = 'active'?"), ("dataset", "2"))
        self.assertEqual(route("Describe the dataset sales.")[0], "dataset")
        # guard: a plain stat query (no dataset) must NOT be claimed by the dataset domain
        self.assertEqual(route("What is the average of 10, 20, 30?")[0], "list_statistic")

    def test_arrow_and_npy(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.feather as feather
        feather.write_feather(pa.table({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]}),
                              os.path.join(self.tmp.name, "metrics.arrow"))
        np.save(os.path.join(self.tmp.name, "matrix.npy"),
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        # Arrow/Feather: tabular, columns by name
        self.assertEqual(self.ds.aggregate("metrics", "b", "sum"), "60")
        self.assertEqual(self.ds.count_rows("metrics"), "3")
        # .npy: 2-D array → columns c0, c1
        self.assertIn("matrix", self.ds.list_datasets())
        self.assertEqual(self.ds.aggregate("matrix", "c0", "sum"), "9")   # 1+3+5
        # round-trip the other way: export to npy and to arrow
        self.ds.convert("metrics", "metrics_npy", "npy")
        self.assertIn("metrics_npy", self.ds.list_datasets())
        self.ds.convert("matrix", "matrix_arrow", "arrow")
        self.assertEqual(self.ds.aggregate("matrix_arrow", "c1", "sum"), "12")  # 2+4+6

    def test_import_file(self):
        src = os.path.join(self.tmp.name, "external.csv")
        with open(src, "w") as f:
            f.write("x\n1\n2\n3\n4\n")
        msg = self.ds.import_file(src, name="imported")
        self.assertIn("4 rows", msg)
        self.assertEqual(self.ds.aggregate("imported", "x", "sum"), "10")


if __name__ == "__main__":
    unittest.main()
