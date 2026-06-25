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

    def test_domain_generators(self):
        from reasoning import datagen
        datagen.generate("primes", "p", n="10")
        self.assertEqual(self.ds.count_rows("p"), "10")
        self.assertEqual(self.ds.query("SELECT max(prime) FROM p"), "29")   # 10th prime
        datagen.generate("fibonacci", "fib", n="20")
        self.assertEqual(self.ds.query("SELECT max(value) FROM fib"), "4181")  # F_19
        datagen.generate("normal_sample", "ns", n="2000", mean="10", seed="0")
        self.assertTrue(9.5 < float(self.ds.aggregate("ns", "value", "avg")) < 10.5)
        datagen.generate("amortization", "am", principal="100000", annual_rate="6", years="30")
        self.assertEqual(self.ds.count_rows("am"), "360")
        datagen.generate("logistic_trajectory", "lt", r="3.9", x0="0.5", n="300")
        self.assertEqual(self.ds.count_rows("lt"), "300")
        # logistics: EOQ cost curve (min total cost ≈ sqrt(2·D·S·H) = 200) and inventory sim
        datagen.generate("eoq_curve", "eoq", demand="1000", order_cost="10", holding_cost="2")
        self.assertEqual(self.ds.count_rows("eoq"), "50")
        self.assertTrue(195 < float(self.ds.query("SELECT min(total_cost) FROM eoq")) < 210)
        datagen.generate("inventory_simulation", "inv", periods="52")
        self.assertEqual(self.ds.count_rows("inv"), "52")

    def test_nl_generation_routing(self):
        from reasoning import router
        q = "Generate the first 100 primes."
        m = router.recognize(q)
        self.assertEqual(m.orchestrator.id, "dataset")
        m.orchestrator.run(q)
        self.assertIn("primes", self.ds.list_datasets())
        self.assertEqual(self.ds.count_rows("primes"), "100")
        # a non-dataset "generate" must not be claimed
        self.assertIsNone(router.recognize("Generate some ideas for dinner."))

    def test_import_file(self):
        src = os.path.join(self.tmp.name, "external.csv")
        with open(src, "w") as f:
            f.write("x\n1\n2\n3\n4\n")
        msg = self.ds.import_file(src, name="imported")
        self.assertIn("4 rows", msg)
        self.assertEqual(self.ds.aggregate("imported", "x", "sum"), "10")


if __name__ == "__main__":
    unittest.main()
