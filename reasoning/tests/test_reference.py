"""Bundled reference data — seed builds into SQLite, lookups, SQL queryability via the dataset
engine, and the CSV import path. Uses isolated temp dirs; no model calls."""

import importlib
import os
import tempfile
import unittest

duckdb = importlib.util.find_spec("duckdb")


@unittest.skipUnless(duckdb, "duckdb not installed")
class TestReference(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.TemporaryDirectory()
        self.ref_dir = tempfile.TemporaryDirectory()
        os.environ["DIFG_DATA_DIR"] = self.data.name
        os.environ["DIFG_REFERENCE_DIR"] = self.ref_dir.name   # build into temp, not the repo
        from reasoning import datasets, reference
        self.ref, self.ds = reference, datasets

    def tearDown(self):
        for k in ("DIFG_DATA_DIR", "DIFG_REFERENCE_DIR"):
            os.environ.pop(k, None)
        self.data.cleanup()
        self.ref_dir.cleanup()

    def test_build_and_lookups(self):
        self.assertEqual(self.ref.country_field("France", "capital"), "Paris")
        self.assertEqual(self.ref.country_field("Japan", "continent"), "Asia")
        self.assertEqual(self.ref.country_population("France", 2020), 67000000)
        self.assertEqual(self.ref.country_population("France", 2000), 61000000)
        self.assertEqual(self.ref.coordinates("Paris"), (48.8566, 2.3522))
        self.assertEqual(self.ref.coordinates("Japan"), (35.6762, 139.6503))  # capital coords
        self.assertEqual(self.ref.city_population("Tokyo"), 13960000)

    def test_sql_queryable_via_dataset_engine(self):
        self.assertEqual(self.ds.aggregate("countries", "name", "count"), "30")
        self.assertEqual(self.ds.count_rows("cities"), "40")
        self.assertEqual(
            self.ds.query("SELECT count(*) FROM population WHERE year=2020 AND population>100000000"),
            "10")
        self.assertIn("Shanghai", self.ds.query("SELECT name FROM cities ORDER BY population DESC LIMIT 1"))
        self.assertIn("countries", self.ds.describe("countries"))

    def test_import_csv_extends_reference(self):
        src = os.path.join(self.data.name, "rivers.csv")
        with open(src, "w") as f:
            f.write("name,length_km\nNile,6650\nAmazon,6400\n")
        self.ref.import_csv(src, "rivers")
        self.assertEqual(self.ds.query("SELECT max(length_km) FROM rivers"), "6650")

    def test_nl_geography_routing(self):
        from reasoning import router

        def route(q):
            m = router.recognize(q)
            return (m.orchestrator.id, m.orchestrator.run(q).answer) if m else (None, None)

        self.assertEqual(route("What is the population of France in 2020?"), ("geography", "67000000"))
        self.assertEqual(route("What is the capital of Japan?"), ("geography", "Tokyo"))
        self.assertEqual(route("What are the coordinates of Paris?"), ("geography", "48.8566, 2.3522"))
        self.assertEqual(route("What continent is Nigeria in?"), ("geography", "Africa"))
        self.assertEqual(route("Where is New Delhi?"), ("geography", "28.6139, 77.209"))
        # guard: geometry's "area of a circle" is NOT stolen by geography's "area" keyword
        self.assertEqual(
            router.recognize("What is the area of a circle with radius 5?").orchestrator.id, "geometry")

    def test_user_dataset_shadows_reference(self):
        # a user dataset named like a reference table takes precedence
        with open(os.path.join(self.data.name, "countries.csv"), "w") as f:
            f.write("name,x\nFoo,1\nBar,2\n")
        self.assertEqual(self.ds.count_rows("countries"), "2")


if __name__ == "__main__":
    unittest.main()
