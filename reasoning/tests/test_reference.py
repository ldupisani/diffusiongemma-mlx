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
        self.assertEqual(self.ref.city_population("Tokyo"), 14010000)          # latest (2023)

    def test_sql_queryable_via_dataset_engine(self):
        self.assertEqual(self.ds.aggregate("countries", "name", "count"), "30")
        self.assertEqual(self.ds.count_rows("cities"), "60")
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

    def test_population_year_awareness(self):
        from reasoning import router

        def ans(q):
            return router.recognize(q).orchestrator.run(q).answer

        # city + a SEED year → honored exactly, no caveat (seed years now span 1990–2023)
        self.assertEqual(ans("What is the population of London in 2010?"), "8000000")
        self.assertEqual(ans("What is the population of London in 2023?"), "8900000")
        # country exact seed year, and city with no year (latest) → plain, no caveat
        self.assertEqual(ans("What is the population of France in 2020?"), "67000000")
        self.assertEqual(ans("What is the population of Tokyo?"), "14010000")
        # provenance flags
        self.assertTrue(self.ref.population_detail("London", 2023)["year_applied"])
        self.assertFalse(self.ref.population_detail("London", 2024)["year_applied"])
        self.assertIsNone(self.ref.population_detail("France", 1995))  # year absent → no fabrication

    def test_population_projection(self):
        from reasoning import router

        def run(q):
            m = router.recognize(q)
            r = m.orchestrator.run(q)
            from reasoning.compiled import model_calls
            return r.answer, model_calls(r)

        # a FUTURE year → trend extrapolation, flagged an estimate, still 0 model calls
        ans, mc = run("What is the projected population for London in 2030?")
        self.assertIn("projected", ans)
        self.assertIn("not curated", ans)
        self.assertEqual(mc, 0)
        # an in-range gap year → interpolation between the bracketing seed years
        self.assertIn("interpolated", run("What is the population of France in 2015?")[0])
        # API: exact year and unknown place don't project
        p = self.ref.project_population("London", 2030)
        self.assertTrue(p["value"] > 8000000 and "trend" in p["basis"])
        self.assertEqual(self.ref.project_population("France", 2015)["value"], 66000000)
        self.assertIsNone(self.ref.project_population("London", 2023))   # exact year
        self.assertIsNone(self.ref.project_population("Atlantis", 2030))  # unknown place

    def test_user_dataset_shadows_reference(self):
        # a user dataset named like a reference table takes precedence
        with open(os.path.join(self.data.name, "countries.csv"), "w") as f:
            f.write("name,x\nFoo,1\nBar,2\n")
        self.assertEqual(self.ds.count_rows("countries"), "2")


@unittest.skipUnless(duckdb, "duckdb not installed")
class TestReferenceRetrieval(unittest.TestCase):
    """Missing data point → retrieved via a (stubbed) Claude Sonnet call, validated and flagged."""

    def setUp(self):
        self.data = tempfile.TemporaryDirectory()
        self.refdir = tempfile.TemporaryDirectory()
        os.environ["DIFG_DATA_DIR"] = self.data.name
        os.environ["DIFG_REFERENCE_DIR"] = self.refdir.name
        os.environ["DIFG_REFERENCE_RETRIEVAL"] = "1"
        import reasoning.teachers as teachers
        self._teachers = teachers
        self._orig = teachers.ask_claude
        self.calls = {"n": 0}

        def stub(prompt, model="sonnet", timeout=60):
            self.calls["n"] += 1
            if "CSV" in prompt:                              # batch (retrieve_set) request
                return "```csv\nname,population\nParis,2100000\nLyon,510000\n```"
            return "3170000"                                 # single point

        teachers.ask_claude = stub
        from reasoning import reference, router
        self.router, self.reference = router, reference

    def tearDown(self):
        self._teachers.ask_claude = self._orig
        for k in ("DIFG_DATA_DIR", "DIFG_REFERENCE_DIR", "DIFG_REFERENCE_RETRIEVAL"):
            os.environ.pop(k, None)
        self.data.cleanup()
        self.refdir.cleanup()

    def _run(self, q):
        from reasoning.compiled import model_calls
        m = self.router.recognize(q)
        if not m:
            return (None, None, None)
        r = m.orchestrator.run(q)
        return (m.orchestrator.id, r.answer, model_calls(r))

    def test_unknown_place_retrieved_and_flagged(self):
        rid, ans, mc = self._run("What is the population of Mongolia in 2015?")
        self.assertEqual(rid, "geography")
        self.assertIn("3170000", ans)
        self.assertIn("Claude", ans)           # provenance flagged
        self.assertEqual(mc, 1)                 # the model call is recorded

    def test_curated_lookup_stays_zero_calls(self):
        self.assertEqual(self._run("What is the population of France in 2020?"),
                         ("geography", "67000000", 0))

    def test_validation_rejects_implausible(self):
        self._teachers.ask_claude = lambda *a, **k: "not a number"
        _, ans, mc = self._run("What is the population of Mongolia in 2015?")
        self.assertTrue(ans is None or ans.startswith("(no result"))   # failed verification → decline
        self.assertEqual(mc or 0, 0)

    def test_retrieve_validates_shape_and_range(self):
        r = self.reference
        self.assertEqual(r.retrieve("coordinates", "Reykjavik", asker=lambda p: "64.1466, -21.9426"),
                         "64.1466, -21.9426")
        self.assertIsNone(r.retrieve("coordinates", "X", asker=lambda p: "999, 999"))  # out of range
        self.assertEqual(r.retrieve("population", "X", asker=lambda p: "3.0 million"), "3000000")

    def test_persist_and_cache(self):
        q = "What is the population of Mongolia in 2015?"
        self.assertEqual(self._run(q)[2], 1)                 # first time: a model call
        rid, ans, mc = self._run(q)                          # second time: served from the log
        self.assertEqual(mc, 0)
        self.assertIn("previously retrieved", ans)
        self.assertEqual(self.calls["n"], 1)                 # the model was asked only once
        self.assertEqual(self.reference.cached_value("population", "Mongolia", 2015), "3170000")
        from reasoning import datasets
        self.assertIn("Mongolia", datasets.query("SELECT place FROM retrieved"))  # queryable log

    def test_promote_to_curated(self):
        self.assertEqual(self._run("What is the population of Mongolia in 2015?")[2], 1)  # retrieve
        msg = self.reference.promote("population", "Mongolia", 2015)
        self.assertIn("Mongolia", msg)
        self.assertEqual(self.reference.list_retrieved(), [])           # log cleared
        # now a verified, curated, 0-model-call lookup — even with retrieval turned OFF
        os.environ["DIFG_REFERENCE_RETRIEVAL"] = ""
        self.assertEqual(self._run("What is the population of Mongolia in 2015?"),
                         ("geography", "3170000", 0))
        # and queryable as curated data
        from reasoning import datasets
        self.assertEqual(datasets.query("SELECT population FROM population WHERE place='Mongolia'"),
                         "3170000")

    def test_batch_set_retrieval(self):
        rid, ans, mc = self._run("List all cities in France.")
        self.assertEqual(rid, "geography")
        self.assertEqual(mc, 1)
        self.assertIn("model-sourced", ans)
        from reasoning import datasets
        self.assertEqual(datasets.count_rows("list_all_cities_in_france"), "2")
        self.assertEqual(datasets.aggregate("list_all_cities_in_france", "population", "max"), "2100000")


@unittest.skipUnless(duckdb, "duckdb not installed")
class TestRetrievalDisabledByDefault(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.TemporaryDirectory()
        self.refdir = tempfile.TemporaryDirectory()
        os.environ["DIFG_DATA_DIR"] = self.data.name
        os.environ["DIFG_REFERENCE_DIR"] = self.refdir.name
        os.environ.pop("DIFG_REFERENCE_RETRIEVAL", None)

    def tearDown(self):
        for k in ("DIFG_DATA_DIR", "DIFG_REFERENCE_DIR"):
            os.environ.pop(k, None)
        self.data.cleanup()
        self.refdir.cleanup()

    def test_unknown_place_not_claimed_when_disabled(self):
        from reasoning import router
        m = router.recognize("What is the population of Mongolia in 2015?")
        self.assertTrue(m is None or m.orchestrator.id != "geography")  # no model call by default


if __name__ == "__main__":
    unittest.main()
