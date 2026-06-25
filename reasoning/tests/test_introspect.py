"""Self-introspection — the capability API, the NL meta-query orchestrator, and the guard that
it never steals a real computational query. Uses temp dirs (introspect reaches the dataset/
reference layers via datagen)."""

import importlib
import os
import tempfile
import unittest

duckdb = importlib.util.find_spec("duckdb")


@unittest.skipUnless(duckdb, "duckdb not installed")
class TestIntrospect(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.TemporaryDirectory()
        self.refdir = tempfile.TemporaryDirectory()
        os.environ["DIFG_DATA_DIR"] = self.data.name
        os.environ["DIFG_REFERENCE_DIR"] = self.refdir.name
        from reasoning import introspect, router
        self.introspect, self.router = introspect, router

    def tearDown(self):
        for k in ("DIFG_DATA_DIR", "DIFG_REFERENCE_DIR"):
            os.environ.pop(k, None)
        self.data.cleanup()
        self.refdir.cleanup()

    def _route(self, q):
        m = self.router.recognize(q)
        return (m.orchestrator.id, m.orchestrator.run(q).answer) if m else (None, None)

    # --- API ---
    def test_summary_and_domains(self):
        s = self.introspect.summary()
        self.assertGreater(s["domains"], 50)
        self.assertGreater(s["skills"], 300)
        ids = {d["id"] for d in self.introspect.domains()}
        self.assertIn("number_theory", ids)
        self.assertIn("introspect", ids)

    def test_domain_detail_and_search(self):
        d = self.introspect.domain("geometry")
        self.assertTrue(d and d["examples"])
        self.assertIsNone(self.introspect.domain("nope_domain"))
        hit = self.introspect.search("prime")
        self.assertIn("number_theory", hit["domains"])
        self.assertIn("primes", hit["generators"])

    # --- NL orchestrator ---
    def test_overview(self):
        rid, ans = self._route("What can you do?")
        self.assertEqual(rid, "introspect")
        self.assertIn("reasoning domains", ans)

    def test_about_and_query_topic(self):
        self.assertEqual(self._route("What can I ask about geometry?")[0], "introspect")
        self.assertIn("area", self._route("What can I ask about geometry?")[1].lower())
        self.assertEqual(self._route("How do I query physics?")[0], "introspect")

    def test_unknown_topic_is_graceful(self):
        rid, ans = self._route("What can I ask about dinosaurs?")
        self.assertEqual(rid, "introspect")
        self.assertIn("don't have", ans)

    def test_does_not_steal_real_queries(self):
        # carries digits / a real domain match → resolved by that domain, not introspect
        self.assertEqual(self.router.recognize("What is the area of a circle with radius 5?")
                         .orchestrator.id, "geometry")
        self.assertEqual(self.router.recognize("What is 2 + 2?").orchestrator.id, "arithmetic")

    # --- introspection skills ---
    def test_introspection_skills(self):
        from reasoning.skills import SKILLS
        self.assertTrue(SKILLS["introspect.domain_exists"].fn("number_theory"))
        self.assertFalse(SKILLS["introspect.domain_exists"].fn("nope"))
        self.assertTrue(SKILLS["introspect.can_answer"].fn("prime"))
        self.assertFalse(SKILLS["introspect.can_answer"].fn("xyzzy_nonsense_topic"))


if __name__ == "__main__":
    unittest.main()
