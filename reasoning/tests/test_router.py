"""EmbeddingMatcher: recognizes covered query types (incl. paraphrases), declines off-domain."""

import unittest

from reasoning import router


class TestEmbeddingMatcher(unittest.TestCase):
    def setUp(self):
        self.orchs = router.load_orchestrators()
        self.matcher = router.EmbeddingMatcher()

    def _route(self, query):
        return router.recognize(query, matcher=self.matcher, orchestrators=self.orchs)

    def test_exact_examples_route_correctly(self):
        cases = [
            ("What is 17% off $240 then 8% tax?", "percent_discount_tax"),
            ("What is the geometric mean of 2, 8, and 32?", "list_statistic"),
            ("What is the monthly payment on a $20000 loan at 6% annual interest over 5 years?",
             "loan_payment"),
            ("Is 2024 a leap year?", "leap_year"),
        ]
        for q, oid in cases:
            with self.subTest(q=q):
                m = self._route(q)
                self.assertIsNotNone(m)
                self.assertEqual(m.orchestrator.id, oid)
                self.assertEqual(m.via, "embedding")

    def test_paraphrases_still_route(self):
        # surface words differ from the stored examples; vector similarity still recognizes
        for q, oid in [("average of 3, 6, and 9", "list_statistic"),
                       ("what weekday was 2000-01-01", "weekday_of_date")]:
            with self.subTest(q=q):
                m = self._route(q)
                self.assertIsNotNone(m, f"{q!r} should route")
                self.assertEqual(m.orchestrator.id, oid)

    def test_off_domain_falls_back(self):
        for q in ["Explain the causes of World War I", "Who painted the Mona Lisa?"]:
            with self.subTest(q=q):
                self.assertIsNone(self._route(q), f"{q!r} should not match any orchestrator")

    def test_embedding_is_deterministic(self):
        # crc32-based hashing must be stable across calls (no salted built-in hash)
        a = router.hashing_embed(["compound annual growth rate"])[0]
        b = router.hashing_embed(["compound annual growth rate"])[0]
        self.assertEqual(a, b)


class TestFallbackTier(unittest.TestCase):
    """A specific orchestrator always beats the generic numeric_consensus fallback."""

    def test_specific_beats_fallback_on_paraphrase(self):
        # not an exact example query — would have TIED numeric_consensus on "many" before
        m = router.recognize("Count the vowels in the word 'banana'")
        self.assertIsNotNone(m)
        self.assertEqual(m.orchestrator.id, "string_operation")
        self.assertFalse(m.orchestrator.fallback)

    def test_pure_numeric_falls_back(self):
        for q in ["What is 17 percent of 240?",
                  "How many times does the digit 7 appear from 1 to 100?"]:
            with self.subTest(q=q):
                m = router.recognize(q)
                self.assertIsNotNone(m)
                self.assertEqual(m.orchestrator.id, "numeric_consensus")
                self.assertTrue(m.orchestrator.fallback)


if __name__ == "__main__":
    unittest.main()
