"""Cross-domain decomposition: code-first (0 model calls), no mis-splitting of single
queries, and the ≤1-call model fallback (only the split uses the model; parts are coded)."""

import unittest

from reasoning import router
from reasoning.compiled import model_calls
from reasoning.decompose import decompose_code_first, decompose_with_model


class TestDecompose(unittest.TestCase):
    def test_code_first_two_clauses_model_free(self):
        r = decompose_code_first("Convert 5 miles to km, and what is 20% of 50?")
        self.assertIsNotNone(r)
        self.assertEqual(model_calls(r), 0)
        self.assertIn("8.05 km", r.answer)
        self.assertIn("10", r.answer)

    def test_routes_via_composer_tier(self):
        m = router.recognize("Convert 5 miles to km, and what is 20% of 50?")
        self.assertIsNotNone(m)
        self.assertEqual(m.orchestrator.id, "compose")
        self.assertEqual(m.via, "compose")

    def test_does_not_missplit_single_query(self):
        # "length 4 and width 6" is ONE geometry query — must NOT be split
        self.assertIsNone(
            decompose_code_first("What is the area of a rectangle with length 4 and width 6?"))
        m = router.recognize("What is the area of a rectangle with length 4 and width 6?")
        self.assertEqual(m.orchestrator.id, "geometry")

    def test_declines_when_a_clause_is_unknown(self):
        self.assertIsNone(decompose_code_first("What is 20% of 50, and who painted the Mona Lisa?"))

    def test_model_fallback_uses_exactly_one_call(self):
        # a compound that does NOT code-split; a stub shot returns the sub-questions
        calls = []

        class _R:
            def __init__(self, text): self.text = text

        def stub_shot(prompt, **kw):
            calls.append(prompt)
            return _R("1. What is 7 times 8?\n2. What is the GCD of 24 and 36?")

        r = decompose_with_model("Give me 7 times 8 plus the gcd thing", shot=stub_shot)
        self.assertIsNotNone(r)
        self.assertEqual(len(calls), 1)          # the model was used ONCE (to split)
        self.assertEqual(model_calls(r), 1)      # sub-answers were coded (0 each)
        self.assertIn("56", r.answer)
        self.assertIn("12", r.answer)


if __name__ == "__main__":
    unittest.main()
