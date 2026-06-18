"""Domain-sanity guards: degenerate inputs must yield an explanation, never nonsense/crash."""

import unittest

from reasoning import router
from reasoning.compiled import model_calls


class TestDomainGuards(unittest.TestCase):
    def _run(self, query):
        m = router.recognize(query)
        self.assertIsNotNone(m, f"no match for {query!r}")
        return m.orchestrator.id, m.orchestrator.run(query)

    def test_break_even_below_cost_explains_not_negative(self):
        # sells for $5 but costs $15 — used to return "-100 units"
        oid, r = self._run("Fixed costs are $1000, each unit sells for $5 and costs $15 to make. "
                           "How many units to break even?")
        self.assertEqual(oid, "break_even")
        self.assertIn("no result", r.answer)
        self.assertNotIn("-100", r.answer)   # not the old confident-nonsense answer
        self.assertNotIn("units", r.answer)  # declined to emit a count
        self.assertEqual(model_calls(r), 0)

    def test_break_even_zero_margin_does_not_crash(self):
        # equal price and cost — used to raise ZeroDivisionError
        oid, r = self._run("Fixed costs are $1000, each unit sells for $15 and costs $15 to make. "
                           "How many units to break even?")
        self.assertIn("no result", r.answer)
        self.assertEqual(model_calls(r), 0)

    def test_valid_break_even_still_works(self):
        oid, r = self._run("Fixed costs are $5000, each unit sells for $25 and costs $15 to make. "
                           "How many units to break even?")
        self.assertEqual(r.answer, "500 units")

    def test_tip_split_zero_people_explains(self):
        oid, r = self._run("A $120 bill with an 18% tip split among 0 people — how much each?")
        self.assertEqual(oid, "tip_split")
        self.assertIn("no result", r.answer)
        self.assertEqual(model_calls(r), 0)

    def test_invalid_date_does_not_crash(self):
        # 2024-33-01 is ISO-shaped but month 33 is invalid — must explain, not raise
        oid, r = self._run("What is the day of the year for 2024-33-01?")
        self.assertEqual(oid, "day_of_year")
        self.assertIn("no result", r.answer)
        self.assertEqual(model_calls(r), 0)

    def test_valid_date_still_works(self):
        oid, r = self._run("What is the day of the year for 2024-03-01?")
        self.assertEqual(r.answer, "61")


if __name__ == "__main__":
    unittest.main()
