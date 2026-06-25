"""Residue guard (§3d): a query value the skill doesn't consume must not yield a confident
answer. Each base query answers; the same query with an extra, irrelevant value declines."""

import importlib
import unittest

from reasoning import router
from reasoning.bind._common import has_unconsumed_numbers, query_numbers

duckdb = importlib.util.find_spec("duckdb")


def _answer(q):
    m = router.recognize(q)
    if not m:
        return None
    a = m.orchestrator.run(q).answer or ""
    return None if a.startswith("(no result") else a


@unittest.skipUnless(duckdb, "registry import needs duckdb (dataset orchestrator)")
class TestResidueGuard(unittest.TestCase):
    # (answers, declines-because-of-an-extra-value) across the audited fixed-arity domains
    PAIRS = [
        ("What is the area of a circle with radius 5?",
         "What is the area of a circle with radius 5 and height 3?"),
        ("What is the sine of 30 degrees?",
         "What is the sine of 30 degrees over 7 days?"),
        ("Convert 100 kilometers to miles.",
         "Convert 100 kilometers to miles over 4 days."),
        ("What is the final balance on $1000 at 5% compounded annually for 3 years?",
         "What is the final balance on $1000 at 5% compounded annually for 3 years in 2 banks?"),
        ("Convert 42 to binary.",
         "Convert 42 to binary with 7 bits?"),
    ]

    def test_extra_value_declines(self):
        for base, extra in self.PAIRS:
            with self.subTest(base=base):
                self.assertIsNotNone(_answer(base), f"base should answer: {base}")
                self.assertIsNone(_answer(extra), f"extra-value variant should decline: {extra}")

    def test_guard_helper(self):
        self.assertTrue(has_unconsumed_numbers("radius 5 and height 3", [5]))
        self.assertFalse(has_unconsumed_numbers("radius 5", [5]))
        # times, dates, ratios are not free-standing magnitudes
        self.assertEqual(query_numbers("9:00 and 2024-03-03 and 3/4"), set())
        # commas and percents read as single values
        self.assertEqual(query_numbers("$1,000 and 5%"), {1000.0, 5.0})


if __name__ == "__main__":
    unittest.main()
