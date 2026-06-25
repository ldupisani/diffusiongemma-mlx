"""Residue guard (§3d): a query value the skill doesn't consume must not yield a confident
answer. Each base query answers; the same query with an extra, irrelevant value declines."""

import importlib
import unittest

from reasoning import router
from reasoning.bind._common import has_unconsumed_numbers, num_near, query_numbers

duckdb = importlib.util.find_spec("duckdb")


class TestNumNear(unittest.TestCase):
    """Bidirectional keyword-anchored extraction — the cases that previously needed hand fixes."""

    def test_number_after_anchor(self):
        self.assertEqual(num_near("demand is 800 and capacity", "demand"), 800.0)
        self.assertEqual(num_near("order cost 50, holding cost 5", "holding cost", "carrying cost"), 5.0)
        self.assertEqual(num_near("n1=1.5, angle=30", "n1"), 1.5)

    def test_number_before_anchor(self):
        self.assertEqual(num_near("0.000001 kg at 0.6c", "kg", "mass"), 0.000001)       # relativity mass
        self.assertEqual(num_near("480 minutes available", "minute", "available"), 480.0)  # takt time
        self.assertEqual(num_near("a 3-qubit system", "qubit"), 3.0)

    def test_nearest_wins_when_anchor_repeats(self):
        q = "capacity utilization if demand is 800 and capacity is 1000"
        self.assertEqual(num_near(q, "capacity"), 1000.0)    # not the distant demand value
        self.assertEqual(num_near(q, "demand"), 800.0)

    def test_no_match_returns_none(self):
        self.assertIsNone(num_near("no numbers here", "demand"))
        self.assertIsNone(num_near("demand is 800", "capacity"))   # anchor absent

    def test_plural_singular_anchor_stemming(self):
        # a plural anchor matches the singular and vice versa (auto-stemmed to the root)
        self.assertEqual(num_near("an 800 lumen lamp", "lumens"), 800.0)   # plural anchor, singular text
        self.assertEqual(num_near("over 5 lumens", "lumen"), 5.0)          # singular anchor, plural text
        self.assertEqual(num_near("using 10 watts", "watt"), 10.0)
        self.assertEqual(num_near("at 5 meters", "meter"), 5.0)
        # 'ss' words and regex/multi-word anchors are left alone
        self.assertEqual(num_near("a mass of 2 kg", "mass"), 2.0)
        self.assertEqual(num_near("what is it at 5 m", r"\bat\b"), 5.0)


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
