"""The coded agreement comparator used by the batch fidelity eval."""

import unittest

from reasoning.eval import agrees


class TestAgrees(unittest.TestCase):
    def test_string_containment(self):
        self.assertTrue(agrees("18.02 g/mol", "≈ 18.02 g/mol"))
        self.assertTrue(agrees("70%", "70 percent"))
        self.assertTrue(agrees("(4, 6)", "the midpoint is (4, 6)"))

    def test_numeric_tolerance(self):
        # coded rounds to 3.1416, teacher says 3.14159 — agree within 1%
        self.assertTrue(agrees("3.1416 rad", "π radians ≈ 3.14159 radians"))
        self.assertTrue(agrees("5", "5.0 units"))

    def test_verdict_agreement(self):
        # prose differs but both say "yes, tautology"
        self.assertTrue(agrees("Yes, it is a tautology.",
                               "Yes, (P OR NOT P) is a tautology (law of excluded middle)."))
        self.assertFalse(agrees("Yes, it is satisfiable.", "No, it is unsatisfiable."))

    def test_disagreement(self):
        self.assertFalse(agrees("5", "13"))
        self.assertFalse(agrees("FF", "binary"))


if __name__ == "__main__":
    unittest.main()
