"""The coded agreement comparator used by the batch fidelity eval."""

import unittest

import reasoning.eval as evmod
from reasoning.eval import agrees, fidelity


class TestFidelityRobustness(unittest.TestCase):
    def test_reports_orchestrator_crash_instead_of_raising(self):
        # an incomplete scaffold whose run() raises must be REPORTED, not crash eval
        class _Bad:
            example_queries = ["q"]

            def run(self, query):
                raise RuntimeError("boom")

        class _Teacher:
            def trace(self, query, seed=0):
                return type("R", (), {"answer": "42"})()

        orig = evmod.get_teacher
        evmod.get_teacher = lambda backend: _Teacher()
        try:
            res = fidelity("q", _Bad(), teacher_backend="stub")
        finally:
            evmod.get_teacher = orig
        self.assertEqual(res["score"], 0)
        self.assertIn("RuntimeError", res["orchestrator"])


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
