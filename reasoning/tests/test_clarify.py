"""Needs-input clarification: a recognized query TYPE with a missing field asks rather than
declines, and re-running with the appended answer computes — all model-free."""

import unittest

from reasoning import router
from reasoning.compiled import model_calls, needs_input


class TestClarify(unittest.TestCase):
    def _run(self, query):
        m = router.recognize(query)
        self.assertIsNotNone(m, f"{query!r} should route")
        return m, m.orchestrator.run(query)

    def test_bmi_asks_for_both_fields(self):
        m, r = self._run("What is my BMI?")
        self.assertEqual(m.orchestrator.id, "health_metrics")
        missing = needs_input(r)
        self.assertIsNotNone(missing)
        self.assertEqual(set(missing), {"weight_kg", "height_m"})
        self.assertEqual(model_calls(r), 0)

    def test_bmi_asks_only_for_missing_field(self):
        _, r = self._run("What is my BMI if I weigh 70 kg?")
        self.assertEqual(set(needs_input(r)), {"height_m"})

    def test_geometry_asks_for_radius(self):
        m, r = self._run("What is the area of a circle?")
        self.assertEqual(m.orchestrator.id, "geometry")
        self.assertEqual(set(needs_input(r)), {"radius"})

    def test_rehydrate_then_compute(self):
        # simulate the CLI loop: append the user's answers via each field's template, re-run
        query, (_m, r) = "What is my BMI?", self._run("What is my BMI?")
        for _field, (_q, template) in needs_input(r).items():
            value = {"weight_kg": "70", "height_m": "1.75"}[_field]
            query = f"{query} {template.format(value)}"
        m2, r2 = self._run(query)
        self.assertEqual(m2.orchestrator.id, "health_metrics")
        self.assertIsNone(needs_input(r2))      # fully bound now
        self.assertEqual(r2.answer, "22.86")    # 70 / 1.75^2
        self.assertEqual(model_calls(r2), 0)

    def test_off_type_still_declines(self):
        # a genuinely unrelated query must NOT trigger a clarify loop
        m = router.recognize("Who painted the Mona Lisa?")
        if m:  # if anything claims it, it must not be a needs-input result
            self.assertIsNone(needs_input(m.orchestrator.run("Who painted the Mona Lisa?")))


if __name__ == "__main__":
    unittest.main()
