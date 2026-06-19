"""SubStep composition: raw-value exposure, sub-orchestrator delegation, model-call folding,
and graceful decline when a sub-question yields no value. All sidecar-cold (0 model calls)."""

import unittest

from reasoning import engine
from reasoning.compiled import (CodeStep, Inp, Lit, Plan, Step, SubStep, model_calls,
                                run_plan)


class TestCompose(unittest.TestCase):
    def test_raw_value_exposed(self):
        plan = Plan(steps=[CodeStep("arith.add", [Inp("a"), Inp("b")], out="s")],
                    answer_ref="s", answer_format=lambda x: f"={x}")
        r = run_plan(plan, {"a": 2.0, "b": 3.0}, shot=engine._no_model)
        self.assertEqual(r.answer, "=5.0")
        self.assertEqual(r.data["__value__"], 5.0)  # raw, pre-format

    def test_substep_delegates_and_stays_model_free(self):
        # convert 5 miles → km (delegated), then × 2
        plan = Plan(
            steps=[
                SubStep("how many km is {q} miles", out="km",
                        args={"q": Inp("q")}, orchestrator="unit_conversion"),
                CodeStep("arith.multiply", [Step("km"), Lit(2.0)], out="t"),
            ],
            answer_ref="t")
        r = run_plan(plan, {"q": 5.0}, shot=engine._no_model)
        self.assertEqual(model_calls(r), 0)           # composition is model-free
        self.assertAlmostEqual(r.data["__value__"], 16.1)  # 8.05 km × 2
        self.assertTrue(any(e.get("via") == "sub" for e in r.trace))

    def test_substep_declines_gracefully(self):
        plan = Plan(steps=[SubStep("the weather in Paris", out="x")], answer_ref="x")
        r = run_plan(plan, {}, shot=engine._no_model)
        self.assertIn("no result", r.answer)
        self.assertEqual(model_calls(r), 0)


if __name__ == "__main__":
    unittest.main()
