import unittest

from reasoning import engine
from reasoning.compiled import CodeStep, Inp, Lit, ModelStep, Plan, Step, model_calls, run_plan


def _jacket_plan() -> Plan:
    # 17% off, then 8% tax, rounded as money — three deterministic skills, no model.
    return Plan(
        steps=[
            CodeStep("arith.apply_discount", [Inp("price"), Inp("discount_pct")], out="discounted"),
            CodeStep("arith.apply_tax", [Step("discounted"), Inp("tax_pct")], out="taxed"),
            CodeStep("arith.round_money", [Step("taxed"), Lit(2)], out="final"),
        ],
        answer_ref="final",
        answer_format="${:.2f}",
    )


class TestCompiled(unittest.TestCase):
    def test_coded_plan_makes_zero_model_calls(self):
        bound = {"price": 240, "discount_pct": 17, "tax_pct": 8}
        # shot=_no_model raises if ANY step tries the model — so reaching the result proves
        # the whole plan ran in pure code.
        r = run_plan(_jacket_plan(), bound, shot=engine._no_model)
        self.assertEqual(r.answer, "$215.14")
        self.assertEqual(model_calls(r), 0)

    def test_hybrid_plan_counts_model_steps(self):
        calls = []

        class FakeResult:
            def __init__(self, text):
                self.text = text

        def fake_shot(prompt, **kw):
            calls.append(prompt)
            return FakeResult("a one-sentence summary")

        plan = Plan(
            steps=[
                CodeStep("arith.percent_of", [Inp("value"), Lit(10)], out="tip"),
                ModelStep("Summarize politely that the tip is {tip}.", out="msg",
                          args={"tip": Step("tip")}),
            ],
            answer_ref="msg",
        )
        r = run_plan(plan, {"value": 80}, shot=fake_shot)
        self.assertEqual(model_calls(r), 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("8.0", calls[0])  # the coded tip flowed into the model prompt

    def test_no_model_guard_raises_on_model_step(self):
        plan = Plan(steps=[ModelStep("do {x}", out="o", args={"x": Lit("thing")})],
                    answer_ref="o")
        with self.assertRaises(RuntimeError):
            run_plan(plan, {}, shot=engine._no_model)


if __name__ == "__main__":
    unittest.main()
