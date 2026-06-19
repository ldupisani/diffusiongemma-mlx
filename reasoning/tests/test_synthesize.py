"""Goal-directed plan synthesis: re-derive a hand-written plan, prove shortest-path, the
memoization cache, and graceful no-path decline — all model-free."""

import unittest

from reasoning import synthesize as syn
from reasoning.compiled import model_calls


class TestSynthesize(unittest.TestCase):
    def test_rederives_discount_then_tax(self):
        seq = syn.synthesize({"price", "discount_pct", "tax_pct"}, "final_price", syn.FINANCE_OPS)
        self.assertEqual([op.skill for op in seq],
                         ["arith.apply_discount", "arith.apply_tax"])  # the shortest path

    def test_synthesized_plan_matches_handwritten_answer(self):
        # percent_discount_tax pins "$215.14" for 17% off $240 then 8% tax
        r = syn.synthesize_and_run(
            {"price": 240.0, "discount_pct": 17.0, "tax_pct": 8.0}, "final_price", syn.FINANCE_OPS)
        self.assertEqual(r.answer, "215.14")     # 240*0.83*1.08 = 215.136
        self.assertEqual(model_calls(r), 0)      # synthesized AND executed model-free

    def test_already_known_goal_is_zero_ops(self):
        self.assertEqual(syn.synthesize({"final_price"}, "final_price", syn.FINANCE_OPS), [])

    def test_no_path_declines(self):
        # cannot reach final_price without tax_pct
        self.assertIsNone(syn.synthesize({"price", "discount_pct"}, "final_price", syn.FINANCE_OPS))

    def test_memoized_search_runs_once(self):
        memo = {}
        before = syn._searches
        syn.synthesize({"price", "discount_pct", "tax_pct"}, "final_price", syn.FINANCE_OPS, memo=memo)
        syn.synthesize({"price", "discount_pct", "tax_pct"}, "final_price", syn.FINANCE_OPS, memo=memo)
        self.assertEqual(syn._searches - before, 1)  # second call was a cache hit


if __name__ == "__main__":
    unittest.main()
