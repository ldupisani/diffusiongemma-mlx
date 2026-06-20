"""Goal-directed plan synthesis, generalized across domains: re-derive hand-written plans,
multi-step chains in new domains, goal-directed op selection, cross-domain isolation (one
shared op pool), ambiguity detection, memoization, and no-path decline — all model-free."""

import unittest

from reasoning import synthesize as syn
from reasoning.compiled import model_calls


class TestSynthesize(unittest.TestCase):
    # --- finance (re-derives percent_discount_tax) ---
    def test_rederives_discount_then_tax(self):
        seq = syn.synthesize({"price", "discount_pct", "tax_pct"}, "final_price", syn.FINANCE_OPS)
        self.assertEqual([op.skill for op in seq],
                         ["arith.apply_discount", "arith.apply_tax"])

    def test_finance_matches_handwritten_answer(self):
        r = syn.synthesize_and_run(
            {"price": 240.0, "discount_pct": 17.0, "tax_pct": 8.0}, "final_price")
        self.assertEqual(r.answer, "215.14")      # 240*0.83*1.08
        self.assertEqual(model_calls(r), 0)

    # --- a NEW domain, a deeper chain: force → work → power (3 ops) ---
    def test_physics_three_op_chain(self):
        seq = syn.synthesize({"mass", "acceleration", "distance", "time"}, "power")
        self.assertEqual([op.skill for op in seq],
                         ["physics.force", "physics.work", "physics.power"])
        r = syn.synthesize_and_run(
            {"mass": 2.0, "acceleration": 3.0, "distance": 4.0, "time": 5.0}, "power")
        # force=6, work=24, power=4.8
        self.assertEqual(r.answer, "4.8")
        self.assertEqual(model_calls(r), 0)

    # --- goal-directed: same domain, the asked goal selects the op ---
    def test_motion_goal_directed(self):
        self.assertEqual([o.skill for o in syn.synthesize({"speed", "time"}, "distance")],
                         ["arith.multiply"])
        self.assertEqual([o.skill for o in syn.synthesize({"distance", "time"}, "speed")],
                         ["physics.speed"])
        self.assertEqual([o.skill for o in syn.synthesize({"distance", "speed"}, "time")],
                         ["arith.divide"])

    # --- simple interest: a third domain in the same pool ---
    def test_interest_domain(self):
        r = syn.synthesize_and_run(
            {"principal": 1000.0, "rate_pct": 5.0, "years": 3.0}, "final_balance")
        self.assertEqual(r.answer, "1150")        # 1000*(1 + 0.05*3) maturity amount
        self.assertEqual(model_calls(r), 0)

    # --- the generalization claim: all domains share ONE pool without interfering ---
    def test_cross_domain_isolation(self):
        # finance knowns must NOT pull in physics ops, and vice versa — disjoint role names
        fin = syn.synthesize({"price", "discount_pct", "tax_pct"}, "final_price")  # ALL_OPS
        self.assertTrue(all(o.skill.startswith(("arith.",)) for o in fin))
        phys = syn.synthesize({"mass", "acceleration", "distance", "time"}, "power")
        self.assertTrue(all(o.skill.startswith("physics.") for o in phys))
        # a physics goal is unreachable from finance knowns
        self.assertIsNone(syn.synthesize({"price", "discount_pct"}, "power"))

    # --- ambiguity: a goal with two distinct shortest derivations is under-specified ---
    def test_ambiguity_detection(self):
        ops = [syn.Op("a.x", ("p",), "g"), syn.Op("a.y", ("p",), "g")]  # two 1-op routes to g
        self.assertEqual(len(syn.shortest_paths({"p"}, "g", ops)), 2)
        self.assertIsNone(syn.synthesize({"p"}, "g", ops, unique=True))   # declines
        self.assertIsNotNone(syn.synthesize({"p"}, "g", ops, unique=False))  # picks one

    def test_already_known_goal_is_zero_ops(self):
        self.assertEqual(syn.synthesize({"final_price"}, "final_price"), [])

    def test_no_path_declines(self):
        self.assertIsNone(syn.synthesize({"price", "discount_pct"}, "final_price"))

    def test_memoized_search_runs_once(self):
        memo = {}
        before = syn._searches
        syn.synthesize({"price", "discount_pct", "tax_pct"}, "final_price", memo=memo)
        syn.synthesize({"price", "discount_pct", "tax_pct"}, "final_price", memo=memo)
        self.assertEqual(syn._searches - before, 1)


if __name__ == "__main__":
    unittest.main()
