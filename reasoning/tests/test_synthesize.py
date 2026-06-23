"""Goal-directed plan synthesis, generalized across domains: re-derive hand-written plans,
multi-step chains in new domains, goal-directed op selection, cross-domain isolation (one
shared op pool), ambiguity detection, memoization, and no-path decline — all model-free."""

import unittest

import sympy

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

    # --- cross-domain composition: a geometry cost flows into a finance-style split ---
    def test_cross_domain_geometry_then_split(self):
        pool = syn.GEOMETRY_COST_OPS + syn.CROSS_OPS
        seq = syn.synthesize({"length", "width", "unit_price", "people"}, "cost_per_person",
                             pool, unique=True)
        # area → total_cost → (funnel) cost → /people; the identity step is the cost funnel
        self.assertEqual([op.skill for op in seq],
                         ["geometry.rectangle_area", "arith.multiply", "arith.identity",
                          "arith.divide"])
        r = syn.synthesize_and_run(
            {"length": 4.0, "width": 6.0, "unit_price": 8.0, "people": 3.0},
            "cost_per_person", pool, unique=True)
        self.assertEqual(r.answer, "64")          # (4*6*8)/3
        self.assertEqual(model_calls(r), 0)

    def test_cross_domain_ambiguous_shape_declines(self):
        # two area derivations (rectangle from length+width, square from side) → not unique
        pool = syn.GEOMETRY_COST_OPS + syn.CROSS_OPS
        self.assertIsNone(syn.synthesize(
            {"length", "width", "side", "unit_price", "people"}, "cost_per_person",
            pool, unique=True))

    # --- unit-aware roles: a non-base unit is canonicalized before synthesis ---
    def test_unit_aware_canonicalizes_before_synthesis(self):
        from reasoning.bind import bind_synthesis_physics
        ok, data, *_ = bind_synthesis_physics(
            "How much work to push a 2 kg object accelerating at 3 m/s^2 over 5 km?")
        self.assertTrue(ok)
        self.assertEqual(data["knowns"]["distance"], 5000.0)  # 5 km → 5000 m
        r = syn.synthesize_and_run(data["knowns"], data["goal"], None)
        self.assertEqual(r.answer, "30000")                   # F=2*3=6 N, W=6*5000 J
        self.assertEqual(model_calls(r), 0)

    def test_parse_quantity_is_collision_safe(self):
        from reasoning.skills import units as U
        self.assertEqual(U.parse_quantity("over 5 km", "length"), (5.0, "km"))
        self.assertIsNone(U.parse_quantity("at 5 m/s", "length"))    # m/s is not a length
        self.assertIsNone(U.parse_quantity("at 3 m/s2", "speed"))    # m/s² is not a speed
        self.assertEqual(U.parse_quantity("at 5 m/s", "speed"), (5.0, "mps"))

    # --- multi-goal: several outputs in one plan, shared sub-steps computed once ---
    def test_multi_goal_shares_substeps(self):
        pool = syn.GEOMETRY_COST_OPS + syn.CROSS_OPS
        merged = syn.synthesize_multi({"length", "width", "unit_price"},
                                      ["area", "total_cost"], pool, unique=True)
        # area is produced once and reused by total_cost (not recomputed)
        self.assertEqual([o.skill for o in merged],
                         ["geometry.rectangle_area", "arith.multiply"])
        res = syn.synthesize_multi_and_run(
            {"length": 4.0, "width": 6.0, "unit_price": 8.0},
            ["area", "total_cost"], pool, unique=True)
        self.assertIsNotNone(res)
        r, values = res
        self.assertEqual(values, {"area": 24.0, "total_cost": 192.0})
        self.assertEqual(model_calls(r), 0)

    def test_multi_goal_declines_if_any_unreachable(self):
        self.assertIsNone(syn.synthesize_multi({"length", "width"}, ["area", "total_cost"]))

    # --- inverse synthesis: solve for an INPUT given the output ---
    def test_inverse_finance_discount(self):
        # 250*(1-d/100)*1.08 = 200  →  d = 700/27 ≈ 25.93
        x = syn.inverse_synthesize({"price": 250.0, "tax_pct": 8.0}, "discount_pct",
                                   "final_price", 200.0, syn.FINANCE_OPS)
        self.assertAlmostEqual(x, 700 / 27, places=4)

    def test_inverse_geometry_and_physics(self):
        side = syn.inverse_synthesize({}, "side", "area", 25.0,
                                      [syn.Op("geometry.square_area", ("side",), "area")])
        self.assertEqual(side, 5.0)                       # positive root chosen
        acc = syn.inverse_synthesize({"mass": 3.0}, "acceleration", "force", 12.0, syn.PHYSICS_OPS)
        self.assertEqual(acc, 4.0)

    def test_inverse_declines_non_invertible_op(self):
        # a chain through a table-lookup skill (chem.molar_mass) has no symbolic mirror
        self.assertIsNone(syn.inverse_synthesize({"formula": "H2O"}, "mass", "moles", 1.0,
                                                 syn.CHEM_OPS))

    def test_symbolic_mirrors_match_their_skills(self):
        # the SYMBOLIC mirror of each skill must compute the same value as the skill itself,
        # else inverse synthesis would solve the wrong equation
        from reasoning import skills
        for skill, mirror in syn.SYMBOLIC.items():
            n = mirror.__code__.co_argcount
            args = [2.0, 3.0, 4.0][:n]
            expected = skills.call(skill, *args)
            got = float(mirror(*[sympy.Float(a) for a in args]))
            self.assertAlmostEqual(got, expected, places=6, msg=skill)

    def test_inverse_cross_domain(self):
        # solve a geometry dimension for a target geometry→finance cost: side² × unit_price = 200
        side = syn.inverse_synthesize({"unit_price": 8.0}, "side", "total_cost", 200.0,
                                      syn.GEOMETRY_COST_OPS + syn.CROSS_OPS)
        self.assertEqual(side, 5.0)
        # solve a chemistry mass for a target chem→finance reagent cost
        mass = syn.inverse_synthesize({"price_per_gram": 0.5}, "mass", "reagent_cost", 50.0,
                                      syn.CROSS_OPS)
        self.assertEqual(mass, 100.0)

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
