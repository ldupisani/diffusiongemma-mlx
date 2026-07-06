"""The world-knowledge resolution loop — fetch a requirement's missing fact, then compute an audited
answer. Never touches reading.solve; arithmetic goes through translate.run_program."""

import unittest

from reasoning import reading_resolve as RR


class TestConvFactorChaining(unittest.TestCase):
    def test_direct_pair(self):
        self.assertEqual(RR._conv_factor("foot", "inch"), 12)

    def test_chained_pair_absent_from_table(self):
        # second→minute→hour is not a direct CONVERSIONS entry; BFS derives 60*60
        self.assertEqual(RR._conv_factor("hour", "second"), 3600)

    def test_same_unit_is_identity(self):
        self.assertEqual(RR._conv_factor("meter", "meter"), 1.0)

    def test_unreachable_is_none(self):
        self.assertIsNone(RR._conv_factor("meter", "gram"))   # no length↔mass edge


class TestResolveLoop(unittest.TestCase):
    def test_part_count_sums_fetched_facts(self):
        r = RR.resolve("What is the total number of legs that Camden's 3 dogs and 2 cats have?")
        self.assertEqual(r["answer"], 20)
        self.assertEqual(r["source"], "world-knowledge")
        self.assertTrue(all(p["source"].startswith("curated") for p in r["provenance"]))

    def test_conversion_chained_factor(self):
        r = RR.resolve("How many seconds are in an hour?")
        self.assertEqual(r["answer"], 3600)
        self.assertEqual(r["provenance"][0]["source"], "derived from CONVERSIONS")

    def test_conversion_with_amount(self):
        self.assertEqual(RR.resolve("How many minutes are in 3 hours?")["answer"], 180)

    def test_solve_passthrough_takes_precedence(self):
        r = RR.resolve("Tom has 3 apples. Mary has twice as many apples as Tom. "
                       "How many apples do they have together?")
        self.assertEqual((r["answer"], r["source"]), (9, "solve"))

    def test_unknown_item_declines_honestly(self):
        r = RR.resolve("How many legs do 3 dinosaurs have?")
        self.assertIsNone(r["answer"])
        self.assertEqual(r["source"], "declined")
        self.assertEqual(r["deficiency"]["kind"], "missing-knowledge")

    def test_injected_provider_fills_the_gap(self):
        def provider(kind, key):
            return (4, "model:retrieved") if key == ("leg", "dinosaur") else None
        r = RR.resolve("How many legs do 3 dinosaurs have?", provider=provider)
        self.assertEqual(r["answer"], 12)
        self.assertEqual(r["provenance"][0]["source"], "model:retrieved")

    def test_answer_is_correct_by_construction(self):
        # the program is what's executed and reported — audit it directly
        r = RR.resolve("How many ounces are in 2 pounds?")
        from reasoning.translate import run_program
        self.assertEqual(run_program(r["program"]), r["answer"])


if __name__ == "__main__":
    unittest.main()
