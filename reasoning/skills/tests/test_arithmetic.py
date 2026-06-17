import unittest

from reasoning import skills
from reasoning.skills import arithmetic as a

from reasoning.skills.tests._helpers import check_examples


class TestArithmetic(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("arith."):
                check_examples(self, skill)

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            a.divide(1, 0)

    def test_negative_percentage(self):
        # a "negative discount" is a surcharge; the math must still hold
        self.assertAlmostEqual(a.apply_discount(100, -10), 110.0)

    def test_round_money_half_up_boundary(self):
        # the bug guard: banker's rounding would give 2.67 here
        self.assertEqual(a.round_money(2.675, 2), 2.68)
        self.assertEqual(a.round_money(2.5, 0), 3.0)

    def test_jacket_pipeline_exact(self):
        # 17% off $240 then 8% tax, rounded -> $215.14
        discounted = a.apply_discount(240, 17)
        taxed = a.apply_tax(discounted, 8)
        self.assertEqual(a.round_money(taxed, 2), 215.14)


if __name__ == "__main__":
    unittest.main()
