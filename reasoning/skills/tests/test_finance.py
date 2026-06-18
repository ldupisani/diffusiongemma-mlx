import unittest

from reasoning import skills
from reasoning.skills import finance as f

from reasoning.skills.tests._helpers import check_examples


class TestFinance(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("finance."):
                check_examples(self, skill)

    def test_compound_matches_captured_teacher_answer(self):
        # captured Claude trace: $1000 at 5% compounded annually for 3 years -> $1157.63
        self.assertAlmostEqual(f.compound_interest(1000, 5, 3), 1157.625, places=4)

    def test_monthly_beats_annual(self):
        # same nominal rate compounded more often yields more
        annual = f.compound_interest(1000, 5, 3, 1)
        monthly = f.compound_interest(1000, 5, 3, 12)
        self.assertGreater(monthly, annual)

    def test_compound_exceeds_simple(self):
        self.assertGreater(f.compound_interest(1000, 5, 3), f.simple_interest(1000, 5, 3))


if __name__ == "__main__":
    unittest.main()
