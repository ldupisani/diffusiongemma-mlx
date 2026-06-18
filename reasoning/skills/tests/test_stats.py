import unittest

from reasoning import skills
from reasoning.skills import stats as s

from reasoning.skills.tests._helpers import check_examples


class TestStats(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("stats."):
                check_examples(self, skill)

    def test_mean_matches_captured_teacher_answer(self):
        # the captured Claude trace answered 17 for the average of 4,8,15,16,42
        self.assertEqual(s.mean([4, 8, 15, 16, 42]), 17.0)

    def test_median_even_length(self):
        self.assertEqual(s.median([1, 2, 3, 4]), 2.5)

    def test_single_element(self):
        self.assertEqual(s.mean([9]), 9.0)
        self.assertEqual(s.range_of([9]), 0)


if __name__ == "__main__":
    unittest.main()
