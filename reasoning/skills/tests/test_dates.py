import unittest
from datetime import date

from reasoning import skills
from reasoning.skills import dates as d

from reasoning.skills.tests._helpers import check_examples


class TestDates(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("date."):
                check_examples(self, skill)

    def test_parse_rejects_garbage(self):
        # conservative: unknown formats raise so the caller can fall back to the model
        with self.assertRaises(ValueError):
            d.parse("sometime next week")

    def test_parse_accepts_year_first_separators(self):
        # YYYY/MM/DD and YYYY.MM.DD are unambiguous (year-first) and supported
        self.assertEqual(d.parse("2900/10/10"), date(2900, 10, 10))
        self.assertEqual(d.parse("2024.03.01"), date(2024, 3, 1))

    def test_parse_still_rejects_invalid(self):
        with self.assertRaises(ValueError):
            d.parse("2024/33/01")

    def test_leap_year_february(self):
        # 2024 is a leap year -> Feb has 29 days
        self.assertEqual(d.days_between(date(2024, 2, 28), date(2024, 3, 1)), 2)
        # 2023 is not -> only 1 day across the same boundary
        self.assertEqual(d.days_between(date(2023, 2, 28), date(2023, 3, 1)), 1)

    def test_add_days_crosses_month(self):
        self.assertEqual(d.add_days(date(2024, 1, 20), 15), date(2024, 2, 4))


if __name__ == "__main__":
    unittest.main()
