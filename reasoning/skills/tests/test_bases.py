import unittest

from reasoning import skills
from reasoning.skills import bases as b

from reasoning.skills.tests._helpers import check_examples


class TestBases(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("radix."):
                check_examples(self, skill)

    def test_round_trips(self):
        self.assertEqual(b.convert(b.convert("100", 10, 2), 2, 10), "100")
        self.assertEqual(b.convert(b.convert("3735928559", 10, 16), 16, 10), "3735928559")

    def test_hex_is_uppercase(self):
        self.assertEqual(b.convert("3735928559", 10, 16), "DEADBEEF")

    def test_zero(self):
        self.assertEqual(b.convert("0", 10, 2), "0")


if __name__ == "__main__":
    unittest.main()
