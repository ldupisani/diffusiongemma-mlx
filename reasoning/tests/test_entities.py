"""The shared entity-extraction layer: correct extraction AND conservative declines."""

import unittest

from reasoning.bind import entities as E


class TestEntities(unittest.TestCase):
    def test_numbers_and_values(self):
        self.assertEqual(E.values("a 3 and -4.5 here"), [3, -4.5])
        ns = E.numbers("x 12")
        self.assertEqual((ns[0].value, ns[0].start, ns[0].end), (12, 2, 4))

    def test_ints_preserve_sign(self):
        self.assertEqual(E.ints("factorial of -3"), [-3])

    def test_only_number(self):
        self.assertEqual(E.only_number("the factorial of 5"), 5)
        self.assertIsNone(E.only_number("gcd of 12 and 8"))     # two → decline
        self.assertIsNone(E.only_number("no numbers here"))     # zero → decline

    def test_exactly(self):
        self.assertEqual(E.exactly("gcd of 12 and 8", 2), [12, 8])
        self.assertIsNone(E.exactly("1. gcd of 12 and 8", 2))   # 3 numbers → decline

    def test_numeric_list_ignores_scattered(self):
        self.assertEqual(E.numeric_list("average of 4, 8, and 15"), [4, 8, 15])
        # words between numbers break the run → not a list
        self.assertEqual(E.numeric_list("total cost of 100 yards at $3"), [100])

    def test_dollars(self):
        self.assertEqual(E.dollars("bought $4000 sold $5200"), [4000.0, 5200.0])
        self.assertEqual(E.only_dollar("a $50 fee and $4000 cost"), None)  # two → decline
        self.assertEqual(E.only_dollar("loan of $20000"), 20000.0)

    def test_percents(self):
        self.assertEqual(E.percents("17% off with 8 percent tax"), [17.0, 8.0])

    def test_count_of(self):
        self.assertEqual(E.count_of("split between 3 people", "people", "friends"), 3)
        self.assertIsNone(E.count_of("split the bill", "people"))

    def test_from_to(self):
        self.assertEqual(E.from_to("from 50 to 75"), (50, 75))
        self.assertIsNone(E.from_to("no range"))

    def test_pairs(self):
        self.assertEqual(E.pairs("between (1, 2) and (4, 6)"), [(1.0, 2.0), (4.0, 6.0)])

    def test_number_before_and_labeled(self):
        self.assertEqual(E.number_before("a mass of 2 kg", r"kg"), 2.0)
        self.assertEqual(E.labeled_number("a circle with radius 5", "radius"), 5.0)


if __name__ == "__main__":
    unittest.main()
