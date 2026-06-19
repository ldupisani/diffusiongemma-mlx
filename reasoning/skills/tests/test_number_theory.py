import unittest

from reasoning import skills
from reasoning.skills import number_theory as nt

from reasoning.skills.tests._helpers import check_examples


class TestNumberTheory(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("numtheory."):
                check_examples(self, skill)

    def test_prime_edge_cases(self):
        self.assertFalse(nt.is_prime(0))
        self.assertFalse(nt.is_prime(1))
        self.assertTrue(nt.is_prime(2))
        self.assertTrue(nt.is_prime(7919))  # a known prime

    def test_prime_factorization_of_prime_is_itself(self):
        self.assertEqual(nt.prime_factorization(13), "13")

    def test_divisor_count_perfect_square(self):
        self.assertEqual(nt.divisor_count(36), 9)  # 1,2,3,4,6,9,12,18,36

    def test_combinations_symmetry(self):
        self.assertEqual(nt.combinations(8, 3), nt.combinations(8, 5))


if __name__ == "__main__":
    unittest.main()
