import unittest

from reasoning import skills
from reasoning.skills import physics as p

from reasoning.skills.tests._helpers import check_examples


class TestPhysics(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("physics."):
                check_examples(self, skill)

    def test_energy_and_weight_use_g_9_8(self):
        self.assertAlmostEqual(p.potential_energy(5, 10), 490.0)
        self.assertAlmostEqual(p.weight(10), 98.0)

    def test_acceleration_handles_zero_initial(self):
        self.assertEqual(p.acceleration(0, 20, 4), 5.0)


if __name__ == "__main__":
    unittest.main()
