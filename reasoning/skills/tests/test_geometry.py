import unittest

from reasoning import skills
from reasoning.skills import geometry as g

from reasoning.skills.tests._helpers import check_examples


class TestGeometry(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("geometry."):
                check_examples(self, skill)

    def test_pythagorean_triples(self):
        self.assertEqual(g.hypotenuse(3, 4), 5.0)
        self.assertEqual(g.hypotenuse(5, 12), 13.0)
        self.assertEqual(g.rectangle_diagonal(8, 15), 17.0)

    def test_circle_area_matches_captured(self):
        self.assertAlmostEqual(g.circle_area(5), 78.5398, places=4)

    def test_sphere_volume_matches_captured(self):
        self.assertAlmostEqual(g.sphere_volume(4), 268.0826, places=4)


if __name__ == "__main__":
    unittest.main()
