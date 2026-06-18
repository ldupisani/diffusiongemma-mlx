import unittest

from reasoning import skills
from reasoning.skills import units as u

from reasoning.skills.tests._helpers import check_examples


class TestUnits(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("units."):
                check_examples(self, skill)

    def test_f_to_c_matches_captured_teacher_answer(self):
        # captured Claude trace answered 37.78°C for 100°F
        self.assertAlmostEqual(u.f_to_c(100), 37.7777778, places=5)

    def test_conversions_are_inverse(self):
        self.assertAlmostEqual(u.c_to_f(u.f_to_c(72)), 72, places=6)
        self.assertAlmostEqual(u.km_to_mi(u.mi_to_km(26.2)), 26.2, places=6)
        self.assertAlmostEqual(u.kg_to_lb(u.lb_to_kg(150)), 150, places=6)

    def test_convert_spans_dimensions(self):
        self.assertAlmostEqual(u.convert(5, "km", "m"), 5000)
        self.assertAlmostEqual(u.convert(2, "gallons", "liters"), 7.570823568)
        self.assertAlmostEqual(u.convert(5, "minutes", "seconds"), 300)
        self.assertAlmostEqual(u.convert(300, "kelvin", "celsius"), 26.85)

    def test_convert_normalizes_aliases_and_is_inverse(self):
        self.assertAlmostEqual(u.convert(u.convert(10, "meters", "feet"), "ft", "m"), 10, places=6)

    def test_convert_rejects_cross_dimension(self):
        with self.assertRaises(ValueError):
            u.convert(5, "km", "kg")           # length vs mass
        with self.assertRaises(ValueError):
            u.convert(5, "celsius", "meters")  # temperature vs length


if __name__ == "__main__":
    unittest.main()
