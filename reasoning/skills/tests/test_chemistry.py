import unittest

from reasoning import skills
from reasoning.skills import chemistry as c

from reasoning.skills.tests._helpers import check_examples


class TestChemistry(unittest.TestCase):
    def test_example_tables(self):
        for name, skill in skills.SKILLS.items():
            if name.startswith("chem."):
                check_examples(self, skill)

    def test_formula_parser_handles_parentheses(self):
        self.assertEqual(c.parse_formula("Ca(OH)2"), {"Ca": 1, "O": 2, "H": 2})
        self.assertEqual(c.parse_formula("C6H12O6"), {"C": 6, "H": 12, "O": 6})
        self.assertEqual(c.parse_formula("H2SO4"), {"H": 2, "S": 1, "O": 4})

    def test_molar_mass_matches_captured(self):
        self.assertAlmostEqual(c.molar_mass("C6H12O6"), 180.156, places=3)
        self.assertAlmostEqual(c.molar_mass("H2SO4"), 98.072, places=3)

    def test_unknown_element_raises(self):
        with self.assertRaises(ValueError):
            c.molar_mass("Xz2")

    def test_ph_neutral(self):
        self.assertAlmostEqual(c.ph(1e-7), 7.0)


if __name__ == "__main__":
    unittest.main()
