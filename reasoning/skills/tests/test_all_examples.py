"""Every registered skill's example table must be correct — the cheapest broad guard.

Covers all families (incl. trig, coord, clock, sets, probability, ratio, health) in one
pass, and any future family automatically.
"""

import unittest

from reasoning import skills

from reasoning.skills.tests._helpers import check_examples


class TestAllSkillExamples(unittest.TestCase):
    def test_every_skill_example_table(self):
        for name, skill in skills.SKILLS.items():
            with self.subTest(skill=name):
                check_examples(self, skill)


if __name__ == "__main__":
    unittest.main()
