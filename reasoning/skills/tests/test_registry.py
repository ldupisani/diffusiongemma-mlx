"""Invariants that keep the library trustworthy as it grows."""

import re
import unittest
from pathlib import Path

from reasoning import skills

_SKILLS_DIR = Path(skills.__file__).parent
# match real model access (imports of engine, calls to one_shot) — not prose in docstrings
_IMPURE = re.compile(r"^\s*(?:from|import)\s+\S*engine\b|\bone_shot\s*\(", re.MULTILINE)


class TestRegistry(unittest.TestCase):
    def test_every_skill_has_an_example(self):
        # an untested skill is a liability; the example table IS the test data
        missing = [n for n, s in skills.SKILLS.items() if not s.examples]
        self.assertEqual(missing, [], f"skills with no examples: {missing}")

    def test_skills_are_pure_no_model_imports(self):
        # purity invariant: a skill must never reach the model. Forbid engine/one_shot
        # references anywhere under skills/ (excluding this tests/ directory).
        offenders = []
        for py in _SKILLS_DIR.glob("*.py"):
            if _IMPURE.search(py.read_text()):
                offenders.append(py.name)
        self.assertEqual(offenders, [], f"skills must not touch the model: {offenders}")


if __name__ == "__main__":
    unittest.main()
