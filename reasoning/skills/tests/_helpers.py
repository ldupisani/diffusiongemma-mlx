"""Shared helper: run a skill's example table as assertions (float-tolerant)."""

from __future__ import annotations


def check_examples(test, skill):
    """Assert skill.fn(*args) == expected for every example, with float tolerance."""
    for args, expected in skill.examples:
        got = skill.fn(*args)
        if isinstance(expected, float) and isinstance(got, (int, float)):
            test.assertAlmostEqual(got, expected, places=6,
                                   msg=f"{skill.name}{args}")
        else:
            test.assertEqual(got, expected, msg=f"{skill.name}{args}")
