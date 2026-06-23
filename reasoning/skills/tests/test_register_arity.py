"""The register-time arity guard: catches the single-arg example gotcha at import."""

import unittest

from reasoning.skills.base import _check_example_arity
from inspect import signature


def _sig(fn):
    return signature(fn)


class TestRegisterArity(unittest.TestCase):
    def test_bare_value_rejected(self):
        # the classic gotcha: 'x^2' instead of ('x^2',) for a 1-arg skill
        with self.assertRaises(ValueError):
            _check_example_arity("t", _sig(lambda a: a), [("x^2", "ok")])

    def test_wrong_count_rejected(self):
        with self.assertRaises(ValueError):
            _check_example_arity("t", _sig(lambda a: a), [((1, 2, 3), "ok")])

    def test_correct_single_arg_accepted(self):
        _check_example_arity("t", _sig(lambda a: a), [(("x^2",), "ok")])

    def test_multi_arg_accepted(self):
        _check_example_arity("t", _sig(lambda a, b: a), [((1, 2), 3)])

    def test_optional_arg_range(self):
        # f(a, b=2): 1 or 2 positional values both valid
        _check_example_arity("t", _sig(lambda a, b=2: a), [((1,), 1), ((1, 2), 3)])

    def test_varargs_skipped(self):
        # *args → can't check arity; must not raise
        _check_example_arity("t", _sig(lambda *xs: xs), [((1, 2, 3, 4), 10)])


if __name__ == "__main__":
    unittest.main()
