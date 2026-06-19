"""Arithmetic-expression parser: precedence, associativity, parens, and graceful errors."""

import unittest

from reasoning.skills import expr


class TestExpr(unittest.TestCase):
    def test_precedence_and_assoc(self):
        cases = {
            "2 + 3 * 4": 14,         # * before +
            "(2 + 3) * 4": 20,       # parens override
            "10 - 2 - 3": 5,         # - left-associative
            "2 ^ 3 ^ 2": 512,        # ^ right-associative (2^(3^2)=2^9)
            "100 / 4 / 5": 5,        # / left-associative
            "-3 + 5": 2,             # unary minus
            "2 * -3": -6,            # unary minus as operand
            "sqrt(144) + 1": 13,     # function call
            "(0.15)*200 + 10": 40,   # percent-of normalized form
        }
        for src, want in cases.items():
            with self.subTest(src=src):
                self.assertAlmostEqual(expr.evaluate(src), want)

    def test_op_count(self):
        self.assertEqual(expr.op_count(expr.compile_expr("7 * 8")), 1)
        self.assertEqual(expr.op_count(expr.compile_expr("(5+3)*2-4")), 3)
        self.assertEqual(expr.op_count(expr.compile_expr("sqrt(9)")), 0)

    def test_malformed_raises(self):
        for bad in ["2 +", "(5 + 3", "* 4", "2 3", "sqrt 9"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    expr.evaluate(bad)

    def test_degenerate_raises(self):
        with self.assertRaises(ZeroDivisionError):
            expr.evaluate("5 / 0")
        with self.assertRaises(ValueError):
            expr.evaluate("sqrt(-1)")


if __name__ == "__main__":
    unittest.main()
