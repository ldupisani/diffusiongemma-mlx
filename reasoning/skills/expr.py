"""Arithmetic-expression evaluation — full order-of-operations, pure code.

A recursive-descent parser (the same shape as skills/logic.py) over numbers with precedence
`+ -` < `* /` < `^` (right-assoc), unary minus, parentheses, and `sqrt(...)`. One AST,
evaluated directly. This turns compound arithmetic ("(5+3)*2-4", "15% of 200 plus 10" after
the bind normalizes percent-of) from a bail into a model-free answer.
"""

from __future__ import annotations

import math
import re

from .base import register

_TOK = re.compile(r"\d+\.?\d*|sqrt|[-+*/^()]")


def _tokenize(expression: str) -> list[str]:
    return _TOK.findall(expression)


class _Parser:
    """Precedence low→high: (+ -), (* /), ^ (right-assoc), unary -, atom."""

    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def _eat(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        node = self._sum()
        if self.i != len(self.t):
            raise ValueError(f"unexpected token {self._peek()!r}")
        return node

    def _sum(self):
        node = self._term()
        while self._peek() in ("+", "-"):
            node = (self._eat(), node, self._term())
        return node

    def _term(self):
        node = self._power()
        while self._peek() in ("*", "/"):
            node = (self._eat(), node, self._power())
        return node

    def _power(self):
        node = self._unary()
        if self._peek() == "^":
            self._eat()
            return ("^", node, self._power())  # right-associative
        return node

    def _unary(self):
        if self._peek() == "-":
            self._eat()
            return ("neg", self._unary())
        return self._atom()

    def _atom(self):
        p = self._peek()
        if p is None:
            raise ValueError("unexpected end of expression")
        if p == "(":
            self._eat()
            node = self._sum()
            if self._peek() != ")":
                raise ValueError("missing )")
            self._eat()
            return node
        if p == "sqrt":
            self._eat()
            if self._peek() != "(":
                raise ValueError("sqrt needs parentheses")
            self._eat()
            node = self._sum()
            if self._peek() != ")":
                raise ValueError("missing )")
            self._eat()
            return ("sqrt", node)
        if re.fullmatch(r"\d+\.?\d*", p):
            self._eat()
            return ("num", float(p))
        raise ValueError(f"unexpected token {p!r}")


def compile_expr(expression: str):
    """Parse an arithmetic expression to an AST (raises ValueError if malformed)."""
    return _Parser(_tokenize(expression)).parse()


def _eval(ast) -> float:
    op = ast[0]
    if op == "num":
        return ast[1]
    if op == "neg":
        return -_eval(ast[1])
    if op == "sqrt":
        v = _eval(ast[1])
        if v < 0:
            raise ValueError("square root of a negative number")
        return math.sqrt(v)
    a, b = _eval(ast[1]), _eval(ast[2])
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("divide by zero")
        return a / b
    if op == "^":
        return a ** b
    raise ValueError(f"bad node {op}")


def op_count(ast) -> int:
    """Number of binary operators in the AST (routing gate: ≥2 ⇒ compound)."""
    op = ast[0]
    if op == "num":
        return 0
    if op in ("neg", "sqrt"):
        return op_count(ast[1])
    return 1 + op_count(ast[1]) + op_count(ast[2])


@register("expr.evaluate",
          "evaluate a full arithmetic expression (+ - * / ^, parentheses, sqrt)",
          [(("200*0.15 + 10",), 40.0), (("(5+3)*2-4",), 12.0),
           (("2^10",), 1024.0), (("sqrt(144)",), 12.0)])
def evaluate(expression: str) -> float:
    return _eval(compile_expr(expression))
