"""Formal-logic skills — propositional evaluation + conditional transforms.

The core is a recursive-descent parser for propositional expressions (NOT/AND/XOR/OR/
IMPLIES/IFF, variables, true/false, parentheses) into an AST. One AST powers the whole
Tier-1 set — tautology, contradiction, satisfiability, equivalence, evaluation, truth
tables — by evaluating over every truth assignment. De Morgan negation is an AST→AST
transform; the conditional transforms (converse/inverse/contrapositive) are clause-level.
"""

from __future__ import annotations

import itertools
import re

from .base import register

_PREC_TOKENS = re.compile(r"<->|->|\(|\)|[A-Za-z]+")
_BIN = {"and", "or", "xor", "implies", "iff"}


def _tokenize(expr: str) -> list[str]:
    return _PREC_TOKENS.findall(expr)


class _Parser:
    """Recursive descent, precedence low→high: IFF, IMPLIES, OR, XOR, AND, NOT, atom."""

    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def _is(self, *names):
        p = self._peek()
        return p is not None and (p.lower() in names or p in names)

    def _eat(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        ast = self._iff()
        if self.i != len(self.t):
            raise ValueError(f"unexpected token {self._peek()!r}")
        return ast

    def _iff(self):
        left = self._implies()
        while self._is("<->", "iff"):
            self._eat()
            left = ("iff", left, self._implies())
        return left

    def _implies(self):
        left = self._or()
        if self._is("->", "implies"):
            self._eat()
            return ("implies", left, self._implies())  # right-associative
        return left

    def _or(self):
        left = self._xor()
        while self._is("or"):
            self._eat()
            left = ("or", left, self._xor())
        return left

    def _xor(self):
        left = self._and()
        while self._is("xor"):
            self._eat()
            left = ("xor", left, self._and())
        return left

    def _and(self):
        left = self._not()
        while self._is("and"):
            self._eat()
            left = ("and", left, self._not())
        return left

    def _not(self):
        if self._is("not"):
            self._eat()
            return ("not", self._not())
        return self._atom()

    def _atom(self):
        p = self._peek()
        if p is None:
            raise ValueError("unexpected end of expression")
        if p == "(":
            self._eat()
            inner = self._iff()
            if self._peek() != ")":
                raise ValueError("missing )")
            self._eat()
            return inner
        if p == ")" or p in ("->", "<->") or p.lower() in _BIN:
            raise ValueError(f"unexpected token {p!r}")
        self._eat()
        if p.lower() == "true":
            return ("const", True)
        if p.lower() == "false":
            return ("const", False)
        return ("var", p)


def compile_expr(expr: str):
    """Parse a propositional expression into an AST (raises ValueError if malformed)."""
    return _Parser(_tokenize(expr)).parse()


def _vars(ast) -> list[str]:
    op = ast[0]
    if op == "var":
        return [ast[1]]
    if op == "const":
        return []
    seen = []
    for child in ast[1:]:
        for v in _vars(child):
            if v not in seen:
                seen.append(v)
    return seen


def _eval(ast, env: dict) -> bool:
    op = ast[0]
    if op == "var":
        return bool(env[ast[1]])
    if op == "const":
        return ast[1]
    if op == "not":
        return not _eval(ast[1], env)
    a, b = ast[1], ast[2]
    if op == "and":
        return _eval(a, env) and _eval(b, env)
    if op == "or":
        return _eval(a, env) or _eval(b, env)
    if op == "xor":
        return _eval(a, env) != _eval(b, env)
    if op == "implies":
        return (not _eval(a, env)) or _eval(b, env)
    if op == "iff":
        return _eval(a, env) == _eval(b, env)
    raise ValueError(f"bad node {op}")


def _rows(ast):
    vs = sorted(_vars(ast))
    for combo in itertools.product([False, True], repeat=len(vs)):
        yield dict(zip(vs, combo))


@register("logic.tautology", "whether a propositional expression is true for all assignments",
          [(("P OR NOT P",), True), (("P AND Q",), False)])
def tautology(expr: str) -> bool:
    ast = compile_expr(expr)
    return all(_eval(ast, env) for env in _rows(ast))


@register("logic.contradiction", "whether an expression is false for all assignments",
          [(("P AND NOT P",), True), (("P OR Q",), False)])
def contradiction(expr: str) -> bool:
    ast = compile_expr(expr)
    return not any(_eval(ast, env) for env in _rows(ast))


@register("logic.satisfiable", "whether some assignment makes the expression true",
          [(("P AND Q",), True), (("P AND NOT P",), False)])
def satisfiable(expr: str) -> bool:
    ast = compile_expr(expr)
    return any(_eval(ast, env) for env in _rows(ast))


@register("logic.equivalent", "whether two expressions have identical truth tables",
          [(("P implies Q", "NOT P OR Q"), True), (("P", "NOT P"), False)])
def equivalent(expr1: str, expr2: str) -> bool:
    a1, a2 = compile_expr(expr1), compile_expr(expr2)
    vs = sorted(set(_vars(a1)) | set(_vars(a2)))
    for combo in itertools.product([False, True], repeat=len(vs)):
        env = dict(zip(vs, combo))
        if _eval(a1, env) != _eval(a2, env):
            return False
    return True


@register("logic.evaluate", "evaluate an expression under a truth assignment (dict)",
          [(("P AND Q", {"P": True, "Q": False}), False), (("true XOR true", {}), False)])
def evaluate(expr: str, values: dict) -> bool:
    return _eval(compile_expr(expr), values)


@register("logic.truth_table", "the truth table of an expression as a formatted string",
          [(("P AND Q",), "P Q | P AND Q\nF F | F\nF T | F\nT F | F\nT T | T")])
def truth_table(expr: str) -> str:
    ast = compile_expr(expr)
    vs = sorted(_vars(ast))
    lines = [" ".join(vs) + " | " + expr]
    for combo in itertools.product([False, True], repeat=len(vs)):
        env = dict(zip(vs, combo))
        row = " ".join("T" if env[v] else "F" for v in vs)
        lines.append(f"{row} | {'T' if _eval(ast, env) else 'F'}")
    return "\n".join(lines)


# --- AST → AST negation (De Morgan) and AST → string -----------------------------------
def _negate(ast):
    op = ast[0]
    if op == "not":
        return ast[1]
    if op == "and":
        return ("or", _negate(ast[1]), _negate(ast[2]))
    if op == "or":
        return ("and", _negate(ast[1]), _negate(ast[2]))
    if op == "const":
        return ("const", not ast[1])
    if op == "var":
        return ("not", ast)
    if op == "xor":
        return ("iff", ast[1], ast[2])
    if op == "implies":
        return ("and", ast[1], _negate(ast[2]))
    if op == "iff":
        return ("xor", ast[1], ast[2])
    raise ValueError(f"bad node {op}")


_SYM = {"and": "AND", "or": "OR", "xor": "XOR", "implies": "->", "iff": "<->"}


def _to_str(ast) -> str:
    op = ast[0]
    if op == "var":
        return ast[1]
    if op == "const":
        return "true" if ast[1] else "false"
    if op == "not":
        inner = ast[1]
        s = _to_str(inner)
        return f"NOT {s}" if inner[0] in ("var", "const", "not") else f"NOT ({s})"
    return f"({_to_str(ast[1])}) {_SYM[op]} ({_to_str(ast[2])})"


@register("logic.de_morgan", "the negation of an expression, pushed inward (De Morgan)",
          [(("A AND B",), "(NOT A) OR (NOT B)"), (("A OR B",), "(NOT A) AND (NOT B)")])
def de_morgan(expr: str) -> str:
    return _to_str(_negate(compile_expr(expr)))


# --- conditional transforms (clause level) --------------------------------------------
def negatable_clause(clause: str) -> bool:
    """Can this clause be cleanly negated? Symbolic (a logic var/expression) or a clause
    with a copula ('X is Y'); plain prose ('it rains') is NOT — bail to the teacher."""
    c = clause.strip()
    if " is " in c or " are " in c:
        return True
    words = c.upper().split()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 ()]*", c)) and (
        len(words) == 1 or any(op in words for op in ("AND", "OR", "XOR", "NOT", "IMPLIES")))


def _neg_clause(clause: str) -> str:
    c = clause.strip()
    if " is " in c:
        return c.replace(" is ", " is not ", 1)
    if " are " in c:
        return c.replace(" are ", " are not ", 1)
    return f"NOT {c}" if len(c.split()) == 1 else f"NOT ({c})"


@register("logic.converse", "the converse of 'if H then C' = 'If C then H'",
          [(("P", "Q"), "If Q then P")])
def converse(hypothesis: str, conclusion: str) -> str:
    return f"If {conclusion} then {hypothesis}"


@register("logic.inverse", "the inverse of 'if H then C' = 'If not-H then not-C'",
          [(("P", "Q"), "If NOT P then NOT Q")])
def inverse(hypothesis: str, conclusion: str) -> str:
    return f"If {_neg_clause(hypothesis)} then {_neg_clause(conclusion)}"


@register("logic.contrapositive", "the contrapositive of 'if H then C' = 'If not-C then not-H'",
          [(("P", "Q"), "If NOT Q then NOT P")])
def contrapositive(hypothesis: str, conclusion: str) -> str:
    return f"If {_neg_clause(conclusion)} then {_neg_clause(hypothesis)}"
