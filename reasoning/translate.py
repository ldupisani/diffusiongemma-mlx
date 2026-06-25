"""Hybrid LLM-as-translator (Program-of-Thought) — the path for parse-limited word problems.

difg is deterministic and answers most queries at 0 model calls. GSM8K-style word problems are
*parse-limited*, not depth-limited (see reasoning/bench.py): the engine can't turn prose into a
structured operation. This module uses the model as a TRANSLATOR, not a solver — one call turns the
prose into a tiny arithmetic PROGRAM, and deterministic code EXECUTES it. The model never does the
arithmetic (where LLMs err); the engine's re-execution of the program is what we trust and report,
so the answer is correct-by-construction and auditable. Cost: 1 model call, 0 arithmetic calls.

The executor (`run_program`) is security-critical: it parses with ast and evaluates against an
operator/function WHITELIST — never eval/exec. A returned program can do arithmetic and nothing
else (no attribute access, no imports, no arbitrary calls)."""
from __future__ import annotations

import ast
import operator
import re

# --- the deterministic executor (0 model calls) ---------------------------------------

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"min": min, "max": max, "abs": abs, "round": round, "int": int, "float": float}
_MAX_POW = 1000  # guard against a model emitting 10**10**9 and hanging the evaluator


class ProgramError(Exception):
    """A program was unsafe, malformed, or referenced an undefined name."""


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ProgramError(f"non-numeric constant {node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ProgramError(f"undefined variable {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow) and (abs(right) > _MAX_POW or abs(left) > _MAX_POW):
            raise ProgramError("exponent/base too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise ProgramError("only min/max/abs/round/int/float calls are allowed")
        return _FUNCS[node.func.id](*[_eval(a, env) for a in node.args])
    raise ProgramError(f"unsupported expression: {type(node).__name__}")


# name = expr, or an augmented assignment (name += expr). Compound ops ordered so ** / // match
# before * / /. A leftover <program>/</program> tag line is tolerated (small models re-emit them).
_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*(\*\*|//|[+\-*/%])?=\s*(.+)$")
_TAG = re.compile(r"^</?program>?$")


def run_program(text: str) -> float:
    """Execute an arithmetic program (one `name = expression` per line) and return `answer`
    (or the last assigned variable). Safe: whitelisted AST nodes only, never eval/exec.
    Tolerates augmented assignments (`x += e`) and stray <program> tag lines. Raises
    ProgramError on anything unsafe/malformed/undefined; arithmetic errors propagate."""
    env: dict[str, float] = {}
    last = None
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or _TAG.match(line):
            continue
        m = _ASSIGN.match(line)
        if not m:
            raise ProgramError(f"not an assignment: {line!r}")
        name, op, expr = m.group(1), m.group(2), m.group(3)
        if op:  # augmented: x += e  ->  x = x + (e)
            if name not in env:
                raise ProgramError(f"augmented assignment to undefined {name!r}")
            expr = f"{name} {op} ({expr})"
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as e:
            raise ProgramError(f"syntax error in {expr.strip()!r}: {e}") from e
        env[name] = _eval(tree, env)
        last = name
    if "answer" in env:
        return env["answer"]
    if last is not None:
        return env[last]
    raise ProgramError("empty program (no assignments)")


# --- the translation prompt + parsing -------------------------------------------------

_SYSTEM = (
    "You convert a math word problem into a short arithmetic PROGRAM that a calculator runs.\n"
    "Rules:\n"
    "- One assignment per line: name = expression.\n"
    "- Expressions use only numbers, names you defined on earlier lines, the operators "
    "+ - * / // % ** and parentheses, and the functions min/max/abs/round.\n"
    "- Do NOT compute results yourself — write the expression and let the calculator evaluate it.\n"
    "- The LAST line must assign a variable named `answer`.\n"
    "- Output ONLY the program, wrapped in <program> and </program>. No prose."
)

_EXAMPLES = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
     "in May. How many clips did she sell altogether in April and May?",
     "april = 48\nmay = april / 2\nanswer = april + may"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
     "babysitting. How much did she earn?",
     "rate_per_hour = 12\nminutes = 50\nanswer = rate_per_hour * minutes / 60"),
    ("Betty is saving for a $100 wallet. She has half the money. Her parents give her $15 and "
     "her grandparents twice as much as her parents. How much more does she need?",
     "cost = 100\nbetty = cost / 2\nparents = 15\ngrandparents = 2 * parents\n"
     "answer = cost - (betty + parents + grandparents)"),
]


def build_prompt(question: str) -> str:
    shots = "\n\n".join(f"Q: {q}\n<program>\n{p}\n</program>" for q, p in _EXAMPLES)
    return f"{_SYSTEM}\n\n{shots}\n\nQ: {question}\n<program>\n"


def parse_program(raw: str) -> str:
    """Extract the program body from the model output. Tolerant of a missing opening tag (the
    prompt primes it), a closing tag only, and a MALFORMED opening tag without its '>' (small
    models emit '<program\\n…')."""
    text = raw or ""
    m = re.search(r"<program>?", text)        # opening tag, '>' optional
    if m:
        text = text[m.end():]
    text = re.split(r"</program>?", text, 1)[0]  # closing tag (and anything after), '>' optional
    return text.strip()


def _default_asker(backend: str, *, model: str, timeout: int, max_tokens: int):
    """Build the model-call function for a backend. `claude` → the Claude CLI (offline of the
    sidecar); `sidecar` → the local MLX model via engine.one_shot (the project's own small LLM —
    the truest 'engine + small LLM fallback'). Both return str|None and never raise."""
    if backend == "sidecar":
        from . import engine

        def asker(prompt):
            try:                                  # greedy + fixed seed → reproducible programs
                return engine.one_shot(prompt, max_tokens=max_tokens, temperature=0.0, seed=0).text
            except Exception:
                return None
        return asker
    if backend == "claude":
        from . import teachers
        return lambda p: teachers.ask_claude(p, model=model, timeout=timeout)
    raise ValueError(f"unknown translator backend {backend!r} (use 'claude' or 'sidecar')")


# --- the unused-given check (a label-free confidence signal) --------------------------

_QNUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _values(text: str) -> set:
    out = set()
    for m in _QNUM.finditer(text or ""):
        try:
            out.add(round(float(m.group().replace(",", "")), 6))
        except ValueError:
            pass
    return out


def question_numbers(question: str) -> set:
    """The digit-literal quantities stated in the question (thousands separators stripped)."""
    return _values(question)


def program_numbers(program: str) -> set:
    """The numeric literals actually USED in the program — collected from the AST (ast.Constant),
    so digits inside variable names (`item2`) are never counted, only real constants."""
    nums: set = set()
    for raw in (program or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or _TAG.match(line):
            continue
        m = _ASSIGN.match(line)
        if not m:
            continue
        try:
            tree = ast.parse(m.group(3), mode="eval")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and not isinstance(node.value, bool) \
                    and isinstance(node.value, (int, float)):
                nums.add(round(float(node.value), 6))
    return nums


def unused_givens(question: str, program: str) -> list:
    """Question quantities that never appear as a literal in the program — a label-free smell
    that the translation ignored a given (or invented a relationship in its place). We check
    only question→program: program-introduced constants (60 for minutes→hours, 100 for percent,
    2 for halving) are expected and never flagged. A non-empty result ⇒ low confidence."""
    p = program_numbers(program)
    return sorted(v for v in question_numbers(question) if v not in p)


def solve(question: str, *, asker=None, model: str = "haiku", backend: str = "claude",
          timeout: int = 60, max_tokens: int = 512) -> dict:
    """Translate the word problem to a program (1 model call) and execute it deterministically.
    Returns {answer, program, model_calls, error, unused_givens}. `unused_givens` is a label-free
    confidence signal (non-empty ⇒ a stated quantity went unused ⇒ low confidence). `asker(prompt)
    ->str|None` is injectable for testing; otherwise `backend` picks the model path."""
    if asker is None:
        asker = _default_asker(backend, model=model, timeout=timeout, max_tokens=max_tokens)

    raw = asker(build_prompt(question))
    if not raw:
        return {"answer": None, "program": None, "model_calls": 1,
                "error": "no response from the model", "unused_givens": []}
    program = parse_program(raw)
    unused = unused_givens(question, program)
    try:
        return {"answer": run_program(program), "program": program, "model_calls": 1,
                "error": None, "unused_givens": unused}
    except (ProgramError, ZeroDivisionError, ArithmeticError, ValueError) as e:
        return {"answer": None, "program": program, "model_calls": 1, "error": str(e),
                "unused_givens": unused}
