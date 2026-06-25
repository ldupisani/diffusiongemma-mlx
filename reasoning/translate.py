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
from collections import Counter

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


# Structured-analysis style: the model first lays out facts, relationships, and the expected
# answer type, THEN writes the program. The analysis is demonstrated IN the exemplars (so the
# format is consistent — the lesson from the deterministic-scaffold ablation), and one example
# explicitly resolves a "N times as many X as the Y *he* removed" relationship (the traced error).
_ANALYZE_SYSTEM = (
    "You solve a math word problem in two parts.\n"
    "First ANALYSE it:\n"
    "  Facts: each known quantity and what it refers to.\n"
    "  Relations: how the quantities combine, in words (mind WHO/WHICH each refers to).\n"
    "  Answer type: what kind of value is asked (a count, an amount of money, hours, …).\n"
    "Then write a PROGRAM that computes it:\n"
    "  - one assignment per line, name = expression; only numbers, names defined above, the\n"
    "    operators + - * / // % ** and parentheses, and min/max/abs/round;\n"
    "  - do NOT compute results yourself; the LAST line must assign `answer`;\n"
    "  - wrap ONLY the program in <program> and </program>."
)

_ANALYZE_EXAMPLES = [
    ("A box has 20 red and 30 blue marbles. Tom removes 5 red, and Jon removes 4 red and three "
     "times as many blue as the red he removed. How many marbles are left?",
     "Facts:\n- red = 20, blue = 30\n- Tom removes 5 red\n- Jon removes 4 red\n"
     "- Jon removes blue = 3 times the red JON removed = 3 * 4\n"
     "Relations:\n- red_left = 20 - 5 - 4\n- blue_left = 30 - 3*4\n- answer = red_left + blue_left\n"
     "Answer type: a count of marbles\n"
     "<program>\nred = 20\nblue = 30\ntom_red = 5\njon_red = 4\njon_blue = 3 * jon_red\n"
     "red_left = red - tom_red - jon_red\nblue_left = blue - jon_blue\n"
     "answer = red_left + blue_left\n</program>"),
    ("Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes of babysitting. How "
     "much did she earn?",
     "Facts:\n- rate = $12 per hour\n- time = 50 minutes\n"
     "Relations:\n- 50 minutes = 50/60 hour; earned = rate * hours\n"
     "Answer type: an amount of money\n"
     "<program>\nrate_per_hour = 12\nminutes = 50\nanswer = rate_per_hour * minutes / 60\n</program>"),
]


def build_analysis_prompt(question: str) -> str:
    shots = "\n\n".join(f"Q: {q}\n{a}" for q, a in _ANALYZE_EXAMPLES)
    return f"{_ANALYZE_SYSTEM}\n\n{shots}\n\nQ: {question}\n"


# A comparison/coreference cue — "twice as many", "3 times as much", or "as many … as … he/the
# number". This is the relationship-binding class where the analysis step demonstrably helped
# (the hard-hats error); style="auto" routes ONLY these to the (costlier, riskier) analyze prompt.
_REL_CUE = re.compile(
    r"\b(?:twice|thrice|double|triple|half|\d+(?:\.\d+)?)\s+(?:times\s+)?as\s+(?:many|much)\b"
    r"|\btimes\s+as\s+(?:many|much)\b"
    r"|\bas\s+(?:many|much)\b[^.?!]*\b(?:he|she|they|him|her|them|the\s+number)\b",
    re.IGNORECASE)


def has_relationship_cue(question: str) -> bool:
    """True if the question has a 'N times as many … as …' / coreference comparison — the class
    where the analyze style earned its cost. Used by style='auto'."""
    return bool(_REL_CUE.search(question or ""))


# --- programmatic query preprocessing (0 extra model calls) ----------------------------
# Independent, toggleable transforms applied to the question BEFORE translation. Each is a pure
# str→str function; `preprocess(q, steps)` composes a chosen subset in a fixed order so they can
# be ablated (each alone, every pair, all) via bench.run_ablation — ship only what measures better.

# Standalone small cardinals + "dozen"/"twice"/"half" only. Compounds ("two hundred", "twenty-five")
# are intentionally excluded — naive word→digit would mangle them ("two hundred" → "2 100").
_WORD_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "dozen": "12",
}
_WORD_NUM_RE = re.compile(r"\b(" + "|".join(_WORD_NUM) + r")\b", re.IGNORECASE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_GIVEN_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
_ASK_CUE = re.compile(r"\s*(how many|how much|what|find|calculate|determine|how long)\b", re.IGNORECASE)


def _normalize(q: str) -> str:
    """Canonicalize literals: strip thousands separators, map small number-words to digits, and
    'twice'/'half' to their factors. Conservative — avoids compound words it can't safely fold."""
    q = _THOUSANDS.sub("", q)
    q = re.sub(r"\btwice\b", "2 times", q, flags=re.IGNORECASE)
    q = re.sub(r"\bhalf\b", "0.5 times", q, flags=re.IGNORECASE)
    return _WORD_NUM_RE.sub(lambda m: _WORD_NUM[m.group(0).lower()], q)


def _givens_scaffold(q: str) -> str:
    """Append a 'Known quantities' checklist — every number with the few words that follow it —
    so the model is anchored to use each given (the traced errors all dropped/misused a given)."""
    items = []
    for m in _GIVEN_NUM.finditer(q):
        tail = q[m.end():]
        cut = re.search(r"[\d,.;]", tail)         # stop the label at the next number / clause break
        label = " ".join(re.findall(r"[a-zA-Z]+", (tail[:cut.start()] if cut else tail[:24]))[:3])
        items.append(f"{m.group(0)} {label}".strip())
    return q + "\nKnown quantities: " + "; ".join(items) + "." if items else q


def _question_isolation(q: str) -> str:
    """Append the sentence that states what's asked, so the model solves for the right unknown."""
    sents = _SENT_SPLIT.split(q.strip())
    asked = next((s for s in sents if "?" in s), None) \
        or next((s for s in sents if _ASK_CUE.match(s)), None)
    return q + "\nQuestion: " + asked.strip() if asked else q


PREPROCESS_STEPS = {"normalize": _normalize, "givens": _givens_scaffold,
                    "question": _question_isolation}
PREPROCESS_ORDER = ["normalize", "givens", "question"]  # normalize first; appenders after


def preprocess(question: str, steps) -> str:
    """Apply the named transforms (a subset of PREPROCESS_STEPS) in canonical order. Unknown names
    raise. `steps` empty/None → the question unchanged (the ablation baseline)."""
    steps = set(steps or ())
    unknown = steps - PREPROCESS_STEPS.keys()
    if unknown:
        raise ValueError(f"unknown preprocess step(s): {sorted(unknown)}")
    q = question
    for name in PREPROCESS_ORDER:
        if name in steps:
            q = PREPROCESS_STEPS[name](q)
    return q


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
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)\b", re.IGNORECASE)  # 1st, 2nd … — labels, not quantities
_FRACTION = re.compile(r"\d+\s*/\s*\d+")                       # 1/2 — encoded as division, not a literal


def _values(text: str) -> set:
    out = set()
    for m in _QNUM.finditer(text or ""):
        try:
            out.add(round(float(m.group().replace(",", "")), 6))
        except ValueError:
            pass
    return out


def question_numbers(question: str) -> set:
    """The digit-literal quantities stated in the question (thousands separators stripped). Ordinals
    (1st/2nd…) and explicit fractions (1/2) are masked out: they are labels or division-encoded, not
    quantities the program must echo as a literal — counting them inflates the unused-given flag's
    false-alarm rate (a correct program writes `/2`, not the literal 1 and 2)."""
    text = _FRACTION.sub(" ", _ORDINAL.sub(" ", question or ""))
    return _values(text)


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


# --- direct baseline (the model solves it ITSELF — no deterministic execution) ---------

_DIRECT_SYSTEM = (
    "Solve the math word problem. Reason briefly step by step, then on a NEW LINE write\n"
    "'#### ' followed by ONLY the final numeric answer (digits, no units)."
)
_DIRECT_ANS = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def build_direct_prompt(question: str) -> str:
    return f"{_DIRECT_SYSTEM}\n\nQuestion: {question}\nAnswer:"


def solve_direct(question: str, *, asker=None, model: str = "haiku", backend: str = "claude",
                 timeout: int = 60, max_tokens: int = 512) -> dict:
    """Baseline: the MODEL solves the problem and emits the final number itself (CoT → '#### N').
    No program, no deterministic execution — the model does the arithmetic. Returns {answer,
    model_calls, error}. The contrast to solve(): same model, but arithmetic by tokens vs by code."""
    if asker is None:
        asker = _default_asker(backend, model=model, timeout=timeout, max_tokens=max_tokens)
    raw = asker(build_direct_prompt(question))
    if not raw:
        return {"answer": None, "model_calls": 1, "error": "no response from the model"}
    m = _DIRECT_ANS.search(raw)
    val = None
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            val = None
    if val is None:                                  # no marker → fall back to the last number
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", raw)
        if nums:
            try:
                val = float(nums[-1].replace(",", ""))
            except ValueError:
                val = None
    return {"answer": val, "model_calls": 1, "error": None if val is not None else "no number found"}


# --- self-consistency (a semantic confidence signal) ----------------------------------

def _sampler(backend: str, *, model: str, timeout: int, max_tokens: int, temperature: float):
    """A diverse program sampler `(prompt, i) -> str|None`. The sidecar varies the seed per draw
    (greedy would give identical samples); claude relies on its own sampling per call."""
    if backend == "sidecar":
        from . import engine

        def sample(prompt, i):
            try:
                return engine.one_shot(prompt, max_tokens=max_tokens,
                                       temperature=temperature, seed=i).text
            except Exception:
                return None
        return sample
    if backend == "claude":
        from . import teachers
        return lambda prompt, _i: teachers.ask_claude(prompt, model=model, timeout=timeout)
    raise ValueError(f"unknown translator backend {backend!r} (use 'claude' or 'sidecar')")


def solve_consistent(question: str, *, k: int = 5, sampler=None, backend: str = "sidecar",
                     model: str = "haiku", timeout: int = 60, max_tokens: int = 512,
                     temperature: float = 0.7) -> dict:
    """Self-consistency: draw K programs, execute each, and MAJORITY-VOTE the answer. Returns
    {answer, program, model_calls, agreement, votes, k}. `agreement` ∈ [0,1] is the fraction of
    the K draws that produced the winning value — a label-free confidence signal (low agreement ⇒
    the model is unsure ⇒ the answer may be wrong). The majority vote also tends to beat a single
    greedy draw. `sampler(prompt, i)->str|None` is injectable for testing."""
    if sampler is None:
        sampler = _sampler(backend, model=model, timeout=timeout, max_tokens=max_tokens,
                           temperature=temperature)
    prompt = build_prompt(question)
    samples = []  # (value|None, program)
    for i in range(k):
        program = parse_program(sampler(prompt, i) or "")
        try:
            samples.append((round(float(run_program(program)), 6), program))
        except (ProgramError, ZeroDivisionError, ArithmeticError, ValueError):
            samples.append((None, program))
    counts = Counter(v for v, _ in samples if v is not None)
    if not counts:
        return {"answer": None, "program": samples[0][1] if samples else None,
                "model_calls": k, "agreement": 0.0, "votes": {}, "k": k}
    top_val, top_n = counts.most_common(1)[0]
    program = next(p for v, p in samples if v == top_val)
    return {"answer": top_val, "program": program, "model_calls": k,
            "agreement": top_n / k, "votes": dict(counts), "k": k}


def solve(question: str, *, asker=None, model: str = "haiku", backend: str = "claude",
          timeout: int = 60, max_tokens: int | None = None, preprocess_steps=None,
          style: str = "program") -> dict:
    """Translate the word problem to a program (1 model call) and execute it deterministically.
    Returns {answer, program, model_calls, error, unused_givens}. `style`: "program" (direct) or
    "analyze" (the model first lays out facts/relations/answer-type, then the program — 1 longer
    call). `preprocess_steps` deterministically rewrites the question first (0 extra calls); the
    unused-given check still uses the ORIGINAL question. `asker(prompt)->str|None` is injectable."""
    if style == "auto":   # route only relationship/coreference problems to the analyze prompt
        style = "analyze" if has_relationship_cue(question) else "program"
    builder = build_analysis_prompt if style == "analyze" else build_prompt
    if max_tokens is None:
        max_tokens = 768 if style == "analyze" else 512   # analysis output is longer
    if asker is None:
        asker = _default_asker(backend, model=model, timeout=timeout, max_tokens=max_tokens)

    prompt_q = preprocess(question, preprocess_steps) if preprocess_steps else question
    raw = asker(builder(prompt_q))
    if not raw:
        return {"answer": None, "program": None, "model_calls": 1,
                "error": "no response from the model", "unused_givens": []}
    program = parse_program(raw)
    unused = unused_givens(question, program)   # original question, not the scaffolded one
    try:
        return {"answer": run_program(program), "program": program, "model_calls": 1,
                "error": None, "unused_givens": unused}
    except (ProgramError, ZeroDivisionError, ArithmeticError, ValueError) as e:
        return {"answer": None, "program": program, "model_calls": 1, "error": str(e),
                "unused_givens": unused}
