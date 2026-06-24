"""Formal logic — propositional evaluation + conditional transforms. Coded, zero model.

Tier 1 (truth-table evaluation): tautology, contradiction, satisfiability, equivalence,
evaluation, truth tables. Tier 2 (symbolic transforms): De Morgan negation, converse /
inverse / contrapositive. Natural-language syllogisms and non-negatable prose conditionals
fall to the teacher.
"""

from reasoning import skills
from reasoning.bind import bind_logic
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "logic",
    "query_type": "logic",
    "description": "propositional logic (tautology, contradiction, satisfiability, equivalence, "
                   "evaluation, truth table) and conditional transforms (De Morgan, "
                   "converse/inverse/contrapositive)",
    "example_queries": [
        "Is (P OR NOT P) a tautology?",
        "Is (P AND NOT P) a contradiction?",
        "Is (P AND Q) satisfiable?",
        "Are (P implies Q) and (NOT P OR Q) logically equivalent?",
        "Evaluate P AND Q when P is true and Q is false.",
        "What is the negation of (A AND B)?",
        "What is the converse of 'if P then Q'?",
        "What is the contrapositive of 'if P then Q'?",
        "Construct the truth table for P AND Q.",
        "What is true XOR true?",
    ],
    "keywords": ["tautology", "contradiction", "satisfiable", "equivalent", "truth", "table",
                 "negation", "converse", "inverse", "contrapositive", "implies", "evaluate"],
    "model_free": True,
}

# boolean-result kinds → (yes phrasing, no phrasing)
_PHRASE = {
    "tautology": ("Yes, it is a tautology.", "No, it is not a tautology."),
    "contradiction": ("Yes, it is a contradiction.", "No, it is not a contradiction."),
    "satisfiable": ("Yes, it is satisfiable.", "No, it is unsatisfiable."),
    "equivalent": ("Yes, they are logically equivalent.", "No, they are not logically equivalent."),
}


def precondition(query):
    return bind_logic(query)[0]  # bind validates by parsing the expression


def run(query):
    ok, data, via, _ = bind_logic(query)
    if not ok:
        return explain("could not parse a propositional expression or conditional")
    kind, skill = data["kind"], data["skill"]
    val = skills.call(skill, *[data[d] for d in data["dims"]])
    if kind in _PHRASE:
        answer = _PHRASE[kind][0] if val else _PHRASE[kind][1]
    elif kind == "evaluate":
        answer = "True" if val else "False"
    else:  # transform / truth_table — already a string
        answer = val
    return PatternResult(answer=answer,
                         trace=[{"step": skill, "via": "code", "output": val},
                                {"step": "summary", "model_calls": 0}])
