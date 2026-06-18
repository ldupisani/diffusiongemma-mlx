"""Replace a substring within a string — coded, zero model calls.

'replace OLD with NEW in TEXT' → three quoted strings.
"""

from reasoning import engine
from reasoning.bind import bind_replace
from reasoning.compiled import CodeStep, Inp, Plan, run_guarded

META = {
    "id": "replace_text",
    "query_type": "replace_text",
    "description": "replace every occurrence of one substring with another in a string",
    "example_queries": [
        "Replace 'cat' with 'dog' in 'the cat sat on the mat'",
        "In 'hello world', replace 'world' with 'there'",
    ],
    "keywords": ["replace", "substitute", "swap"],
    "sub_orchestrators": [],
    "model_free": True,
}

PLAN = Plan(
    steps=[CodeStep("text.replace", [Inp("text"), Inp("old"), Inp("new")], out="r")],
    answer_ref="r",
)


def precondition(query):
    return "replace" in query.lower() and query.count("'") + query.count('"') >= 6


def run(query):
    return run_guarded(bind_replace(query), PLAN,
                       parse_message="could not find old/new/text as three quoted strings")
