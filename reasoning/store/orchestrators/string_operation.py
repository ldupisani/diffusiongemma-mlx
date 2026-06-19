"""Single-operand string operations — coded, zero model calls.

One orchestrator covers reverse / case / vowel-char-word counts / title-case / longest-word
/ acronym: the bind detects the operation and the quoted operand, run() applies the matching
skill. (Boolean palindrome and 3-arg replace are separate — different return type / arity.)
"""

from reasoning import engine
from reasoning.bind import bind_string_op, detect_text_op
from reasoning.compiled import CodeStep, Inp, Plan, explain, run_guarded

META = {
    "id": "string_operation",
    "query_type": "string_operation",
    "description": "a single-string operation: reverse, upper/lowercase, title case, "
                   "count vowels/consonants/characters/words, longest word, or acronym",
    "example_queries": [
        "Reverse the string 'hello world'",
        "How many vowels are in 'encyclopedia'?",
        "How many characters are in 'hello world'?",
        "Convert 'hello world' to uppercase",
        "How many words are in 'the quick brown fox'?",
        "How many consonants are in 'python'?",
        "Convert 'HELLO' to lowercase",
        "Capitalize each word in 'the lazy dog'",
        "What is the longest word in 'the quick brown fox'?",
        "What is the acronym of 'North Atlantic Treaty Organization'?",
    ],
    "keywords": ["reverse", "uppercase", "lowercase", "capitalize", "vowels", "consonants",
                 "characters", "word", "words", "longest", "acronym", "string"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    return detect_text_op(query) is not None and ("'" in query or '"' in query)


def run(query):
    ok, data, via, _ = bind_string_op(query)
    if not ok:
        return explain("need one quoted string and a known operation")
    plan = Plan(steps=[CodeStep(data["skill"], [Inp("text")], out="r")], answer_ref="r")
    return run_guarded((ok, data, via, 0), plan, shot=engine._no_model)
