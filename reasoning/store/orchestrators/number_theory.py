"""Number theory / integer math — coded, zero model calls.

One orchestrator dispatches gcd, lcm, factorial, prime check, prime factorization,
divisor count, combinations/permutations, and perfect-square check. Mixed return types:
int/string ops run through a plan; boolean ops (is_prime, is_perfect_square) phrase Yes/No.
"""

from reasoning import engine, skills
from reasoning.bind import bind_number_theory
from reasoning.compiled import CodeStep, Inp, Plan, explain, run_guarded
from reasoning.patterns import PatternResult

META = {
    "id": "number_theory",
    "query_type": "number_theory",
    "description": "integer math: gcd, lcm, factorial, prime check, prime factorization, "
                   "divisor count, combinations/permutations, perfect-square check",
    "example_queries": [
        "What is the GCD of 24 and 36?",
        "What is the LCM of 4 and 6?",
        "What is the factorial of 5?",
        "Is 17 a prime number?",
        "What is the prime factorization of 60?",
        "How many divisors does 28 have?",
        "What is 8 choose 3?",
        "In how many ways can 5 distinct items be arranged?",
        "In how many ways can 4 people be seated in a row?",
        "Is 100 a perfect square?",
    ],
    "keywords": ["gcd", "lcm", "factorial", "prime", "factorization", "divisors", "choose",
                 "combinations", "permutations", "arranged", "seated", "row", "ways",
                 "perfect", "square"],
    "sub_orchestrators": [],
    "model_free": True,
}

_PHRASE = {
    "numtheory.is_prime": ("prime", "not prime"),
    "numtheory.is_perfect_square": ("a perfect square", "not a perfect square"),
}


def precondition(query):
    return bind_number_theory(query)[0]  # bindability, not just keywords


def run(query):
    ok, data, via, _ = bind_number_theory(query)
    if not ok:
        return explain("could not parse the integers for this operation")
    skill, labels, kind = data["skill"], data["dims"], data["kind"]

    if kind == "bool":  # phrase as a Yes/No sentence (like leap_year / palindrome)
        val = skills.call(skill, *[data[label] for label in labels])
        yes, no = _PHRASE[skill]
        answer = f"Yes, {data['n']} is {yes}." if val else f"No, {data['n']} is {no}."
        return PatternResult(answer=answer,
                             trace=[{"step": skill, "via": "code", "output": val},
                                    {"step": "summary", "model_calls": 0}])

    plan = Plan(steps=[CodeStep(skill, [Inp(label) for label in labels], out="r")],
                answer_ref="r")
    return run_guarded((ok, data, via, 0), plan, shot=engine._no_model)
