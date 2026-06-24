"""Probability (computable subset) — complement, coin runs, die, dice-sum, expected value.

Card/sample-space probabilities that need domain knowledge fall to the teacher; pure
combinatorial counts ("8 choose 3") are handled by number_theory. Coded, zero model calls.
"""

from reasoning.bind import bind_probability
from reasoning.compiled import dispatch

META = {
    "id": "probability",
    "query_type": "probability",
    "description": "complement of a chance, all-heads coin runs, a die face, a two-dice sum, "
                   "or the expected value of a fair die",
    "example_queries": [
        "If there is a 30 percent chance of rain, what is the chance it does not rain?",
        "What is the probability of flipping heads twice in a row?",
        "If you flip a coin 3 times, what is the probability of getting all heads?",
        "What is the probability of rolling a 6 on a fair die?",
        "What is the probability of rolling a sum of 7 with two dice?",
        "What is the expected value of rolling a fair six-sided die?",
    ],
    "keywords": ["probability", "chance", "coin", "heads", "die", "dice", "expected", "roll"],
    "model_free": True,
}


def precondition(query):
    return bind_probability(query)[0]


def run(query):
    return dispatch(bind_probability(query), parse_message="could not parse a computable probability")
