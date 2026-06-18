"""A question with a single short factual/numeric answer, stabilized by voting.

Replaces "let me double-check my arithmetic" with code: sample N seed-diverse one-shot
answers and take the majority. Best for short answers where the model is noisy.
"""

from reasoning.patterns import self_consistency

META = {
    "id": "numeric_consensus",
    "query_type": "numeric_consensus",
    "description": "a question with a single short factual or numeric answer",
    "example_queries": [
        "How many times does the digit 7 appear from 1 to 100?",
        "What is 17 percent of 240?",
        "How many days are between March 3 and July 15 in the same year?",
    ],
    "keywords": ["many", "percent", "calculate", "total", "sum", "average", "count", "times", "digit"],
    "sub_orchestrators": [],
    # generic catch-all: only matched when no specific orchestrator claims the query, so its
    # broad "many/count/percent" keywords no longer tie against specific coded orchestrators.
    "fallback": True,
}


def precondition(query):
    return True


def run(query):
    return self_consistency(query, n=5)
