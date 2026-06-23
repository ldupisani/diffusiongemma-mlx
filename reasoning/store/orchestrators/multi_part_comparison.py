"""Compare two or more options across several named dimensions.

Each dimension becomes an independent sub-question; code decomposes, answers each in
one shot, then recombines — replacing the model's "let me weigh each aspect" thinking.
"""

from reasoning.patterns import decompose_solve_combine

META = {
    "id": "multi_part_comparison",
    "query_type": "multi_part_comparison",
    "description": "compare two or more options across several named dimensions",
    "example_queries": [
        "Compare the iPhone and the Pixel on camera, battery life and price",
        "What are the differences between React and Vue for a large application?",
        "Compare a Civic and a Corolla on price, reliability and resale value",
    ],
    "keywords": ["compare", "comparison", "vs", "versus", "difference", "differences", "better"],
    "sub_orchestrators": [],
    "fallback": True,  # a MODEL orchestrator — only when no coded orchestrator claims the query
}


def precondition(query):
    return True


def run(query):
    return decompose_solve_combine(query, max_subq=5, quality="balanced")
