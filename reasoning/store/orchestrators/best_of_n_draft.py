"""Produce several candidate drafts of a short output and return the best one.

Replaces "let me think of a few options and pick the strongest" with code: generate N
seed-diverse one-shot drafts, then one shot to rank them.
"""

from reasoning.patterns import generate_rank_select

META = {
    "id": "best_of_n_draft",
    "query_type": "best_of_n_draft",
    "description": "produce several candidate drafts and return the best one",
    "example_queries": [
        "Write the best possible tagline for an eco-friendly water bottle",
        "Give me a strong opening line for a data-analyst cover letter",
        "Draft a concise product description for noise-cancelling headphones",
    ],
    "keywords": ["best", "tagline", "slogan", "draft", "headline", "opening", "catchy", "polish", "rewrite"],
    "sub_orchestrators": [],
    "fallback": True,  # a MODEL orchestrator — only when no coded orchestrator claims the query
}


def precondition(query):
    return True


def run(query):
    return generate_rank_select(query, n=3, criteria="quality, creativity and fit to the request")
