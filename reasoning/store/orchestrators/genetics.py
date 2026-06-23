"""Genetics — Hardy–Weinberg frequencies, monohybrid (Punnett) cross ratio, allele frequency.
Coded, zero model calls."""

from reasoning import skills
from reasoning.bind import bind_genetics
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "genetics",
    "query_type": "genetics",
    "description": "Hardy–Weinberg genotype frequencies, a monohybrid (Punnett) cross ratio, "
                   "or an allele frequency",
    "example_queries": [
        "What are the Hardy-Weinberg genotype frequencies if the dominant allele frequency is 0.7?",
        "What is the offspring ratio of a cross of Aa and Aa?",
        "What is the allele frequency of 35 out of 50 alleles?",
    ],
    "keywords": ["hardy-weinberg", "hardy weinberg", "allele", "genotype", "cross", "punnett"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_genetics(query)[0]


def run(query):
    ok, data, via, _ = bind_genetics(query)
    if not ok:
        return explain("could not parse the genetics inputs")
    r = skills.call(data["skill"], *data["args"])
    if data["kind"] == "genotypes":
        answer = f"AA: {r[0]}, Aa: {r[1]}, aa: {r[2]}"
    elif data["kind"] == "num":
        answer = str(round(r, 4))
    else:
        answer = str(r)
    return PatternResult(answer=answer,
                         trace=[{"step": data["skill"], "via": "code", "output": str(r)},
                                {"step": "summary", "model_calls": 0}])
