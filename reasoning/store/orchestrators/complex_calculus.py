"""Complex calculus — residues, contour integrals (residue theorem), complex derivatives.
SymPy-backed, coded, zero model calls."""
from reasoning.bind import bind_ccalc
from reasoning.compiled import dispatch

META = {
    "id": "complex_calculus", "query_type": "complex_calculus",
    "description": "complex analysis: residue of f(z) at a pole, contour integral via the "
                   "residue theorem, and the complex derivative",
    "example_queries": [
        "What is the residue of 1/z at 0?",
        "What is the residue of 1/(z^2+1) at i?",
        "What is the contour integral of 1/z around 0?",
        "What is the complex derivative of z^2?",
    ],
    "keywords": ["residue", "contour", "complex derivative", "pole", "analytic"],
    "sub_orchestrators": [], "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_ccalc(query)[0]


def run(query):
    return dispatch(bind_ccalc(query), parse_message="could not parse the complex expression")
