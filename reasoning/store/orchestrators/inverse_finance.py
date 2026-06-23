"""Inverse finance — solve for the DISCOUNT that yields a target final price (output known,
input unknown), via symbolic inverse synthesis. Coded, zero model calls."""

from reasoning.bind import bind_inverse_finance
from reasoning.compiled import explain, format_number
from reasoning.patterns import PatternResult
from reasoning.synthesize import FINANCE_OPS, inverse_synthesize

META = {
    "id": "inverse_finance",
    "query_type": "inverse_finance",
    "description": "the discount percentage needed to reach a target final price, given the "
                   "item price and tax (solving backward for an input from the output)",
    "example_queries": [
        "What discount on a $250 jacket gives a final price of $200 with 8% tax?",
        "What discount on a $80 item results in a final price of $60?",
    ],
    "keywords": ["what discount", "discount needed", "final price", "discount to"],
    "sub_orchestrators": [],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_inverse_finance(query)[0]


def run(query):
    ok, data, via, _ = bind_inverse_finance(query)
    if not ok:
        return explain("could not extract an item price, a tax, and a target final price")
    x = inverse_synthesize(data["knowns"], data["unknown"], data["goal"], data["target"],
                           FINANCE_OPS)
    if x is None:
        return explain("no discount reaches that final price")
    return PatternResult(answer=format_number(round(x, 2)) + data["suffix"],
                         trace=[{"step": "inverse_synthesize", "via": "code",
                                 "unknown": data["unknown"], "goal": data["goal"],
                                 "target": data["target"], "output": x},
                                {"step": "summary", "model_calls": 0}])
