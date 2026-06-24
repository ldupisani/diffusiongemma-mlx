"""Summarize a list of numbers (average / median / sum / min / max / range) — coded.

One orchestrator covers every list-summary query: the bind detects BOTH the numbers and
which summary is wanted, and run() builds a one-step plan around the chosen stats skill.
This is the concept-level replacement for the teacher's fixed add-chain — it works for a
list of any length, with zero model calls.
"""

from reasoning import engine
from reasoning.bind import (bind_linalg, bind_list_statistic, bind_set_operations, bind_tensor,
                           detect_stat_op)
from reasoning.compiled import CodeStep, Inp, Plan, explain, run_plan

META = {
    "id": "list_statistic",
    "query_type": "list_statistic",
    "description": "a summary statistic (average, median, sum, min, max, range) of a list of numbers",
    "example_queries": [
        "What is the average of 4, 8, 15, 16, and 42?",
        "What is the median of 3, 9, 2, 7, and 5?",
        "What is the highest of 12, 45, 7, and 33?",
        "What is the product of 2, 3, 4, and 5?",
        "What is the mode of 4, 7, 4, 9, and 4?",
        "What is the standard deviation of 2, 4, 4, 5, and 7?",
        "What is the geometric mean of 2, 8, and 32?",
    ],
    "keywords": ["average", "mean", "median", "sum", "total", "highest", "lowest",
                 "maximum", "minimum", "range", "mode", "product", "geometric",
                 "deviation", "variance", "standard"],
    "sub_orchestrators": [],
    "model_free": True,
}


def precondition(query):
    # bindability, not just the keyword: "total for 5 shirts at $20 each …" has "total" but no
    # numeric LIST, so it must not claim-and-decline (blocking a more specific orchestrator).
    # Defer "product" queries that belong elsewhere: set theory ("Cartesian product of {1,2}…")
    # and vectors ("dot/cross product of (1,2,3)…").
    return (bind_list_statistic(query)[0]
            and not bind_set_operations(query)[0] and not bind_linalg(query)[0]
            and not bind_tensor(query)[0])  # "outer/kronecker PRODUCT of […]" → tensor_math


def run(query):
    ok, data, via, _bind_calls = bind_list_statistic(query)
    if not ok:
        return explain("could not find a statistic and at least two numbers in the query")
    plan = Plan(steps=[CodeStep(data["skill"], [Inp("nums")], out="r")],
                answer_ref="r", answer_format="{:g}")
    return run_plan(plan, {"nums": data["nums"]}, shot=engine._no_model)
