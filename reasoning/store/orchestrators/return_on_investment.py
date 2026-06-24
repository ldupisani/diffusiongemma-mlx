"""Return on investment % — reuses arith.percent_change (cost → sale). Zero model calls.

ROI is structurally identical to percentage change: (sale − cost) / cost × 100. No new
skill needed — the existing percent_change skill generalizes straight into finance.
"""

from reasoning.bind import bind_roi
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_guarded

META = {
    "id": "return_on_investment",
    "query_type": "return_on_investment",
    "description": "return on investment as a percentage, from cost and sale price",
    "example_queries": [
        "I bought shares for $4000 and sold them for $5200. What is my return on investment as a percentage?",
        "Bought a bike for $300, sold it for $450 — what was my ROI?",
    ],
    "keywords": ["return", "investment", "roi", "bought", "sold", "gain"],
    "model_free": True,
}

PLAN = Plan(
    steps=[
        CodeStep("arith.percent_change", [Inp("old"), Inp("new")], out="pc"),
        CodeStep("arith.round_to", [Step("pc"), Lit(2)], out="r"),
    ],
    answer_ref="r",
    answer_format="{:g}%",
)


def precondition(query):
    q = query.lower()
    return "return on investment" in q or "roi" in q


def run(query):
    return run_guarded(bind_roi(query), PLAN,
                       parse_message="could not find a cost and sale amount")
