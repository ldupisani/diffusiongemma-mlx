"""End-to-end regression guard for the model-free orchestrators.

Each case routes a query (LexicalMatcher — no model) and runs its orchestrator, asserting
the answer and that ZERO model calls were made. No sidecar needed: a model call would try
to contact it and these must not. This pins the bind + plan + routing for every coded
query type, including the ones grown from captured Claude traces (17, 50%, 37.78°C).
"""

import unittest

from reasoning import router
from reasoning.compiled import model_calls

CASES = [
    # query, expected answer, expected orchestrator id
    ("What is 17% off $240 then 8% tax?", "$215.14", "percent_discount_tax"),
    ("How many days are between 2024-03-03 and 2024-07-15?", "134", "days_between_dates"),
    ("How many times does 'ss' appear in 'mississippi'?", "2", "count_occurrences"),
    ("What is the average of 4, 8, 15, 16, and 42?", "17", "list_statistic"),
    ("What is the median of 3, 9, 2, 7, and 5?", "5", "list_statistic"),
    ("What is the highest of 12, 45, 7, and 33?", "45", "list_statistic"),
    ("What is the percentage increase from 50 to 75?", "50%", "percentage_change"),
    ("What is the percent decrease from 200 to 150?", "-25%", "percentage_change"),
    ("Convert 100 degrees Fahrenheit to Celsius.", "37.78°C", "unit_conversion"),
    ("How many km is 5 miles?", "8.05 km", "unit_conversion"),
    ("Convert 150 pounds to kg.", "68.04 kg", "unit_conversion"),
    ("What is the final balance on $1000 invested at 5% annual interest compounded annually for 3 years?",
     "$1157.63", "compound_interest"),
    # finance family — each pinned to the captured Claude answer
    ("I bought shares for $4000 and sold them for $5200. What is my return on investment as a percentage?",
     "30%", "return_on_investment"),
    ("A product costs $40 to make and sells for $60. What is the profit margin as a percentage?",
     "33.33%", "profit_margin"),
    ("An investment grew from $10000 to $16000 over 4 years. What is the compound annual growth rate?",
     "12.47%", "cagr"),
    ("What is the present value of $10000 received in 5 years at a 4% annual discount rate?",
     "$8219.27", "present_value"),
    ("If I save $200 per month at 5% annual interest for 10 years, how much will I have?",
     "$31056.46", "annuity_future_value"),
    ("What is the monthly payment on a $20000 loan at 6% annual interest over 5 years?",
     "$386.66", "loan_payment"),
    ("A machine costs $15000 with a salvage value of $3000 over 6 years. What is the annual straight-line depreciation?",
     "$2000.00", "straight_line_depreciation"),
    ("Fixed costs are $5000, each unit sells for $25 and costs $15 to make. How many units to break even?",
     "500 units", "break_even"),
    ("A dinner bill is $120. Add an 18% tip and split it among 4 people. How much does each person pay?",
     "$35.40", "tip_split"),
]


class TestOrchestrators(unittest.TestCase):
    def test_coded_orchestrators_reproduce_and_stay_model_free(self):
        for query, expected, orch_id in CASES:
            with self.subTest(query=query):
                m = router.recognize(query)
                self.assertIsNotNone(m, f"no match for {query!r}")
                self.assertEqual(m.orchestrator.id, orch_id)
                r = m.orchestrator.run(query)
                self.assertEqual(r.answer, expected)
                self.assertEqual(model_calls(r), 0, f"{query!r} used the model")


if __name__ == "__main__":
    unittest.main()
