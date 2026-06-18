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
