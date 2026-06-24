"""reformulate_with_model: rewrite an unroutable single question into a coded query, verify,
and answer with code — or decline (→ teacher). Uses a stub `shot` so no model is loaded; the
routing/answering of the rewritten query is real (exercises actual coded orchestrators)."""

import unittest

from reasoning.compiled import model_calls
from reasoning.reformulate import reformulate_with_model


class _Shot:
    """Stand-in for engine.one_shot: returns canned `.text` per call, records call count."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, prompt, *, max_tokens=256, quality="fast"):
        self.calls.append(prompt)
        text = self._responses.pop(0) if self._responses else ""
        return type("Resp", (), {"text": text})()


class TestReformulateWithModel(unittest.TestCase):
    def test_happy_path_rewrites_routes_and_verifies(self):
        shot = _Shot("area of a circle with radius 5", "YES")
        r = reformulate_with_model("if a circle's radius is five, how big is it inside?", shot=shot)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "78.54")
        self.assertEqual(len(shot.calls), 2)          # rewrite + verify
        self.assertEqual(model_calls(r), 2)            # coded orchestrator adds 0
        self.assertEqual(r.data["rewrite"], "area of a circle with radius 5")

    def test_rewrite_none_declines_without_verifying(self):
        shot = _Shot("NONE")
        r = reformulate_with_model("write me a haiku about the sea", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)           # no verify call once the rewrite bails

    def test_rewrite_that_does_not_route_declines_before_verify(self):
        # A grammatical rewrite that no coded orchestrator answers: stop before spending the
        # verify call.
        shot = _Shot("explain quantum gravity", "YES")
        r = reformulate_with_model("what's the deal with gravity at small scales?", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)

    def test_verify_rejection_declines(self):
        # Rewrite routes and answers, but the verifier says it drifted from the original.
        shot = _Shot("15% of 240", "NO")
        r = reformulate_with_model("something about 240 and percentages", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 2)

    def test_empty_rewrite_declines(self):
        shot = _Shot("")
        r = reformulate_with_model("anything", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)

    def test_verify_accepts_on_leading_y(self):
        # Verifier replies verbosely but affirmatively.
        shot = _Shot("convert 100 F to Celsius", "Yes, identical question.")
        r = reformulate_with_model("how many C is 100 fahrenheit?", shot=shot)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "37.78°C")


if __name__ == "__main__":
    unittest.main()
