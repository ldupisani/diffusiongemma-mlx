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
    def test_happy_path_rewrites_routes_and_passes_number_check(self):
        # One model call (rewrite); "five" in the original matches "5" in the rewrite.
        shot = _Shot("area of a circle with radius 5")
        r = reformulate_with_model("if a circle's radius is five, how big is it inside?", shot=shot)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "78.54")
        self.assertEqual(len(shot.calls), 1)           # rewrite only; verify is deterministic
        self.assertEqual(model_calls(r), 1)            # coded orchestrator adds 0
        self.assertEqual(r.data["rewrite"], "area of a circle with radius 5")

    def test_pregate_skips_conversational_turns(self):
        # No digit, number word, or domain keyword -> never spend a model call.
        shot = _Shot("should not be called")
        for q in ["write me a haiku about the sea", "thanks, that's helpful!",
                  "what do you think about that?", "tell me more"]:
            with self.subTest(q=q):
                self.assertIsNone(reformulate_with_model(q, shot=shot))
        self.assertEqual(len(shot.calls), 0)

    def test_rewrite_none_declines(self):
        # Passes the pre-gate (has a number), but the rewrite bails with NONE.
        shot = _Shot("NONE")
        r = reformulate_with_model("do something artful with 240", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)

    def test_rewrite_that_does_not_route_declines(self):
        shot = _Shot("explain quantum gravity")
        r = reformulate_with_model("what's the deal with the 5 forces of nature?", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)

    def test_number_drift_declines(self):
        # Rewrite routes and answers, but it INVENTED a number (15) absent from the original —
        # the deterministic guard rejects it (no extra model call).
        shot = _Shot("15% of 240")
        r = reformulate_with_model("something about 240, give me a percentage", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)

    def test_changed_quantity_declines(self):
        # Classic drift: original says radius 5, rewrite says 6 -> rejected.
        shot = _Shot("area of a circle with radius 6")
        r = reformulate_with_model("area of my circle, radius is 5", shot=shot)
        self.assertIsNone(r)

    def test_empty_rewrite_declines(self):
        shot = _Shot("")
        r = reformulate_with_model("what is 5 of anything", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)

    def test_extracts_tagged_rewrite_ignoring_trailing_explanation(self):
        # The model often appends an explanation after the rewrite; the </q> tag bounds it.
        shot = _Shot("area of a circle with radius 5</q> To find this, use pi*r^2 ...")
        r = reformulate_with_model("how much room is inside a circle of radius 5?", shot=shot)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "78.54")
        self.assertEqual(r.data["rewrite"], "area of a circle with radius 5")

    def test_tagged_none_declines(self):
        shot = _Shot("NONE</q>")
        r = reformulate_with_model("the area of a poem with 3 stanzas", shot=shot)
        self.assertIsNone(r)
        self.assertEqual(len(shot.calls), 1)

    def test_number_preserving_rewrite_accepted(self):
        shot = _Shot("convert 100 F to Celsius")
        r = reformulate_with_model("how many C is 100 fahrenheit?", shot=shot)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "37.78°C")


if __name__ == "__main__":
    unittest.main()
