"""try_coded: the code-only ladder the chat REPLs put in front of the LLM (no teacher, no
interactive clarify). Stub `shot`; the routing/answering is real."""

import unittest

from reasoning.solve import try_coded


class _Shot:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, prompt, *, max_tokens=256, quality="fast"):
        self.calls.append(prompt)
        text = self._responses.pop(0) if self._responses else ""
        return type("Resp", (), {"text": text})()


class TestTryCoded(unittest.TestCase):
    def test_direct_route_no_model_calls(self):
        shot = _Shot()  # must not be called
        r = try_coded("15% of 240", shot=shot)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "36")
        self.assertEqual(len(shot.calls), 0)

    def test_compound_split_no_model_calls(self):
        shot = _Shot()
        r = try_coded("area of a circle with radius 5, and 15% of 240", shot=shot)
        self.assertIsNotNone(r)
        self.assertIn("78.54", r.answer)
        self.assertIn("36", r.answer)
        self.assertEqual(len(shot.calls), 0)  # code-first split, no model

    def test_reformulation_path(self):
        shot = _Shot("area of a circle with radius 5")
        r = try_coded("how much room is inside a circle whose radius is five?", shot=shot)
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "78.54")
        self.assertEqual(len(shot.calls), 1)  # rewrite only; the drift check is deterministic

    def test_non_computational_declines(self):
        shot = _Shot("NONE")
        r = try_coded("tell me a joke about cats", shot=shot)
        self.assertIsNone(r)  # caller's LLM handles it
        self.assertEqual(len(shot.calls), 1)


if __name__ == "__main__":
    unittest.main()
