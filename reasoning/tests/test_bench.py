"""Unit tests for the GSM8K benchmark harness. Uses ONLY synthetic inputs — no GSM8K
question text is committed (the firewall: benchmark questions live in the gitignored
data dir, never in version control)."""
from reasoning import bench


def test_gold_answer_extracts_after_hash():
    assert bench.gold_answer("blah blah\n#### 18") == 18.0
    assert bench.gold_answer("steps...\n#### 1,024") == 1024.0       # thousands separator
    assert bench.gold_answer("neg case\n#### -7.5") == -7.5
    assert bench.gold_answer("no marker here") is None


def test_extract_value_takes_last_number():
    assert bench.extract_value("The area is 18.0 square meters") == 18.0
    assert bench.extract_value("$1,200.00") == 1200.0
    assert bench.extract_value("9 eggs") == 9.0
    assert bench.extract_value("input 5 gives an output of 42") == 42.0   # last, not first
    assert bench.extract_value("no digits") is None
    assert bench.extract_value(None) is None


def test_is_correct_tolerant():
    assert bench.is_correct("the answer is 18.0", 18.0)
    assert bench.is_correct("18", 18.0)
    assert not bench.is_correct("19", 18.0)
    assert not bench.is_correct(None, 18.0)
    assert not bench.is_correct("18", None)


def test_classify_buckets():
    O = bench.Outcome
    assert bench.classify(O(True, 0, "arithmetic", "5"), correct=True) == "correct"
    assert bench.classify(O(True, 0, "arithmetic", "5"), correct=False) == "wrong_answer"
    assert bench.classify(O(True, 0, "x", None), correct=False) == "needs_llm"
    assert bench.classify(O(False, 0, None, None), correct=False) == "no_route"


def test_solve_routes_plain_arithmetic_code_only():
    # A query the engine genuinely owns must solve at 0 model calls.
    o = bench.solve("what is 12 times 8?")
    assert o.matched and o.model_calls == 0
    assert bench.is_correct(o.answer, 96.0)
