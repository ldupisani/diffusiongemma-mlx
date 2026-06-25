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


def test_is_decline():
    assert bench.is_decline(None)
    assert bench.is_decline("(no result: could not find two dates)")
    assert bench.is_decline("  (no result: …)")
    assert not bench.is_decline("18.0 square meters")


def test_classify_buckets():
    O = bench.Outcome
    assert bench.classify(O(True, 0, "arithmetic", "5"), correct=True) == "correct"
    # a confident WRONG number — the real safety bug
    assert bench.classify(O(True, 0, "ratio_proportion", "1666.67"), correct=False) == "wrong_answer"
    # an honest code decline — claimed but bailed
    assert bench.classify(O(True, 0, "days_between_dates", "(no result: …)"), correct=False) == "declined"
    # a model call was attempted (leaky model_free / fallback)
    assert bench.classify(O(True, 1, None, None), correct=False) == "needs_llm"
    assert bench.classify(O(False, 0, None, None), correct=False) == "no_route"


def test_structure_counts_calc_markers():
    one = "She bakes <<16-3=13>>13.\n#### 13"
    two = "a <<16-3-4=9>>9 then <<9*2=18>>18.\n#### 18"
    assert bench.structure(one)["steps"] == 1
    assert bench.structure(one)["depth"] == "single"
    assert bench.structure(two)["steps"] == 2
    assert bench.structure(two)["depth"] == "double"
    assert bench.structure(two)["ops"] == {"-", "*"}
    assert bench.structure("no markers, pure prose\n#### 5")["depth"] == "none"
    deep = "".join(f"<<{i}+1={i+1}>>" for i in range(5)) + "\n#### 5"
    assert bench.structure(deep)["depth"] == "deep(5+)"


def test_solve_routes_plain_arithmetic_code_only():
    # A query the engine genuinely owns must solve at 0 model calls.
    o = bench.solve("what is 12 times 8?")
    assert o.matched and o.model_calls == 0
    assert bench.is_correct(o.answer, 96.0)
