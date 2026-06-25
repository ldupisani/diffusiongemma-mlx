"""Tests for the hybrid LLM-as-translator. The executor is security-critical, so its safety
boundary (reject anything that isn't whitelisted arithmetic) is tested explicitly. The model
call is stubbed via an injected asker — no network, fully deterministic."""
import pytest

from reasoning import translate


# --- executor: correctness ------------------------------------------------------------

def test_run_program_multi_step():
    prog = "april = 48\nmay = april / 2\nanswer = april + may"
    assert translate.run_program(prog) == 72.0


def test_run_program_uses_answer_var_not_last_line():
    # `answer` wins even if another assignment follows it
    prog = "answer = 5 + 5\nleftover = 1"
    assert translate.run_program(prog) == 10


def test_run_program_falls_back_to_last_when_no_answer():
    assert translate.run_program("x = 2\ny = x * 3") == 6


def test_run_program_functions_and_parens():
    assert translate.run_program("answer = max(3, 7) - min(2, 1) + abs(-4)") == 10
    assert translate.run_program("answer = round((10 / 3), 2)") == 3.33


def test_run_program_ignores_comments_and_blanks():
    assert translate.run_program("# plan\na = 4\n\nanswer = a * 2  # double\n") == 8


def test_run_program_augmented_assignment():
    prog = "total = 0\ntotal += 5\ntotal += 3\nanswer = total"
    assert translate.run_program(prog) == 8
    assert translate.run_program("x = 10\nx *= 3\nanswer = x") == 30


def test_run_program_augmented_to_undefined_rejected():
    with pytest.raises(translate.ProgramError):
        translate.run_program("answer += 5")


def test_run_program_tolerates_stray_tag_lines():
    # small models re-emit a (sometimes malformed) <program tag inside the body
    assert translate.run_program("<program\na = 3\nanswer = a * 2\n</program>") == 6


# --- executor: safety boundary --------------------------------------------------------

@pytest.mark.parametrize("prog", [
    "answer = __import__('os').system('echo hi')",   # no arbitrary calls
    "answer = (1).__class__",                          # no attribute access
    "answer = foo(3)",                                 # non-whitelisted function
    "answer = unknown_var + 1",                        # undefined name
    "answer = 'string'",                               # non-numeric constant
    "answer = 10 ** 100000",                           # exponent guard
    "5 + 5",                                            # not an assignment
    "1bad = 3",                                         # invalid identifier
])
def test_run_program_rejects_unsafe(prog):
    with pytest.raises(translate.ProgramError):
        translate.run_program(prog)


def test_run_program_division_by_zero_propagates():
    with pytest.raises(ZeroDivisionError):
        translate.run_program("answer = 1 / 0")


# --- parsing --------------------------------------------------------------------------

def test_parse_program_strips_tags():
    raw = "sure!\n<program>\na = 1\nanswer = a + 2\n</program>\nhope that helps"
    assert translate.parse_program(raw) == "a = 1\nanswer = a + 2"


def test_parse_program_handles_primed_opening_tag():
    # build_prompt ends with "<program>\n", so the model often returns only the body + close tag
    assert translate.parse_program("a = 3\nanswer = a\n</program>") == "a = 3\nanswer = a"


# --- solve (stubbed model) ------------------------------------------------------------

def test_solve_executes_translated_program():
    asker = lambda _p: "<program>\nx = 6\nanswer = x * 7\n</program>"
    out = translate.solve("six sevens?", asker=asker)
    assert out["answer"] == 42 and out["model_calls"] == 1 and out["error"] is None


def test_solve_reports_error_on_bad_program_without_crashing():
    out = translate.solve("q", asker=lambda _p: "<program>\nanswer = oops\n</program>")
    assert out["answer"] is None and out["error"] and out["model_calls"] == 1


def test_solve_handles_no_response():
    out = translate.solve("q", asker=lambda _p: None)
    assert out["answer"] is None and out["model_calls"] == 1


# --- unused-given check ---------------------------------------------------------------

def test_program_numbers_from_ast_not_var_names():
    # 60 (constant) is counted; the 2 in the variable name `item2` is NOT
    nums = translate.program_numbers("item2 = 50\nanswer = item2 * 60")
    assert nums == {50.0, 60.0}


def test_unused_givens_flags_the_hardhats_error():
    # the real wrong answer from the trace: the model dropped John's literal "6" and invented *2
    q = ("In a truck there are 26 pink, 15 green, and 24 yellow hard hats. Carl takes 4 pink, and "
         "John takes 6 pink and twice as many green as the pink he removed. How many remain?")
    program = ("pink = 26 - 4 - 4 * 2\ngreen = 15 - (4 * 2) / 2\nyellow = 24\n"
               "answer = pink + green + yellow")
    assert translate.unused_givens(q, program) == [6.0]   # "6" is stated but never used


def test_unused_givens_empty_when_all_used():
    q = "Natalia sold 48 clips, then half as many. How many altogether?"
    program = "april = 48\nmay = april / 2\nanswer = april + may"
    assert translate.unused_givens(q, program) == []      # 48 used; "/2" is program-introduced


def test_unused_givens_ignores_thousands_separator():
    q = "He made $100,000 and spent $40,000. How much is left?"
    assert translate.unused_givens(q, "answer = 100000 - 40000") == []


def test_question_numbers_masks_ordinals_and_fractions():
    # ordinals (1st/2nd/3rd) and a 1/2 fraction are not quantities → not required as literals
    q = "After the 1st and 2nd turns, add 1/2 cup. The 3rd box holds 8."
    assert translate.question_numbers(q) == {8.0}
    assert translate.unused_givens(q, "answer = 8") == []


def test_solve_reports_unused_givens():
    asker = lambda _p: "<program>\nx = 5\nanswer = x\n</program>"
    out = translate.solve("a value of 5 and an unused 99", asker=asker)
    assert out["unused_givens"] == [99.0]


# --- self-consistency -----------------------------------------------------------------

def test_solve_consistent_majority_vote():
    # 3 of 5 samples say 10, 2 say 12 → majority 10, agreement 3/5, k model calls
    progs = ["answer = 10", "answer = 10", "answer = 12", "answer = 10", "answer = 12"]
    out = translate.solve_consistent("q", k=5, sampler=lambda _p, i: progs[i])
    assert out["answer"] == 10.0
    assert out["agreement"] == 0.6
    assert out["model_calls"] == 5
    assert out["votes"] == {10.0: 3, 12.0: 2}


def test_solve_consistent_unanimous_is_high_agreement():
    out = translate.solve_consistent("q", k=4, sampler=lambda _p, _i: "answer = 7")
    assert out["answer"] == 7.0 and out["agreement"] == 1.0


def test_solve_consistent_all_unparseable():
    out = translate.solve_consistent("q", k=3, sampler=lambda _p, _i: "not a program")
    assert out["answer"] is None and out["agreement"] == 0.0


# --- query preprocessing (independent, ablatable transforms) ---------------------------

def test_normalize_words_and_separators():
    assert translate._normalize("He has three apples") == "He has 3 apples"
    assert translate._normalize("twice as many") == "2 times as many"
    assert translate._normalize("half the money") == "0.5 times the money"
    assert translate._normalize("made $1,000,000") == "made $1000000"
    # compounds are intentionally NOT folded (would mangle), and substrings are safe
    assert "someone" in translate._normalize("someone")           # \bone\b doesn't fire inside


def test_givens_scaffold_lists_quantities():
    out = translate._givens_scaffold("Carl has 4 pink hats and 6 green hats.")
    assert out.startswith("Carl has 4 pink hats and 6 green hats.")
    assert "Known quantities:" in out and "4 pink hats" in out and "6 green hats" in out


def test_question_isolation_appends_asked_sentence():
    out = translate._question_isolation("A long setup. How many remain in the box?")
    assert out.endswith("Question: How many remain in the box?")


def test_analysis_prompt_demonstrates_structure_and_parses_program():
    p = translate.build_analysis_prompt("How many are left?")
    # the analysis structure is shown IN the exemplars (consistent format), and a coreference
    # relation ("the red JON removed") is demonstrated to target the traced error class
    assert "Facts:" in p and "Relations:" in p and "Answer type:" in p
    assert "JON removed" in p
    # the program parser still extracts the program from an analyze-style (prose-then-program) reply
    reply = "Facts:\n- a = 6\nRelations:\n- answer = a*7\nAnswer type: a count\n" \
            "<program>\na = 6\nanswer = a * 7\n</program>"
    out = translate.solve("q", asker=lambda _p: reply, style="analyze")
    assert out["answer"] == 42


def test_solve_direct_parses_hash_marker_and_fallback():
    # the model's own answer via the '#### N' marker
    out = translate.solve_direct("q", asker=lambda _p: "step 1... so 6*7.\n#### 42")
    assert out["answer"] == 42.0 and out["model_calls"] == 1
    # thousands separator in the marker
    assert translate.solve_direct("q", asker=lambda _p: "#### 1,024")["answer"] == 1024.0
    # no marker → fall back to the last number in the reply
    assert translate.solve_direct("q", asker=lambda _p: "the total is 18 dollars")["answer"] == 18.0


def test_relationship_cue_detector():
    # fires on the relationship-binding class
    assert translate.has_relationship_cue("twice as many green as the pink he removed")
    assert translate.has_relationship_cue("3 times as much flour as sugar")
    assert translate.has_relationship_cue("half as many clips in May")
    # does NOT fire on frequency/round-trip phrasing or plain problems
    assert not translate.has_relationship_cue("he walks to work three times a week")
    assert not translate.has_relationship_cue("she bought 5 apples and 3 oranges")


def test_style_auto_routes_by_cue():
    seen = {}
    def asker(prompt):
        seen["analyze"] = "Facts:" in prompt
        return "<program>\nanswer = 1\n</program>"
    translate.solve("twice as many X as the Y he took", asker=asker, style="auto")
    assert seen["analyze"] is True                      # relationship → analyze prompt
    translate.solve("she bought 5 apples", asker=asker, style="auto")
    assert seen["analyze"] is False                     # plain → program prompt


def test_preprocess_composes_in_order_and_validates():
    q = "He had three boxes. How many are left?"
    both = translate.preprocess(q, ["normalize", "question"])
    assert "3 boxes" in both and both.rstrip().endswith("How many are left?")
    assert translate.preprocess(q, []) == q                       # baseline = unchanged
    with pytest.raises(ValueError):
        translate.preprocess(q, ["bogus"])


def test_default_asker_backends():
    # both known backends build a callable without making a call; unknown raises
    assert callable(translate._default_asker("claude", model="haiku", timeout=5, max_tokens=64))
    assert callable(translate._default_asker("sidecar", model="x", timeout=5, max_tokens=64))
    with pytest.raises(ValueError):
        translate._default_asker("bogus", model="x", timeout=5, max_tokens=64)
