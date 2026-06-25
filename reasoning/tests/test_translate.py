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


def test_default_asker_backends():
    # both known backends build a callable without making a call; unknown raises
    assert callable(translate._default_asker("claude", model="haiku", timeout=5, max_tokens=64))
    assert callable(translate._default_asker("sidecar", model="x", timeout=5, max_tokens=64))
    with pytest.raises(ValueError):
        translate._default_asker("bogus", model="x", timeout=5, max_tokens=64)
