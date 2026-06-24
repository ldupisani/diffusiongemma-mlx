"""Apply a percentage discount then a tax to a price — fully coded, zero model calls.

The vertical slice that proves the thesis: when the coded bind succeeds, this orchestrator
answers with the sidecar cold (shot=_no_model would raise if any step tried the model).
Only if the bind falls back to the model extractor does it touch the sidecar.
"""

from reasoning import engine
from reasoning.bind import bind_discount_tax, bind_inverse_finance, bind_synthesis_finance
from reasoning.compiled import CodeStep, Inp, Lit, Plan, Step, run_plan

META = {
    "id": "percent_discount_tax",
    "query_type": "percent_discount_tax",
    "description": "apply a percentage discount then a tax to a price",
    "example_queries": [
        "What is 17% off $240 then 8% tax?",
        "A $50 shirt is 20% off with 7% sales tax — final price?",
        "$120 jacket, 25% off, plus 9% tax. Final price?",
    ],
    "keywords": ["off", "discount", "tax", "vat", "percent", "price", "%"],
    "model_free": True,
}

# discount → tax → round, all deterministic arithmetic skills.
PLAN = Plan(
    steps=[
        CodeStep("arith.apply_discount", [Inp("price"), Inp("discount_pct")], out="discounted"),
        CodeStep("arith.apply_tax", [Step("discounted"), Inp("tax_pct")], out="taxed"),
        CodeStep("arith.round_money", [Step("taxed"), Lit(2)], out="final"),
    ],
    answer_ref="final",
    answer_format="${:.2f}",
)


def precondition(query):
    # narrow routing: needs a percentage AND a discount/tax cue, so the lexical matcher
    # prefers this over the generic numeric_consensus orchestrator.
    q = query.lower()
    has_pct = "%" in q or "percent" in q
    return (has_pct and ("off" in q or "discount" in q) and ("tax" in q or "vat" in q)
            and not bind_synthesis_finance(query)[0]   # defer the +split chain to synthesis
            and not bind_inverse_finance(query)[0])    # defer "what discount → target" to inverse


def run(query):
    ok, data, via, bind_calls = bind_discount_tax(query)
    if not ok:
        # bind failed even via the model — nothing to compute deterministically
        # custom result (not explain()) to preserve bind_calls: a model-bind that failed
        # may have made model calls, which explain() would report as 0.
        from reasoning.patterns import PatternResult
        return PatternResult(answer="(no result: could not parse a price, discount, and tax)",
                             trace=[{"step": "bind", "via": via, "ok": False},
                                    {"step": "summary", "model_calls": bind_calls}])
    # coded bind → forbid model calls (invariant); model bind → allow them downstream
    shot = engine._no_model if via == "code" else engine.one_shot
    return run_plan(PLAN, data, shot=shot,
                    prelude_trace=[{"step": "bind", "via": via, "model_calls": bind_calls}],
                    prelude_model_calls=bind_calls)
