"""Finance skills — named formulas that subsume repeated arithmetic.

compound_interest replaces the teacher's "multiply by 1.05, again, again" chain: that
chain is hard-wired to 3 years of annual compounding, whereas the formula generalizes to
any term and any compounding frequency in one step.
"""

from __future__ import annotations

from .base import register


@register("finance.compound_interest",
          "the final balance of a principal under compound interest "
          "(principal, annual rate %, years, periods per year)",
          [((1000, 5, 3), 1157.625),            # annual, 3y
           ((1000, 5, 3, 1), 1157.625),
           ((1000, 100, 1, 1), 2000.0)])         # 100% for 1 year doubles
def compound_interest(principal: float, rate_pct: float, years: float,
                      periods_per_year: int = 1) -> float:
    rate = rate_pct / 100 / periods_per_year
    return principal * (1 + rate) ** (periods_per_year * years)


@register("finance.simple_interest",
          "the final balance of a principal under simple interest "
          "(principal, annual rate %, years)",
          [((1000, 5, 3), 1150.0), ((1000, 10, 2), 1200.0)])
def simple_interest(principal: float, rate_pct: float, years: float) -> float:
    return principal * (1 + rate_pct / 100 * years)


@register("finance.present_value",
          "the present value of a future amount discounted at a rate "
          "(future value, annual rate %, years)",
          [((104, 4, 1), 100.0), ((1000, 0, 5), 1000.0)])
def present_value(fv: float, rate_pct: float, years: float,
                  periods_per_year: int = 1) -> float:
    return fv / (1 + rate_pct / 100 / periods_per_year) ** (periods_per_year * years)


@register("finance.cagr",
          "compound annual growth rate % from a begin value to an end value over years",
          [((100, 200, 1), 100.0), ((500, 500, 3), 0.0)])
def cagr(begin: float, end: float, years: float) -> float:
    return ((end / begin) ** (1 / years) - 1) * 100


@register("finance.future_value_annuity",
          "the future value of regular deposits earning compound interest "
          "(deposit per period, annual rate %, years, periods per year)",
          [((100, 0, 1, 12), 1200.0)])  # zero rate → just the deposits
def future_value_annuity(pmt: float, rate_pct: float, years: float,
                         periods_per_year: int = 12) -> float:
    i = rate_pct / 100 / periods_per_year
    m = periods_per_year * years
    if i == 0:
        return pmt * m
    return pmt * (((1 + i) ** m - 1) / i)


@register("finance.loan_payment",
          "the level payment per period to amortize a loan "
          "(principal, annual rate %, years, payments per year)",
          [((1200, 0, 1, 12), 100.0)])  # zero rate → principal spread evenly
def loan_payment(principal: float, rate_pct: float, years: float,
                 periods_per_year: int = 12) -> float:
    i = rate_pct / 100 / periods_per_year
    m = periods_per_year * years
    if i == 0:
        return principal / m
    return principal * i * (1 + i) ** m / ((1 + i) ** m - 1)


@register("finance.margin",
          "profit margin % = (price − cost) / price × 100 (cost, price)",
          [((50, 100), 50.0), ((40, 60), 33.333333)])
def margin(cost: float, price: float) -> float:
    return (price - cost) / price * 100


@register("finance.straight_line_depreciation",
          "annual straight-line depreciation = (cost − salvage) / years (cost, salvage, years)",
          [((15000, 3000, 6), 2000.0), ((1000, 0, 10), 100.0)])
def straight_line_depreciation(cost: float, salvage: float, years: float) -> float:
    return (cost - salvage) / years


@register("finance.break_even_units",
          "units to break even = fixed costs / (price per unit − cost per unit)",
          [((5000, 25, 15), 500.0), ((100, 10, 5), 20.0)])
def break_even_units(fixed: float, price_per_unit: float, cost_per_unit: float) -> float:
    margin = price_per_unit - cost_per_unit
    if margin <= 0:  # defensive: no finite break-even when you don't profit per unit
        raise ValueError("contribution margin must be positive (price must exceed unit cost)")
    return fixed / margin
