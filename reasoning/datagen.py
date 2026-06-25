"""Domain dataset generators — produce a domain-relevant dataset and save it as a handle.

Like reasoning.datasets, this is an I/O layer (it writes files), not a pure skill. Each
generator builds rows (reusing the pure domain math where useful, plus numpy/SymPy) and saves
them via datasets.save_rows, returning the new handle's description (schema / row count / head)
— never a text dump. Random generators take a `seed` (default 0) so output is reproducible.

Registry: GENERATORS maps a name -> (function, one-line doc). Each function's signature is
fn(save_as, **params); params arrive as strings from the CLI and are coerced here.
"""

from __future__ import annotations

import math

from . import datasets as ds


def _i(v, default):
    return int(v) if v is not None else default


def _f(v, default):
    return float(v) if v is not None else default


_INT64_MAX = 2 ** 63 - 1


def _fit_int64(rows, col):
    """Cast a column to float if any value exceeds int64 (Arrow can't hold a Python bigint).
    Keeps small sequences exact; large ones become approximate floats rather than crashing."""
    if any(r[col] > _INT64_MAX for r in rows):
        return [tuple(float(v) if i == col else v for i, v in enumerate(r)) for r in rows]
    return rows


# --- number theory & combinatorics -----------------------------------------

def gen_primes(save_as, n=100, **_):
    n = _i(n, 100)
    primes, c = [], 2
    while len(primes) < n:
        if all(c % p for p in primes if p * p <= c):
            primes.append(c)
        c += 1
    return ds.save_rows(save_as, ["index", "prime"], list(enumerate(primes, 1)))


def gen_fibonacci(save_as, n=90, **_):
    n = _i(n, 90)               # F_92 is the largest Fibonacci that fits in int64
    a, b, rows = 0, 1, []
    for i in range(n):
        rows.append((i, a))
        a, b = b, a + b
    return ds.save_rows(save_as, ["index", "value"], _fit_int64(rows, 1))


def gen_factorials(save_as, n=20, **_):
    n = _i(n, 20)               # 20! fits in int64; larger fall back to float
    rows = [(k, math.factorial(k)) for k in range(n + 1)]
    return ds.save_rows(save_as, ["k", "factorial"], _fit_int64(rows, 1))


def gen_partitions(save_as, n=50, **_):
    from .skills.discrete import integer_partitions
    n = _i(n, 50)
    return ds.save_rows(save_as, ["k", "partitions"],
                        [(k, integer_partitions(k)) for k in range(n + 1)])


# --- statistics & probability -----------------------------------------------

def _rng(seed):
    import numpy as np
    return np.random.default_rng(_i(seed, 0))


def gen_normal_sample(save_as, n=1000, mean=0, std=1, seed=0, **_):
    vals = _rng(seed).normal(_f(mean, 0), _f(std, 1), _i(n, 1000))
    return ds.save_rows(save_as, ["value"], [(round(float(v), 6),) for v in vals])


def gen_uniform_sample(save_as, n=1000, lo=0, hi=1, seed=0, **_):
    vals = _rng(seed).uniform(_f(lo, 0), _f(hi, 1), _i(n, 1000))
    return ds.save_rows(save_as, ["value"], [(round(float(v), 6),) for v in vals])


def gen_coin_flips(save_as, n=1000, seed=0, **_):
    vals = _rng(seed).integers(0, 2, _i(n, 1000))
    return ds.save_rows(save_as, ["flip", "heads"], list(enumerate((int(v) for v in vals), 1)))


def gen_dice_rolls(save_as, n=1000, sides=6, seed=0, **_):
    vals = _rng(seed).integers(1, _i(sides, 6) + 1, _i(n, 1000))
    return ds.save_rows(save_as, ["roll", "value"], list(enumerate((int(v) for v in vals), 1)))


def gen_regression(save_as, n=100, slope=2, intercept=1, noise=1, seed=0, **_):
    import numpy as np
    rng = _rng(seed)
    n, m, b, s = _i(n, 100), _f(slope, 2), _f(intercept, 1), _f(noise, 1)
    x = rng.uniform(0, 10, n)
    y = m * x + b + rng.normal(0, s, n)
    return ds.save_rows(save_as, ["x", "y"],
                        [(round(float(a), 6), round(float(c), 6)) for a, c in zip(x, y)])


# --- dynamical systems, numerical & physics ---------------------------------

def gen_logistic_trajectory(save_as, r=3.9, x0=0.5, n=500, **_):
    r, x, rows = _f(r, 3.9), _f(x0, 0.5), []
    for step in range(_i(n, 500)):
        rows.append((step, round(x, 8)))
        x = r * x * (1 - x)
    return ds.save_rows(save_as, ["step", "x"], rows)


def gen_bifurcation(save_as, r_min=2.5, r_max=4.0, r_steps=400, transient=300, samples=20, **_):
    r_min, r_max = _f(r_min, 2.5), _f(r_max, 4.0)
    rs, tr, samp = _i(r_steps, 400), _i(transient, 300), _i(samples, 20)
    rows = []
    for i in range(rs):
        r = r_min + (r_max - r_min) * i / max(rs - 1, 1)
        x = 0.5
        for _t in range(tr):
            x = r * x * (1 - x)
        for _s in range(samp):
            x = r * x * (1 - x)
            rows.append((round(r, 6), round(x, 8)))
    return ds.save_rows(save_as, ["r", "x"], rows)


def gen_sample_function(save_as, expr="sin(x)", a=0, b=6.283185, n=200, **_):
    import sympy as sp
    from sympy.parsing.sympy_parser import (implicit_multiplication_application,
                                            parse_expr, standard_transformations)
    x = sp.Symbol("x")
    f = parse_expr(str(expr).replace("^", "**"), local_dict={"x": x},
                   transformations=standard_transformations + (implicit_multiplication_application,))
    a, b, n = _f(a, 0), _f(b, 6.283185), _i(n, 200)
    rows = []
    for i in range(n):
        xi = a + (b - a) * i / max(n - 1, 1)
        rows.append((round(xi, 6), round(float(f.subs(x, xi)), 6)))
    return ds.save_rows(save_as, ["x", "y"], rows)


def gen_euler_trajectory(save_as, expr="y", x0=0, y0=1, h=0.1, steps=100, **_):
    import sympy as sp
    x, y = sp.symbols("x y")
    f = sp.sympify(str(expr).replace("^", "**"), locals={"x": x, "y": y})
    xi, yi, h = _f(x0, 0), _f(y0, 1), _f(h, 0.1)
    rows = [(round(xi, 6), round(yi, 6))]
    for _s in range(_i(steps, 100)):
        yi += h * float(f.subs({x: xi, y: yi}))
        xi += h
        rows.append((round(xi, 6), round(yi, 6)))
    return ds.save_rows(save_as, ["x", "y"], rows)


def gen_projectile(save_as, v0=20, angle=45, g=9.8, dt=0.1, **_):
    v0, ang, g, dt = _f(v0, 20), math.radians(_f(angle, 45)), _f(g, 9.8), _f(dt, 0.1)
    vx, vy = v0 * math.cos(ang), v0 * math.sin(ang)
    t, x, y, rows = 0.0, 0.0, 0.0, []
    while y >= 0 or t == 0:
        rows.append((round(t, 4), round(x, 4), round(y, 4)))
        t += dt
        x, y = vx * t, vy * t - 0.5 * g * t * t
        if t > 1e5:
            break
    return ds.save_rows(save_as, ["t", "x", "y"], rows)


# --- finance & ML -----------------------------------------------------------

def gen_amortization(save_as, principal=100000, annual_rate=6, years=30, **_):
    P, r = _f(principal, 100000), _f(annual_rate, 6) / 100 / 12
    n = _i(years, 30) * 12
    pay = P * r / (1 - (1 + r) ** -n) if r else P / n
    bal, rows = P, []
    for m in range(1, n + 1):
        interest = bal * r
        princ = pay - interest
        bal = max(bal - princ, 0.0)
        rows.append((m, round(pay, 2), round(interest, 2), round(princ, 2), round(bal, 2)))
    return ds.save_rows(save_as, ["month", "payment", "interest", "principal", "balance"], rows)


def gen_compound_growth(save_as, principal=1000, rate=5, years=30, **_):
    P, rate = _f(principal, 1000), _f(rate, 5) / 100
    rows = [(yr, round(P * (1 + rate) ** yr, 2)) for yr in range(_i(years, 30) + 1)]
    return ds.save_rows(save_as, ["year", "value"], rows)


def gen_classification(save_as, n=200, seed=0, **_):
    rng = _rng(seed)
    n = _i(n, 200)
    rows = []
    for label, (cx, cy) in enumerate([(-2.0, -2.0), (2.0, 2.0)]):
        pts = rng.normal([cx, cy], 1.0, (n // 2, 2))
        rows += [(round(float(px), 6), round(float(py), 6), label) for px, py in pts]
    return ds.save_rows(save_as, ["x1", "x2", "label"], rows)


# --- logistics --------------------------------------------------------------

def gen_eoq_curve(save_as, demand=1000, order_cost=10, holding_cost=2, q_max=None, points=50, **_):
    """EOQ total-cost curve: ordering cost D/Q·S, holding cost Q/2·H, and their sum over a range
    of order quantities Q — the U-shaped total minimized at Q = sqrt(2·D·S/H)."""
    D, S, H = _f(demand, 1000), _f(order_cost, 10), _f(holding_cost, 2)
    eoq = math.sqrt(2 * D * S / H)
    qmax = _f(q_max, round(3 * eoq))
    pts = _i(points, 50)
    rows = []
    for i in range(1, pts + 1):
        q = qmax * i / pts
        order, hold = D / q * S, q / 2 * H
        rows.append((round(q, 4), round(order, 4), round(hold, 4), round(order + hold, 4)))
    return ds.save_rows(save_as, ["order_quantity", "ordering_cost", "holding_cost", "total_cost"],
                        rows)


def gen_inventory_simulation(save_as, periods=52, demand_rate=20, reorder_point=100,
                             order_qty=200, initial=200, **_):
    """Multi-period (Q,R) inventory sim: each period consumes demand, and reorders a fixed
    quantity when stock falls to/below the reorder point."""
    d, rop = _f(demand_rate, 20), _f(reorder_point, 100)
    oq, stock = _f(order_qty, 200), _f(initial, 200)
    rows = []
    for p in range(1, _i(periods, 52) + 1):
        start = stock
        stock = max(stock - d, 0.0)
        ordered = oq if stock <= rop else 0.0
        stock += ordered
        rows.append((p, round(start, 2), round(d, 2), round(ordered, 2), round(stock, 2)))
    return ds.save_rows(save_as, ["period", "starting_stock", "demand", "ordered", "ending_stock"],
                        rows)


GENERATORS = {
    "primes": (gen_primes, "first n primes (--n)"),
    "fibonacci": (gen_fibonacci, "first n Fibonacci numbers (--n)"),
    "factorials": (gen_factorials, "k! for k=0..n (--n)"),
    "partitions": (gen_partitions, "integer partition counts p(k), k=0..n (--n)"),
    "normal_sample": (gen_normal_sample, "n normal samples (--n --mean --std --seed)"),
    "uniform_sample": (gen_uniform_sample, "n uniform samples (--n --lo --hi --seed)"),
    "coin_flips": (gen_coin_flips, "n coin flips (--n --seed)"),
    "dice_rolls": (gen_dice_rolls, "n dice rolls (--n --sides --seed)"),
    "regression": (gen_regression, "synthetic (x,y) regression data (--n --slope --intercept --noise --seed)"),
    "logistic_trajectory": (gen_logistic_trajectory, "logistic-map trajectory (--r --x0 --n)"),
    "bifurcation": (gen_bifurcation, "logistic bifurcation diagram (--r_min --r_max --r_steps)"),
    "sample_function": (gen_sample_function, "y=f(x) sampled over [a,b] (--expr --a --b --n)"),
    "euler_trajectory": (gen_euler_trajectory, "Euler ODE path for dy/dx=f (--expr --x0 --y0 --h --steps)"),
    "projectile": (gen_projectile, "projectile motion over time (--v0 --angle --g --dt)"),
    "amortization": (gen_amortization, "loan amortization schedule (--principal --annual_rate --years)"),
    "compound_growth": (gen_compound_growth, "compound growth series (--principal --rate --years)"),
    "classification": (gen_classification, "two-blob classification dataset (--n --seed)"),
    "eoq_curve": (gen_eoq_curve, "EOQ total-cost curve (--demand --order_cost --holding_cost --q_max --points)"),
    "inventory_simulation": (gen_inventory_simulation,
                             "(Q,R) inventory simulation (--periods --demand_rate --reorder_point --order_qty --initial)"),
}


# Each domain's dataset-creation tools. Every generator belongs to exactly one domain
# (asserted in tests), so "what domains are available" is answerable.
DOMAINS = {
    "number-theory": ["primes", "fibonacci", "factorials", "partitions"],
    "statistics-probability": ["normal_sample", "uniform_sample", "coin_flips", "dice_rolls",
                               "regression", "classification"],
    "dynamics-numerical-physics": ["logistic_trajectory", "bifurcation", "sample_function",
                                   "euler_trajectory", "projectile"],
    "finance": ["amortization", "compound_growth"],
    "logistics": ["eoq_curve", "inventory_simulation"],
}


def list_domains() -> dict:
    """{domain: [(generator, doc), …]} — the dataset-creation tools available per domain."""
    return {dom: [(n, GENERATORS[n][1]) for n in gens] for dom, gens in DOMAINS.items()}


def domain_of(generator: str) -> str | None:
    return next((dom for dom, gens in DOMAINS.items() if generator in gens), None)


def generate(name: str, save_as: str, **params) -> str:
    if name not in GENERATORS:
        raise ValueError(f"unknown generator {name!r} (have: {sorted(GENERATORS)})")
    return GENERATORS[name][0](save_as, **params)
