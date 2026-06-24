"""Quantum mechanics — single-qubit states with real amplitudes. Zero model calls.

A qubit α|0⟩ + β|1⟩ is written as the amplitude pair "α, β" (amplitudes may be expressions like
1/sqrt(2)). Supports normalization, Born-rule measurement probabilities, the real single-qubit
gates X / Z / H, and expectation values of the X and Z observables. (Gates/observables with
complex entries — Y, S, T — are omitted to keep amplitudes real.)
"""

from __future__ import annotations

import re

import sympy as sp

from .base import register

_GATES = {
    "X": ((0, 1), (1, 0)),
    "Z": ((1, 0), (0, -1)),
    "H": ((1 / sp.sqrt(2), 1 / sp.sqrt(2)), (1 / sp.sqrt(2), -1 / sp.sqrt(2))),
}


def _num(x: float):
    x = round(float(x), 4)
    return int(x) if x == int(x) else x


def _state(s: str):
    parts = [p for p in re.split(r"[,;]", str(s)) if p.strip()]
    vals = [float(sp.N(sp.sympify(p.replace("^", "**")))) for p in parts]
    if len(vals) < 2:
        raise ValueError("need two amplitudes")
    return vals[0], vals[1]


@register("quantum.normalize", "normalize a qubit state α|0⟩+β|1⟩ given as 'α, β'",
          [(("3, 4",), "0.6, 0.8"), (("1, 1",), "0.7071, 0.7071")])
def normalize(state: str) -> str:
    a, b = _state(state)
    n = (a * a + b * b) ** 0.5
    return f"{_num(a / n)}, {_num(b / n)}"


@register("quantum.measurement_probabilities", "Born-rule P(0), P(1) for a qubit state",
          [(("0.6, 0.8",), "P(0) = 0.36, P(1) = 0.64"), (("1, 1",), "P(0) = 0.5, P(1) = 0.5")])
def measurement_probabilities(state: str) -> str:
    a, b = _state(state)
    n = a * a + b * b
    return f"P(0) = {_num(a * a / n)}, P(1) = {_num(b * b / n)}"


@register("quantum.apply_gate", "apply a single-qubit gate (X, Z, H) to a state",
          [(("X", "1, 0"), "0, 1"), (("H", "1, 0"), "0.7071, 0.7071"),
           (("Z", "0.6, 0.8"), "0.6, -0.8")])
def apply_gate(gate: str, state: str) -> str:
    g = _GATES[gate.strip().upper()]
    a, b = _state(state)
    c = float(g[0][0]) * a + float(g[0][1]) * b
    d = float(g[1][0]) * a + float(g[1][1]) * b
    return f"{_num(c)}, {_num(d)}"


@register("quantum.expectation", "expectation value ⟨P⟩ of a Pauli observable (X or Z)",
          [(("Z", "1, 0"), 1.0), (("Z", "0.6, 0.8"), -0.28), (("X", "0.6, 0.8"), 0.96)])
def expectation(pauli: str, state: str) -> float:
    a, b = _state(state)
    n = a * a + b * b
    p = pauli.strip().upper()
    if p == "Z":
        return round((a * a - b * b) / n, 6)
    if p == "X":
        return round(2 * a * b / n, 6)
    raise ValueError("supported observables: X, Z")
