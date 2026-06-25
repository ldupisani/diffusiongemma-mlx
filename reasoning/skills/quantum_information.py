"""Quantum information science — Hilbert space dimensions, qubit counts, Grover iterations,
state fidelity, and tensor products of quantum systems. Zero model calls.
Complements quantum_mechanics (qubit gates, measurement, Pauli operators)."""

from __future__ import annotations

import math

from .base import register


@register("quantum_info.hilbert_dimension", "Hilbert space dimension of an n-qubit system = 2^n",
          [((1,), 2), ((2,), 4), ((3,), 8)])
def hilbert_dimension(n: int) -> int:
    return 2 ** n


@register("quantum_info.qubits_needed", "Number of qubits needed to represent N classical states = ceil(log₂(N))",
          [((4,), 2), ((8,), 3), ((10,), 4)])
def qubits_needed(n: int) -> int:
    return math.ceil(math.log2(n))


@register("quantum_info.grover_iterations", "Grover's algorithm optimal iterations for N items = floor((π/4)·√N)",
          [((4,), 1), ((16,), 3), ((100,), 7)])
def grover_iterations(n: int) -> int:
    return math.floor((math.pi / 4) * math.sqrt(n))


@register("quantum_info.state_fidelity", "Fidelity between two pure single-qubit states (a₁,b₁) and (a₂,b₂) = (a₁·a₂ + b₁·b₂)²",
          [((1.0, 0.0, 1.0, 0.0), 1.0),
           ((0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.7071067811865476), 1.0),
           ((1.0, 0.0, 0.0, 1.0), 0.0),
           ((1.0, 0.0, 0.7071067811865476, 0.7071067811865476), 0.5)])
def state_fidelity(a1: float, b1: float, a2: float, b2: float) -> float:
    return round((a1 * a2 + b1 * b2) ** 2, 6)


@register("quantum_info.tensor_product_dimension", "Combined Hilbert space of subsystems with dimensions d₁ and d₂ = d₁·d₂",
          [((2, 4), 8), ((4, 3), 12)])
def tensor_product_dimension(d1: int, d2: int) -> int:
    return d1 * d2