from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_NUM = r"(-?\d+(?:\.\d+)?)"


def _extract_numbers(q: str, count: int) -> list[float] | None:
    """Extract first `count` numbers from query."""
    numbers = re.findall(_NUM, q)
    if len(numbers) < count:
        return None
    return [float(n) for n in numbers[:count]]


def bind_quantum_information(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    # Hilbert space dimension: gate on "hilbert space" + "qubit"
    if "hilbert space" in q and "qubit" in q:
        nums = _extract_numbers(q, 1)
        if nums:
            return ok("quantum_info.hilbert_dimension", ["n"], nums, round=False)

    # Qubits needed: gate on "qubit" + ("needed"/"required" or "represent")
    if "qubit" in q and ("needed" in q or "required" in q or ("represent" in q and "state" in q)):
        nums = _extract_numbers(q, 1)
        if nums:
            return ok("quantum_info.qubits_needed", ["n"], nums, round=False)

    # Grover iterations: gate on "grover" (distinctive)
    if "grover" in q:
        nums = _extract_numbers(q, 1)
        if nums:
            return ok("quantum_info.grover_iterations", ["n"], nums, round=False)

    # State fidelity: gate on "fidelity" + ("state" or "quantum") for conservatism
    if "fidelity" in q and ("state" in q or "quantum" in q):
        nums = _extract_numbers(q, 4)
        if nums and len(nums) == 4:
            return ok("quantum_info.state_fidelity", ["a1", "b1", "a2", "b2"], nums, round=True, places=6)

    # Tensor product dimension: gate on ("tensor product" or "combined system") + dimension/quantum context
    if (("tensor product" in q or ("combined" in q and "system" in q)) and
        ("dimension" in q or "quantum" in q or "hilbert" in q or "qubit" in q)):
        nums = _extract_numbers(q, 2)
        if nums and len(nums) == 2:
            return ok("quantum_info.tensor_product_dimension", ["d1", "d2"], nums, round=False)

    return False, None, "code", 0