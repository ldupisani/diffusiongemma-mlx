"""Quantum mechanics (single qubit) — normalization, measurement probabilities, gates,
expectation values. Coded, 0 calls."""
from reasoning.bind import bind_quantum
from reasoning.compiled import dispatch

META = {
    "id": "quantum_mechanics", "query_type": "quantum_mechanics",
    "description": "single-qubit quantum mechanics: normalize a state, Born-rule measurement "
                   "probabilities, apply the X/Z/H gates, and expectation values of X/Z",
    "example_queries": [
        "What are the measurement probabilities for the qubit (0.6, 0.8)?",
        "Apply the Hadamard gate to |0>.",
        "What is the expectation value of Z for the state (0.6, 0.8)?",
        "Normalize the qubit state (3, 4).",
    ],
    "keywords": ["qubit", "quantum", "hadamard", "pauli", "measurement", "expectation value"],
    "model_free": True,
}


def precondition(query):
    return bind_quantum(query)[0]


def run(query):
    return dispatch(bind_quantum(query), parse_message="could not parse the quantum query")
