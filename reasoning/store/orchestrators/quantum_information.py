"""Quantum information science — Hilbert dimensions, qubit requirements, Grover's algorithm,
state fidelity, and tensor products. Coded, 0 calls."""

from reasoning.bind import bind_quantum_information
from reasoning.compiled import dispatch

META = {
    "id": "quantum_information", "query_type": "quantum_information",
    "description": "quantum information: Hilbert space dimensions of multi-qubit systems, "
                   "qubit count for representing classical states, Grover's algorithm iteration count, "
                   "fidelity between pure quantum states, and combined Hilbert space dimensions",
    "example_queries": [
        "What is the Hilbert space dimension of a 3-qubit system?",
        "How many qubits are needed to represent 256 classical states?",
        "How many iterations does Grover's algorithm require for a database of 100 items?",
        "What is the fidelity between states (1, 0) and (0.7071, 0.7071)?",
        "What is the dimension of the combined system of two subsystems with dimensions 2 and 4?",
    ],
    "keywords": ["hilbert space", "qubit", "grover", "fidelity", "tensor product", "combined system", "dimension"],
    "model_free": True,
}


def precondition(query):
    return bind_quantum_information(query)[0]


def run(query):
    return dispatch(bind_quantum_information(query), parse_message="could not parse the quantum information query")