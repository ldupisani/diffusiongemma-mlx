"""Complex numbers — modulus, argument, conjugate, product, quotient, power. Coded, 0 calls."""
from reasoning.bind import bind_complex
from reasoning.compiled import dispatch

META = {
    "id": "complex_numbers", "query_type": "complex_numbers",
    "description": "operations on complex numbers (a+bi): modulus, argument, conjugate, "
                   "multiply, divide, and integer power (De Moivre)",
    "example_queries": [
        "What is the modulus of 3+4i?",
        "What is the argument of 3+4i?",
        "What is the conjugate of 3+4i?",
        "What is (1+1i) raised to the power 3?",
    ],
    "keywords": ["complex", "modulus", "argument", "conjugate", "imaginary", "i"],
    "sub_orchestrators": [], "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_complex(query)[0]


def run(query):
    return dispatch(bind_complex(query), parse_message="could not parse the complex number")
