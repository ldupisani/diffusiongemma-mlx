"""Cryptography — modular exponentiation/inverse, Euler's totient, RSA, Diffie–Hellman,
Caesar cipher. Coded, 0 calls."""
from reasoning.bind import bind_crypto
from reasoning.compiled import dispatch

META = {
    "id": "cryptography", "query_type": "cryptography",
    "description": "cryptographic primitives: modular exponentiation, modular inverse, Euler's "
                   "totient φ, RSA key/encrypt/decrypt, Diffie–Hellman shared secret, Caesar cipher",
    "example_queries": [
        "What is the Euler totient of 12?",
        "What is the modular inverse of 3 mod 11?",
        "What is the RSA private key for p=61, q=53, e=17?",
        "What is the Diffie-Hellman secret for g=5, p=23, a=4, b=3?",
    ],
    "keywords": ["rsa", "diffie-hellman", "caesar", "totient", "modular inverse", "cryptography"],
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_crypto(query)[0]


def run(query):
    return dispatch(bind_crypto(query), parse_message="could not parse the cryptography query")
