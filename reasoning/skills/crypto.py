"""Cryptography — modular arithmetic primitives and classic ciphers. Zero model calls.

Covers the number-theoretic core of public-key crypto: fast modular exponentiation, modular
inverse, Euler's totient φ, an RSA key/encrypt/decrypt triple, a Diffie–Hellman shared secret,
and the Caesar shift cipher. All exact integer/string arithmetic.
"""

from __future__ import annotations

import string

from .base import register


@register("crypto.mod_exp", "fast modular exponentiation base^exp mod m",
          [((2, 10, 1000), 24), ((4, 13, 497), 445), ((7, 0, 13), 1)])
def mod_exp(base: int, exp: int, m: int) -> int:
    return pow(int(base), int(exp), int(m))


@register("crypto.mod_inverse", "modular inverse of a (mod m): the x with a·x ≡ 1",
          [((3, 11), 4), ((17, 3120), 2753), ((7, 26), 15)])
def mod_inverse(a: int, m: int) -> int:
    return pow(int(a), -1, int(m))


@register("crypto.totient", "Euler's totient φ(n): count of integers in [1,n] coprime to n",
          [((10,), 4), ((7,), 6), ((12,), 4), ((1,), 1)])
def totient(n: int) -> int:
    n = int(n)
    result, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1
    if m > 1:
        result -= result // m
    return result


@register("crypto.rsa_private_key", "RSA private exponent d for primes p,q and public e",
          [((61, 53, 17), 2753), ((11, 13, 7), 103)])
def rsa_private_key(p: int, q: int, e: int) -> int:
    phi = (int(p) - 1) * (int(q) - 1)
    return pow(int(e), -1, phi)


@register("crypto.rsa_encrypt", "RSA ciphertext m^e mod n", [((65, 17, 3233), 2790)])
def rsa_encrypt(m: int, e: int, n: int) -> int:
    return pow(int(m), int(e), int(n))


@register("crypto.rsa_decrypt", "RSA plaintext c^d mod n", [((2790, 2753, 3233), 65)])
def rsa_decrypt(c: int, d: int, n: int) -> int:
    return pow(int(c), int(d), int(n))


@register("crypto.diffie_hellman", "Diffie–Hellman shared secret g^(ab) mod p",
          [((5, 23, 4, 3), 18), ((5, 23, 6, 15), 2)])
def diffie_hellman(g: int, p: int, a: int, b: int) -> int:
    return pow(int(g), int(a) * int(b), int(p))


def _caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch in string.ascii_lowercase:
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif ch in string.ascii_uppercase:
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


@register("crypto.caesar_encrypt", "Caesar cipher: shift each letter forward by k",
          [(("abc", 3), "def"), (("Hello, World!", 3), "Khoor, Zruog!")])
def caesar_encrypt(text: str, k: int) -> str:
    return _caesar(text, int(k))


@register("crypto.caesar_decrypt", "Caesar cipher: shift each letter back by k",
          [(("def", 3), "abc"), (("Khoor, Zruog!", 3), "Hello, World!")])
def caesar_decrypt(text: str, k: int) -> str:
    return _caesar(text, -int(k))
