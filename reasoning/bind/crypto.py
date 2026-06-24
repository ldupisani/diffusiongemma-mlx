"""bind layer — cryptography. Gated on crypto-specific cues (RSA / Diffie–Hellman / Caesar /
totient / modular inverse / modular exponentiation). Plain "a mod b" is left to arithmetic —
modular exponentiation binds only on the explicit phrase, not on a bare "x^y mod z"."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403

_GATE = re.compile(r"rsa|diffie|caesar|totient|\bphi\b|φ|modular (?:inverse|exponentiation)|"
                   r"mod exp|euler", re.IGNORECASE)


def bind_crypto(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    def ok(skill, dims, vals, **extra):
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    if not _GATE.search(query):
        return False, None, "code", 0

    if "caesar" in q:
        tm = re.search(r"""['"]([^'"]+)['"]""", query)
        sm = re.search(r"(?:shift|by)\s*(?:of\s*)?(-?\d+)", q)
        if tm and sm:
            skill = "crypto.caesar_decrypt" if "decrypt" in q else "crypto.caesar_encrypt"
            return ok(skill, ["text", "k"], [tm[1], int(sm[1])], round=False)
        return False, None, "code", 0

    if "diffie" in q:
        d = dict(re.findall(r"\b([gpab])\s*=\s*(\d+)", q))
        if all(k in d for k in "gpab"):
            return ok("crypto.diffie_hellman", ["g", "p", "a", "b"],
                      [int(d["g"]), int(d["p"]), int(d["a"]), int(d["b"])], round=False)
        return False, None, "code", 0

    if "totient" in q or "phi" in q or "φ" in query or "euler" in q:
        m = re.search(r"(?:totient|phi|φ|euler[''’]?s? totient)\s*(?:of\s*|\(\s*)?(\d+)", q)
        if m:
            return ok("crypto.totient", ["n"], [int(m[1])], round=False)
        return False, None, "code", 0

    if "rsa" in q:
        d = dict(re.findall(r"\b([pqendmc])\s*=\s*(\d+)", q))
        if "decrypt" in q and all(k in d for k in "cdn"):
            return ok("crypto.rsa_decrypt", ["c", "d", "n"],
                      [int(d["c"]), int(d["d"]), int(d["n"])], round=False)
        if "encrypt" in q and all(k in d for k in "men"):
            return ok("crypto.rsa_encrypt", ["m", "e", "n"],
                      [int(d["m"]), int(d["e"]), int(d["n"])], round=False)
        if all(k in d for k in "pqe"):
            return ok("crypto.rsa_private_key", ["p", "q", "e"],
                      [int(d["p"]), int(d["q"]), int(d["e"])], round=False)
        return False, None, "code", 0

    if "modular inverse" in q or ("inverse" in q and re.search(r"mod(?:ulo)?", q)):
        m = re.search(r"inverse of\s+(\d+)\s*(?:mod(?:ulo)?|,)\s*(\d+)", q)
        if m:
            return ok("crypto.mod_inverse", ["a", "m"], [int(m[1]), int(m[2])], round=False)
        return False, None, "code", 0

    # crypto.mod_exp exists as a skill (and powers RSA) but is left unbound: a bare "x^y mod z"
    # collides with the arithmetic/expression tier, so modular exponentiation isn't NL-routed.
    return False, None, "code", 0
